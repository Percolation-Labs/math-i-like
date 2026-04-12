"""Quick ablation probe for multiscale checkpoint.

Tests whether the slow field is contributing by:
1. Normal forward pass → per-chunk losses
2. Zero out slow field modulation → per-chunk losses
3. Zero out ALL field modulation → per-chunk losses
4. Compare deltas: if slow field matters, zeroing it hurts later chunks more.

Run alongside training (inference only, low memory).
"""

import os
import sys
import math
import torch
import tiktoken

sys.path.insert(0, os.path.dirname(__file__))
from model_multiscale import GPT


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    enc = tiktoken.get_encoding("gpt2")
    model = GPT(
        vocab_size=enc.n_vocab,
        d_model=args["d_model"],
        n_head=args["n_head"],
        n_layer=args["n_layer"],
        dropout=0.0,
        max_len=args["context_length"],
        chunk_size=args["chunk_size"],
        evap_rate=args.get("evap_rate", 0.05),
        use_field=True,
        mod_type=args.get("mod_type", "additive"),
        multi_scale=True,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", 0)
    return model, enc, args, step


@torch.no_grad()
def per_chunk_loss(model, x, y, chunk_size):
    """Get loss broken down by chunk position."""
    logits = model(x)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    seq_len = y.shape[1]
    n_chunks = seq_len // chunk_size

    chunk_losses = []
    for c in range(n_chunks):
        start = c * chunk_size
        end = (c + 1) * chunk_size
        chunk_logp = log_probs[:, start:end, :]
        chunk_y = y[:, start:end]
        loss = -chunk_logp.gather(2, chunk_y.unsqueeze(-1)).squeeze(-1).mean()
        chunk_losses.append(loss.item())
    return chunk_losses


@torch.no_grad()
def ablate_slow_field(model):
    """Zero out slow field modulation weights, return originals."""
    originals = {}
    for name, param in model.named_parameters():
        if "w_mod_slow" in name:
            originals[name] = param.data.clone()
            param.data.zero_()
    return originals


@torch.no_grad()
def ablate_all_fields(model):
    """Zero out both fast and slow field modulation weights."""
    originals = {}
    for name, param in model.named_parameters():
        if "w_mod" in name:
            originals[name] = param.data.clone()
            param.data.zero_()
    return originals


def restore_params(model, originals):
    for name, param in model.named_parameters():
        if name in originals:
            param.data.copy_(originals[name])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/owt_multiscale_step5000.pt"
    print(f"Loading {ckpt_path} on {device}...")
    model, enc, args, step = load_model(ckpt_path, device)
    chunk_size = args["chunk_size"]
    ctx_len = args["context_length"]
    print(f"Step {step}, chunk_size={chunk_size}, ctx_len={ctx_len}")

    # Load some validation data
    val_path = os.path.join(os.path.dirname(__file__),
                            ".data/openwebtext_validation_tokens.pt")
    val_tokens = torch.load(val_path, map_location="cpu", weights_only=True)
    n_examples = min(64, len(val_tokens) // ctx_len)
    xs, ys = [], []
    for i in range(n_examples):
        start = i * ctx_len
        seq = val_tokens[start:start + ctx_len + 1]
        xs.append(seq[:-1])
        ys.append(seq[1:])
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    print(f"Eval on {n_examples} sequences of length {ctx_len}\n")

    # Process in small batches to save memory
    batch_size = 4
    n_chunks = ctx_len // chunk_size

    def avg_chunk_losses():
        all_losses = [0.0] * n_chunks
        for b in range(0, n_examples, batch_size):
            bx = x[b:b+batch_size]
            by = y[b:b+batch_size]
            cl = per_chunk_loss(model, bx, by, chunk_size)
            for c in range(n_chunks):
                all_losses[c] += cl[c]
        return [l / (n_examples / batch_size) for l in all_losses]

    # 1. Normal (both fields active)
    print("=== Normal (fast + slow fields active) ===")
    normal_losses = avg_chunk_losses()
    for c, l in enumerate(normal_losses):
        print(f"  Chunk {c}: loss={l:.4f} ppl={math.exp(l):.1f}")
    print(f"  Overall: loss={sum(normal_losses)/len(normal_losses):.4f} "
          f"ppl={math.exp(sum(normal_losses)/len(normal_losses)):.1f}")

    # 2. Ablate slow field only
    print("\n=== Slow field ablated (fast only) ===")
    originals = ablate_slow_field(model)
    slow_ablated = avg_chunk_losses()
    restore_params(model, originals)
    for c, l in enumerate(slow_ablated):
        delta = l - normal_losses[c]
        print(f"  Chunk {c}: loss={l:.4f} ppl={math.exp(l):.1f}  "
              f"delta={delta:+.4f}")
    print(f"  Overall: loss={sum(slow_ablated)/len(slow_ablated):.4f} "
          f"ppl={math.exp(sum(slow_ablated)/len(slow_ablated)):.1f}")

    # 3. Ablate ALL fields
    print("\n=== All fields ablated (no field modulation) ===")
    originals = ablate_all_fields(model)
    all_ablated = avg_chunk_losses()
    restore_params(model, originals)
    for c, l in enumerate(all_ablated):
        delta = l - normal_losses[c]
        print(f"  Chunk {c}: loss={l:.4f} ppl={math.exp(l):.1f}  "
              f"delta={delta:+.4f}")
    print(f"  Overall: loss={sum(all_ablated)/len(all_ablated):.4f} "
          f"ppl={math.exp(sum(all_ablated)/len(all_ablated)):.1f}")

    # 4. Summary
    print("\n=== SUMMARY: Field contribution by chunk ===")
    print(f"{'Chunk':>6} {'Normal':>8} {'Slow Δ':>8} {'All Δ':>8} {'Slow/All%':>10}")
    for c in range(n_chunks):
        slow_d = slow_ablated[c] - normal_losses[c]
        all_d = all_ablated[c] - normal_losses[c]
        pct = (slow_d / all_d * 100) if all_d > 0.0001 else 0
        print(f"  {c:>4}   {normal_losses[c]:.4f}   {slow_d:+.4f}   {all_d:+.4f}   {pct:>8.1f}%")

    # Slow field retention values
    print("\n=== Slow field retention (learned) ===")
    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
            hl = (-1 / torch.log(retain.mean())).item()
            print(f"  Layer {i}: retain={retain.mean().item():.4f} (hl≈{hl:.0f} tokens)")


if __name__ == "__main__":
    main()
