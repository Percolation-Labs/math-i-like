"""Scale test: Stigmergic Attention on FineWeb-Edu (real web text).

Compares baseline GPT vs stigmergic GPT (EWMA key coupling) at ~50M params
on FineWeb-Edu, streaming from HuggingFace.

The EWMA is computed via the cumsum trick — fully parallel, no sequential
loop, native MPS/CUDA.

Model: d=512, H=8, L=8 (~50M params), ctx=512
Data:  FineWeb-Edu streamed, ~50M tokens pre-tokenized buffer

Modes:
  baseline    - standard transformer
  stigmergic  - EWMA key coupling, learnable β_h, α=0.9

Usage:
    cd sip-sim
    PYTHONUNBUFFERED=1 uv run python experiments/rg/run_scale_fineweb.py
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# Data: stream FineWeb-Edu, tokenize a buffer
# ══════════════════════════════════════════════════════════════

def load_fineweb_buffer(n_tokens=50_000_000, cache_dir=None):
    """Pre-tokenize a buffer of FineWeb-Edu tokens.

    Streams from HuggingFace, tokenizes with tiktoken gpt2,
    caches result to disk for reuse.
    """
    cache_path = Path(cache_dir or Path(__file__).resolve().parent / ".data")
    cache_path.mkdir(parents=True, exist_ok=True)
    tok_file = cache_path / f"fineweb_edu_{n_tokens // 1_000_000}M_tokens.pt"

    if tok_file.exists():
        print(f"Loading cached tokens from {tok_file}")
        tokens = torch.load(tok_file, weights_only=True)
        print(f"  {len(tokens)/1e6:.1f}M tokens loaded")
        return tokens

    print(f"Streaming FineWeb-Edu and tokenizing {n_tokens/1e6:.0f}M tokens...")
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    all_tokens = []
    for i, example in enumerate(ds):
        text = example["text"]
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        all_tokens.append(eot)
        if len(all_tokens) >= n_tokens:
            break
        if (i + 1) % 10_000 == 0:
            print(f"  {i+1} docs, {len(all_tokens)/1e6:.1f}M tokens")

    all_tokens = all_tokens[:n_tokens]
    tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(tensor, tok_file)
    print(f"  Cached {len(tensor)/1e6:.1f}M tokens to {tok_file}")
    return tensor


class TokenBuffer:
    """Simple dataset: random slices from a flat token buffer."""

    def __init__(self, tokens, context_length, split="train", val_fraction=0.005):
        n_val = int(len(tokens) * val_fraction)
        if split == "train":
            self.tokens = tokens[:-n_val]
        else:
            self.tokens = tokens[-n_val:]
        self.context_length = context_length

    def sample_batch(self, batch_size, device):
        max_start = len(self.tokens) - self.context_length - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.tokens[s:s + self.context_length] for s in starts])
        y = torch.stack([self.tokens[s + 1:s + self.context_length + 1] for s in starts])
        return x.to(device), y.to(device)


# ══════════════════════════════════════════════════════════════
# Components
# ══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, freqs):
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ══════════════════════════════════════════════════════════════
# Parallel EWMA via cumsum trick (no loop!)
# ══════════════════════════════════════════════════════════════

def ewma_parallel(x, alpha):
    """EWMA: s_t = α·s_{t-1} + (1-α)·x_t, computed in O(1) depth.

    Uses the identity:
      s_t = (1-α) · α^t · Σ_{i=0}^{t} x_i · α^{-i}

    The inner sum is a cumsum of x_i · α^{-i}, fully parallel.

    x: (B, H, T, d)
    alpha: scalar tensor

    Returns: (B, H, T, d)

    Safe for float32 when α ≥ 0.8 and T ≤ 512.
    """
    B, H, T, d = x.shape
    t_idx = torch.arange(T, device=x.device, dtype=x.dtype)

    # α^{-t} and α^t as (1, 1, T, 1) for broadcasting
    inv_alpha_t = (1.0 / alpha) ** t_idx
    alpha_t = alpha ** t_idx

    # y_i = x_i · α^{-i}, then cumsum, then scale by (1-α)·α^t
    y = x * inv_alpha_t.view(1, 1, T, 1)
    C = torch.cumsum(y, dim=2)
    return (1 - alpha) * C * alpha_t.view(1, 1, T, 1)


# ══════════════════════════════════════════════════════════════
# Attention modules
# ══════════════════════════════════════════════════════════════

class StigmergicAttention(nn.Module):
    """Multi-head attention with EWMA key trails and inter-head coupling.

    Fully parallel — uses ewma_parallel (cumsum trick), no position loop.
    """

    def __init__(self, d_model, n_head, dropout=0.1, max_len=512,
                 alpha=0.9, beta_init=0.1, learnable_beta=True,
                 tie_coup=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)

        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

        # Trail persistence
        self.register_buffer("alpha", torch.tensor(alpha))

        # Per-head coupling strength β_h ∈ [0, 4]
        self.learnable_beta = learnable_beta
        if learnable_beta:
            self.beta_logit = nn.Parameter(
                torch.full((n_head,), math.log(beta_init / (1 - beta_init + 1e-8))))
        else:
            self.register_buffer("beta_val", torch.tensor(beta_init))

        # Coupling projection
        if tie_coup is not None:
            self.w_coup = tie_coup
        else:
            self.w_coup = nn.Linear(self.head_dim, self.head_dim, bias=False)
            nn.init.normal_(self.w_coup.weight, std=0.02)

    def get_beta(self):
        if self.learnable_beta:
            return torch.sigmoid(self.beta_logit) * 4.0
        return self.beta_val

    def forward(self, x):
        B, T, C = x.shape
        H, d_h = self.n_head, self.head_dim
        scale = d_h ** -0.5

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # EWMA key trails — fully parallel via cumsum
        trails = ewma_parallel(k, self.alpha)  # (B, H, T, d_h)

        # Inter-head coupling
        trail_sum = trails.sum(dim=1, keepdim=True)
        other_trails = trail_sum - trails
        coupling = self.w_coup(other_trails)

        beta = self.get_beta()
        if isinstance(beta, torch.Tensor) and beta.dim() > 0:
            beta = beta.view(1, H, 1, 1)
        k_mod = k + beta * coupling

        # Standard causal attention with modulated keys
        att = (q @ k_mod.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class BaselineAttention(nn.Module):
    """Standard multi-head causal attention."""

    def __init__(self, d_model, n_head, dropout=0.1, max_len=512):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        H, d_h = self.n_head, self.head_dim
        scale = d_h ** -0.5

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        freqs = self.rope(T, x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ══════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_len,
                 use_stigmergic=False, alpha=0.9, beta_init=0.1,
                 tie_coup=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        if use_stigmergic:
            self.attn = StigmergicAttention(
                d_model, n_head, dropout, max_len,
                alpha=alpha, beta_init=beta_init,
                learnable_beta=True, tie_coup=tie_coup)
        else:
            self.attn = BaselineAttention(d_model, n_head, dropout, max_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_head=8, n_layer=8,
                 dropout=0.1, max_len=512,
                 use_stigmergic=False, alpha=0.9, beta_init=0.1,
                 tie_coup_across_layers=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        tie_coup = None
        if tie_coup_across_layers and use_stigmergic:
            head_dim = d_model // n_head
            tie_coup = nn.Linear(head_dim, head_dim, bias=False)
            nn.init.normal_(tie_coup.weight, std=0.02)
            self.shared_w_coup = tie_coup

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, dropout, max_len,
                  use_stigmergic=use_stigmergic, alpha=alpha,
                  beta_init=beta_init, tie_coup=tie_coup)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight,
                            std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp[-2].weight,
                            std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1), ignore_index=-1)
            return loss, logits
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def get_beta_info(self):
        info = {}
        for i, block in enumerate(self.blocks):
            attn = block.attn
            if hasattr(attn, 'get_beta'):
                beta = attn.get_beta()
                if isinstance(beta, torch.Tensor) and beta.dim() > 0:
                    info[f"L{i}_beta"] = beta.detach().cpu().tolist()
                else:
                    info[f"L{i}_beta"] = [float(beta)] * attn.n_head
        return info


# ══════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════

def train_model(mode, tokens, config):
    """Train one model and return metrics."""
    d_model = config["d_model"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    ctx = config["ctx"]
    batch_size = config["batch_size"]
    train_steps = config["train_steps"]
    lr = config["lr"]
    grad_accum = config["grad_accum"]

    use_stigmergic = (mode != "baseline")

    train_data = TokenBuffer(tokens, ctx, split="train")
    val_data = TokenBuffer(tokens, ctx, split="val")

    model = GPT(
        vocab_size=50257, d_model=d_model, n_head=n_head,
        n_layer=n_layer, dropout=0.1, max_len=ctx,
        use_stigmergic=use_stigmergic,
        alpha=0.9, beta_init=0.1,
    ).to(DEVICE)

    n_params = model.count_params()
    print(f"  Model: {n_params/1e6:.1f}M params, stigmergic={use_stigmergic}")

    # Separate weight decay
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    warmup_steps = min(200, train_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    log = []
    t0 = time.perf_counter()
    running_loss = 0.0
    micro_step = 0

    torch.manual_seed(42)

    model.train()
    for step in range(1, train_steps + 1):
        # Gradient accumulation
        total_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_data.sample_batch(batch_size, DEVICE)
            loss, _ = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += total_loss

        if step % 50 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            avg_loss = running_loss / min(step, 50)
            eff_batch = batch_size * grad_accum
            total_tokens = step * eff_batch * ctx
            tok_per_sec = total_tokens / elapsed

            entry = {
                "step": step, "loss": avg_loss,
                "ppl": math.exp(min(avg_loss, 20)),
                "tok_s": tok_per_sec, "elapsed": elapsed,
            }
            log.append(entry)

            ppl_str = f"{math.exp(min(avg_loss, 20)):.1f}"
            print(f"    step {step:5d}/{train_steps}  loss={avg_loss:.4f}  "
                  f"ppl={ppl_str:>8s}  "
                  f"{tok_per_sec:.0f} tok/s  ({elapsed:.0f}s)")
            running_loss = 0.0

        if step % 500 == 0:
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20):
                    vx, vy = val_data.sample_batch(batch_size, DEVICE)
                    vl, _ = model(vx, vy)
                    val_losses.append(vl.item())
            val_loss = np.mean(val_losses)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"    >>> val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")
            log[-1]["val_loss"] = val_loss
            log[-1]["val_ppl"] = val_ppl

            # Beta info
            beta_info = model.get_beta_info()
            if beta_info:
                betas = beta_info.get("L0_beta", [])
                print(f"    >>> β_h (L0): {['%.3f' % b for b in betas]}")
                log[-1]["beta_info"] = beta_info

            # Sample generation
            enc = tiktoken.get_encoding("gpt2")
            prompt = "The scientists discovered that"
            tok_ids = enc.encode(prompt)
            idx = torch.tensor([tok_ids], device=DEVICE)
            gen = model.generate(idx, max_new_tokens=60, temperature=0.8)
            text = enc.decode(gen[0].tolist())
            print(f"    >>> gen: {text[:200]}")

            model.train()

    # Final validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            vx, vy = val_data.sample_batch(batch_size, DEVICE)
            vl, _ = model(vx, vy)
            val_losses.append(vl.item())
    final_val = np.mean(val_losses)
    final_ppl = math.exp(min(final_val, 20))
    elapsed = time.perf_counter() - t0
    total_tokens = train_steps * batch_size * grad_accum * ctx
    avg_tok_s = total_tokens / elapsed

    result = {
        "mode": mode,
        "n_params": n_params,
        "final_val_loss": final_val,
        "final_val_ppl": final_ppl,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
        "avg_tok_s": avg_tok_s,
        "log": log,
        "beta_info": model.get_beta_info(),
    }
    print(f"\n  FINAL: val_loss={final_val:.4f}  ppl={final_ppl:.1f}  "
          f"{avg_tok_s:.0f} tok/s  {elapsed:.0f}s")
    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("SCALE TEST: Stigmergic Attention on FineWeb-Edu")
    print("=" * 70)

    config = {
        "d_model": 512,
        "n_head": 8,
        "n_layer": 8,
        "ctx": 256,
        "batch_size": 8,
        "grad_accum": 4,
        "train_steps": 1500,
        "lr": 3e-4,
    }

    eff_batch = config["batch_size"] * config["grad_accum"]
    total_tokens = config["train_steps"] * eff_batch * config["ctx"]
    print(f"Config: d={config['d_model']}, H={config['n_head']}, "
          f"L={config['n_layer']}, ctx={config['ctx']}")
    print(f"Training: {config['train_steps']} steps, "
          f"eff_batch={eff_batch}, {total_tokens/1e6:.0f}M tokens")
    print(f"Device: {DEVICE}")

    # Load data
    tokens = load_fineweb_buffer(n_tokens=50_000_000)

    modes = ["baseline", "stigmergic"]
    all_results = {}

    for mode in modes:
        print(f"\n{'═' * 70}")
        print(f"  MODE: {mode}")
        print(f"{'═' * 70}")

        result = train_model(mode, tokens, config)
        all_results[mode] = result

        # Save intermediate
        out_path = OUT_DIR / "exp_scale_fineweb.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    header = f"  {'Mode':>15s}  {'Params':>8s}  {'Val Loss':>10s}  " \
             f"{'Val PPL':>10s}  {'Tok/s':>10s}  {'Time':>8s}  {'Overhead':>8s}"
    print(header)
    print(f"  {'─' * 80}")

    baseline_tok_s = all_results.get("baseline", {}).get("avg_tok_s", 1)
    for mode in modes:
        r = all_results[mode]
        overhead = ((baseline_tok_s / r["avg_tok_s"]) - 1) * 100 if mode != "baseline" else 0
        print(f"  {mode:>15s}  {r['n_params']/1e6:>7.1f}M  "
              f"{r['final_val_loss']:>10.4f}  {r['final_val_ppl']:>10.1f}  "
              f"{r['avg_tok_s']:>10.0f}  {r['elapsed_s']:>7.0f}s  "
              f"{overhead:>+7.1f}%")

    ppl_delta = all_results["stigmergic"]["final_val_ppl"] - all_results["baseline"]["final_val_ppl"]
    print(f"\n  PPL difference: {ppl_delta:+.1f} (negative = stigmergic better)")
    print(f"  Throughput overhead: "
          f"{((baseline_tok_s / all_results['stigmergic']['avg_tok_s']) - 1) * 100:+.1f}%")

    out_path = OUT_DIR / "exp_scale_fineweb.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
