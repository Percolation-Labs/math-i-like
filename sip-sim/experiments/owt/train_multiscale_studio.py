"""Train multiscale GPT on OpenWebText — warm-started from single-field checkpoint.

Studio version: imports model from model_multiscale.py (same directory).

Usage on Lightning AI studio:
    cd ~/owt
    nohup python train_multiscale_studio.py \
        --resume-from checkpoints/owt_social_step70000.pt \
        --name owt_multiscale \
        --total-steps 70000 \
        > train_multiscale.log 2>&1 &
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import tiktoken

sys.path.insert(0, os.path.dirname(__file__))
from model_multiscale import GPT
from data import get_dataloaders


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def warm_start_multiscale(model, checkpoint_path, device):
    """Load single-field checkpoint into multiscale model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = ckpt["model"]
    target_state = model.state_dict()

    loaded, skipped = [], []
    for key in target_state:
        if key in source_state:
            if source_state[key].shape == target_state[key].shape:
                target_state[key] = source_state[key]
                loaded.append(key)
            else:
                skipped.append(f"{key} (shape mismatch: "
                               f"{source_state[key].shape} vs {target_state[key].shape})")
        else:
            skipped.append(f"{key} (new parameter)")

    model.load_state_dict(target_state)

    print(f"\nWarm-start from {checkpoint_path}:")
    print(f"  Loaded: {len(loaded)} parameters")
    print(f"  New (initialized fresh): {len(skipped)} parameters")
    for s in skipped:
        print(f"    → {s}")

    source_step = ckpt.get("step", 0)
    print(f"  Source checkpoint was at step {source_step}")
    return source_step


def save_checkpoint(model, step, args, path, optimizer=None):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": raw_model.state_dict(),
        "step": step,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, device, optimizer=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


@torch.no_grad()
def estimate_loss(model, val_loader, device, max_batches=50, amp_dtype=None):
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def generate_sample(model, device, enc, prompt="Once upon a time",
                    max_tokens=200, temperature=0.8, top_k=50):
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    model.train()
    return enc.decode(out[0].tolist())


def cosine_lr_schedule(warmup_steps, total_steps, min_lr_ratio=0.0):
    def schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    if args.no_amp:
        use_amp, amp_dtype = False, None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"AMP: {use_amp} (dtype={amp_dtype})")

    # Data
    print("Loading data...")
    context_length = args.context_length
    batch_size = args.batch_size
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, context_length=context_length,
        num_workers=args.num_workers, data_dir=args.data_dir)

    enc = get_tokenizer()
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model — multiscale
    use_grad_ckpt = device.type == "cuda" and not args.no_grad_ckpt
    args.use_field = True
    args.multi_scale = True
    args.context_length = context_length

    model = GPT(
        vocab_size=enc.n_vocab,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        max_len=context_length,
        chunk_size=args.chunk_size,
        evap_rate=args.evap_rate,
        use_field=True,
        mod_type=args.mod_type,
        gradient_checkpointing=use_grad_ckpt,
        multi_scale=True,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: {n_params/1e6:.1f}M params, multi_scale=True, "
          f"grad_ckpt={use_grad_ckpt}")

    # Warm-start or resume
    start_step = 0
    source_step = 0

    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, device)
        print(f"Resumed multiscale training from {args.resume} at step {start_step}")
    elif args.resume_from:
        source_step = warm_start_multiscale(model, args.resume_from, device)

    # Optimizer
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    print(f"Optimizer: {len(decay_params)} decay params, {len(nodecay_params)} no-decay params")

    use_fused = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        optim_groups, lr=args.lr, betas=(0.9, 0.95),
        **({"fused": True} if use_fused else {}))

    total_steps = args.total_steps

    warmup_steps = min(2000, total_steps // 10)
    schedule_fn = cosine_lr_schedule(warmup_steps, total_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        for _ in range(start_step):
            scheduler.step()

    # Compile
    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, f"{args.name}_log.jsonl")
    log_file = open(log_path, "a")

    # Initial validation
    print(f"\nPre-training validation...")
    val_loss = estimate_loss(model, val_loader, device, amp_dtype=amp_dtype)
    print(f"  val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")
    if source_step > 0:
        print(f"  (warm-started from single-field step {source_step})")

    # Log slow field retention
    print("\nSlow field initial retention values:")
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    for i, block in enumerate(raw.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
            hl = (-1 / torch.log(retain.mean())).item()
            print(f"  Layer {i}: retain={retain.mean().item():.4f} (hl≈{hl:.0f} tokens)")

    # Train loop
    print(f"\nTraining for {total_steps} steps...")
    print(f"Effective batch size: {batch_size} * {args.grad_accum} = "
          f"{batch_size * args.grad_accum}")
    model.train()
    step = start_step
    t0 = time.perf_counter()
    running_loss = 0.0
    grad_accum = args.grad_accum

    while step < total_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= total_steps:
                break

            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, _ = model(x, y)
                loss = loss / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum != 0:
                running_loss += loss.item()
                continue

            running_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - t0
                avg_loss = running_loss / args.log_every if step > 1 else running_loss
                lr = scheduler.get_last_lr()[0]
                effective_batch = batch_size * grad_accum
                toks_since_start = (step - start_step) * effective_batch * context_length
                tokens_per_sec = toks_since_start / elapsed if elapsed > 0 else 0

                entry = {
                    "step": step, "loss": avg_loss, "lr": lr,
                    "elapsed": elapsed, "tokens_per_sec": tokens_per_sec,
                    "source_step": source_step,
                }
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

                print(f"  step {step:6d}/{total_steps}  "
                      f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                      f"{tokens_per_sec:.0f} tok/s  "
                      f"({elapsed:.0f}s)")

                running_loss = 0.0

            if step % args.eval_every == 0:
                val_loss = estimate_loss(model, val_loader, device, amp_dtype=amp_dtype)
                sample = generate_sample(model, device, enc,
                                         prompt="The meaning of life is")
                print(f"  → val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")
                print(f"  → sample: {sample[:300]}...")

                raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                for i, block in enumerate(raw.blocks):
                    if hasattr(block.attn, 'slow_retain_logit'):
                        retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
                        hl = (-1 / torch.log(retain.mean())).item()
                        print(f"  → L{i} slow retain={retain.mean().item():.4f} "
                              f"(hl≈{hl:.0f})")
                        break

                entry = {"step": step, "val_loss": val_loss, "ppl": math.exp(val_loss)}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.name}_step{step}.pt")
                save_checkpoint(model, step, args, ckpt_path, optimizer)
                latest_path = os.path.join(args.checkpoint_dir, f"{args.name}_latest.pt")
                save_checkpoint(model, step, args, latest_path)
                print(f"  → saved {ckpt_path}")

    # Final save
    elapsed = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    final_path = os.path.join(args.checkpoint_dir, f"{args.name}_final.pt")
    save_checkpoint(model, step, args, final_path)
    print(f"Saved {final_path}")

    val_loss = estimate_loss(model, val_loader, device, max_batches=200, amp_dtype=amp_dtype)
    print(f"Final val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")

    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    print("\nFinal slow field retention:")
    for i, block in enumerate(raw.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
            hl = (-1 / torch.log(retain.mean())).item()
            print(f"  Layer {i}: retain={retain.mean().item():.4f} (hl≈{hl:.0f} tokens)")

    print("\n--- Sample generations ---")
    for prompt in ["The meaning of life is", "In a surprising turn of events",
                   "The scientists discovered that"]:
        text = generate_sample(model, device, enc, prompt=prompt)
        print(f"\n[{prompt}]\n{text[:500]}\n")

    log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train multiscale GPT on OpenWebText (warm-start)")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--evap-rate", type=float, default=0.05)
    parser.add_argument("--mod-type", type=str, default="additive")

    parser.add_argument("--total-steps", type=int, default=70000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM, faster)")

    parser.add_argument("--name", type=str, default="owt_multiscale")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default=None)

    parser.add_argument("--resume-from", type=str, default=None,
                        help="Single-field checkpoint to warm-start from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Multiscale checkpoint to resume training")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
