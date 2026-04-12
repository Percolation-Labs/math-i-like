"""Train 125M GPT on OpenWebText — social field vs published GPT-2 Small.

Designed for A100 with bf16 mixed precision and gradient checkpointing.
Works locally (CPU/MPS) for testing.

Usage:
    # Local test (verify everything works)
    python train.py --test

    # Full training — social field (A100, ~12h)
    python train.py --name owt_social --use-field true

    # Full training — baseline (only if you want your own baseline)
    python train.py --name owt_baseline --use-field false

125M config (GPT-2 Small scale):
    d_model=768, n_head=12, n_layer=12, context=1024
    ~124M params (baseline), ~128M params (social field)
"""

import argparse
import json
import math
import os
import time

import torch

from sip_sim.neural import (
    GPT, get_device, get_tokenizer, estimate_loss, generate_sample,
    save_checkpoint, load_checkpoint, cosine_lr_schedule,
    setup_training_precision, maybe_compile,
)
from data import get_dataloaders


def train(args):
    device = get_device()
    print(f"Device: {device}")

    use_amp, amp_dtype = setup_training_precision(device)
    if args.no_amp:
        use_amp, amp_dtype = False, None
    print(f"AMP: {use_amp} (dtype={amp_dtype})")

    # Data
    print("Loading data...")
    if args.test:
        context_length = 128
        batch_size = 4
        # Use TinyStories for local test (OpenWebText is huge)
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinystories"))
        from data import get_dataloaders as get_ts_dataloaders
        train_loader, val_loader = get_ts_dataloaders(
            batch_size=batch_size, context_length=context_length, num_workers=0)
    else:
        context_length = args.context_length
        batch_size = args.batch_size
        train_loader, val_loader = get_dataloaders(
            batch_size=batch_size, context_length=context_length,
            num_workers=args.num_workers, data_dir=args.data_dir)

    enc = get_tokenizer()
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model — 125M config
    use_grad_ckpt = device.type == "cuda" and not args.test
    model = GPT(
        vocab_size=enc.n_vocab,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        max_len=context_length,
        chunk_size=args.chunk_size,
        evap_rate=args.evap_rate,
        use_field=args.use_field,
        mod_type=args.mod_type,
        gradient_checkpointing=use_grad_ckpt,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: {n_params/1e6:.1f}M params, use_field={args.use_field}, "
          f"mod_type={args.mod_type}, grad_ckpt={use_grad_ckpt}")

    # Optimizer — separate weight decay for bias/norm
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

    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    if args.test:
        total_steps = min(total_steps, 30)

    warmup_steps = min(2000, total_steps // 10)
    schedule_fn = cosine_lr_schedule(warmup_steps, total_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, device, optimizer)
        print(f"Resumed from {args.resume} at step {start_step}")

    # Compile
    model = maybe_compile(model, device, enabled=not args.test)

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, f"{args.name}_log.jsonl")
    log_file = open(log_path, "a")

    # Train loop
    print(f"\nTraining for {total_steps} steps ({args.epochs} epochs)...")
    print(f"Effective batch size: {batch_size} * {args.grad_accum} = "
          f"{batch_size * args.grad_accum}")
    model.train()
    step = start_step
    t0 = time.perf_counter()
    running_loss = 0.0
    grad_accum = args.grad_accum

    for epoch in range(args.epochs):
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

            # Log
            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - t0
                avg_loss = running_loss / args.log_every if step > 1 else running_loss
                lr = scheduler.get_last_lr()[0]
                effective_batch = batch_size * grad_accum
                tokens_per_sec = (step * effective_batch * context_length) / elapsed

                entry = {
                    "step": step, "loss": avg_loss, "lr": lr,
                    "elapsed": elapsed, "tokens_per_sec": tokens_per_sec,
                }
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

                print(f"  step {step:6d}/{total_steps}  "
                      f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                      f"{tokens_per_sec:.0f} tok/s  "
                      f"({elapsed:.0f}s)")

                running_loss = 0.0

            # Validate
            if step % args.eval_every == 0:
                val_loss = estimate_loss(model, val_loader, device, amp_dtype=amp_dtype)
                sample = generate_sample(model, device, enc,
                                         prompt="The meaning of life is")
                print(f"  → val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")
                print(f"  → sample: {sample[:300]}...")

                entry = {"step": step, "val_loss": val_loss, "ppl": math.exp(val_loss)}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            # Checkpoint
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.name}_step{step}.pt")
                save_checkpoint(model, step, args, ckpt_path, optimizer)
                latest_path = os.path.join(args.checkpoint_dir, f"{args.name}_latest.pt")
                save_checkpoint(model, step, args, latest_path)
                print(f"  → saved {ckpt_path}")

        if step >= total_steps:
            break

    # Final save
    elapsed = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    final_path = os.path.join(args.checkpoint_dir, f"{args.name}_final.pt")
    save_checkpoint(model, step, args, final_path)
    print(f"Saved {final_path}")

    val_loss = estimate_loss(model, val_loader, device, max_batches=200, amp_dtype=amp_dtype)
    print(f"Final val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")

    print("\n--- Sample generations ---")
    for prompt in ["The meaning of life is", "In a surprising turn of events",
                   "The scientists discovered that"]:
        text = generate_sample(model, device, enc, prompt=prompt)
        print(f"\n[{prompt}]\n{text[:500]}\n")

    log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train 125M GPT on OpenWebText")

    # Model — GPT-2 Small defaults
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)

    # Social field
    parser.add_argument("--use-field", type=lambda s: s.lower() in ("true", "1", "yes"),
                        default=True)
    parser.add_argument("--evap-rate", type=float, default=0.05)
    parser.add_argument("--mod-type", type=str, default="additive",
                        choices=["additive", "gating"])

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")

    # Logging
    parser.add_argument("--name", type=str, default="owt_social")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default=None)

    # Util
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.test:
        args.d_model = 128
        args.n_head = 4
        args.n_layer = 4
        args.chunk_size = 32

    train(args)


if __name__ == "__main__":
    main()
