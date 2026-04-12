"""Train GPT on TinyStories — baseline vs social field.

Works locally (CPU/MPS) and on Lightning AI (A100).

Usage:
    # Local test (tiny run to verify everything works)
    python train.py --test

    # Full training — baseline
    python train.py --name baseline --use-field false

    # Full training — social field
    python train.py --name social --use-field true

    # Resume from checkpoint
    python train.py --name social --use-field true --resume checkpoints/social_latest.pt
"""

import argparse
import json
import math
import os
import time

import torch

from sip_sim.neural import (
    GPT, get_device, get_tokenizer, estimate_loss, generate_sample,
    save_checkpoint, load_checkpoint, cosine_lr_schedule, maybe_compile,
)
from data import get_dataloaders


def train(args):
    device = get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Data
    print("Loading data...")
    if args.test:
        context_length = 64
        batch_size = 4
        train_loader, val_loader = get_dataloaders(
            batch_size=batch_size, context_length=context_length,
            num_workers=0, data_dir=args.data_dir)
    else:
        context_length = args.context_length
        batch_size = args.batch_size
        train_loader, val_loader = get_dataloaders(
            batch_size=batch_size, context_length=context_length,
            num_workers=args.num_workers, data_dir=args.data_dir)

    enc = get_tokenizer()
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model
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
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: {n_params/1e6:.1f}M params, use_field={args.use_field}, "
          f"mod_type={args.mod_type}, evap_rate={args.evap_rate}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95))

    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    if args.test:
        total_steps = min(total_steps, 50)

    warmup_steps = min(500, total_steps // 10)
    schedule_fn = cosine_lr_schedule(warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, device, optimizer)
        print(f"Resumed from {args.resume} at step {start_step}")

    # Compile
    if device.type == "cuda" and not args.test:
        model = maybe_compile(model, device)

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, f"{args.name}_log.jsonl")
    log_file = open(log_path, "a")

    # Wandb (optional)
    use_wandb = args.wandb and not args.test
    if use_wandb:
        try:
            import wandb
            wandb.init(project="tinystories-social-field", name=args.name,
                       config=vars(args))
        except ImportError:
            use_wandb = False
            print("wandb not installed, skipping")

    # Train loop
    print(f"\nTraining for {total_steps} steps ({args.epochs} epochs)...")
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
                avg_loss = running_loss / args.log_every if step > 1 else loss.item()
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

                if use_wandb:
                    import wandb
                    wandb.log(entry, step=step)

                running_loss = 0.0

            # Validate + generate sample
            if step % args.eval_every == 0:
                val_loss = estimate_loss(model, val_loader, device)
                sample = generate_sample(model, device, enc)
                print(f"  → val_loss={val_loss:.4f}")
                print(f"  → sample: {sample[:300]}...")

                entry = {"step": step, "val_loss": val_loss}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

                if use_wandb:
                    import wandb
                    wandb.log({"val_loss": val_loss}, step=step)

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

    val_loss = estimate_loss(model, val_loader, device, max_batches=200)
    print(f"Final val_loss={val_loss:.4f}")

    print("\n--- Sample generations ---")
    for prompt in ["Once upon a time", "The little cat", "Tom wanted to"]:
        text = generate_sample(model, device, enc, prompt=prompt)
        print(f"\n[{prompt}]\n{text[:500]}\n")

    log_file.close()
    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories")

    # Model
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)

    # Social field
    parser.add_argument("--use-field", type=lambda s: s.lower() in ("true", "1", "yes"),
                        default=True)
    parser.add_argument("--evap-rate", type=float, default=0.05)
    parser.add_argument("--mod-type", type=str, default="additive",
                        choices=["additive", "gating"])

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--num-workers", type=int, default=4)

    # Logging
    parser.add_argument("--name", type=str, default="social")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")

    # Util
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Quick local sanity check (tiny run)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
