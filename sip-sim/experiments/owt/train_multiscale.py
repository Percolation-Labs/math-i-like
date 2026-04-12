"""Train multiscale GPT on OpenWebText — warm-started from single-field checkpoint.

Loads a trained single-field model, adds the slow field (multiscale),
and continues training. All existing weights (attention, MLP, embeddings,
fast field) are preserved. Only the new slow field parameters start fresh.

Usage:
    # Local test (CPU, verify warm-start logic)
    python train_multiscale.py --test

    # Full training — warm-start from step 70K checkpoint
    python train_multiscale.py \
        --resume-from checkpoints/owt_social_step70000.pt \
        --name owt_multiscale \
        --total-steps 70000

    # Resume interrupted multiscale training
    python train_multiscale.py \
        --resume checkpoints/owt_multiscale_latest.pt \
        --name owt_multiscale
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


def warm_start_multiscale(model, checkpoint_path, device):
    """Load single-field checkpoint into multiscale model.

    Loads all matching weights (attention, MLP, embeddings, fast field)
    and leaves the new slow field parameters at their init values.
    Returns the step number from the source checkpoint.
    """
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
                skipped.append(f"{key} (shape mismatch: {source_state[key].shape} vs {target_state[key].shape})")
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
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinystories"))
        from data import get_dataloaders as get_ts_dataloaders
        train_loader, val_loader = get_ts_dataloaders(
            batch_size=batch_size, context_length=context_length, num_workers=0)
    else:
        context_length = args.context_length
        batch_size = args.batch_size
        from data import get_dataloaders
        train_loader, val_loader = get_dataloaders(
            batch_size=batch_size, context_length=context_length,
            num_workers=args.num_workers, data_dir=args.data_dir)

    enc = get_tokenizer()
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model — multiscale variant
    args.use_field = True
    args.multi_scale = True
    args.context_length = context_length
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

    total_steps = args.total_steps
    if args.test:
        total_steps = min(total_steps, 30)

    warmup_steps = min(2000, total_steps // 10)
    schedule_fn = cosine_lr_schedule(warmup_steps, total_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    # If resuming multiscale training, reload optimizer and advance scheduler
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        for _ in range(start_step):
            scheduler.step()

    # Compile
    model = maybe_compile(model, device, enabled=not args.test)

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

    # Log slow field retention values
    print("\nSlow field initial retention values:")
    for i, block in enumerate(model.blocks if not hasattr(model, '_orig_mod') else model._orig_mod.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
            print(f"  Layer {i}: retain = {retain.mean().item():.4f} "
                  f"(half-life ≈ {(-1/torch.log(retain.mean())).item():.0f} tokens)")

    # Train loop
    print(f"\nTraining for {total_steps} steps...")
    print(f"Effective batch size: {batch_size} * {args.grad_accum} = "
          f"{batch_size * args.grad_accum}")
    model.train()
    step = start_step
    t0 = time.perf_counter()
    running_loss = 0.0
    grad_accum = args.grad_accum
    epoch = 0

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

                # Log slow field retention
                raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                for i, block in enumerate(raw.blocks):
                    if hasattr(block.attn, 'slow_retain_logit'):
                        retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
                        hl = (-1 / torch.log(retain.mean())).item()
                        print(f"  → L{i} slow retain={retain.mean().item():.4f} "
                              f"(hl≈{hl:.0f})")
                        break  # just show first layer as indicator

                entry = {"step": step, "val_loss": val_loss, "ppl": math.exp(val_loss)}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.name}_step{step}.pt")
                save_checkpoint(model, step, args, ckpt_path, optimizer)
                latest_path = os.path.join(args.checkpoint_dir, f"{args.name}_latest.pt")
                save_checkpoint(model, step, args, latest_path)
                print(f"  → saved {ckpt_path}")

        epoch += 1

    # Final save
    elapsed = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    final_path = os.path.join(args.checkpoint_dir, f"{args.name}_final.pt")
    save_checkpoint(model, step, args, final_path)
    print(f"Saved {final_path}")

    val_loss = estimate_loss(model, val_loader, device, max_batches=200, amp_dtype=amp_dtype)
    print(f"Final val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")

    # Final slow field retention
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

    # Model — GPT-2 Small defaults (must match source checkpoint)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)

    # Field config
    parser.add_argument("--evap-rate", type=float, default=0.05)
    parser.add_argument("--mod-type", type=str, default="additive")

    # Training
    parser.add_argument("--total-steps", type=int, default=70000,
                        help="Total training steps for this run")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")

    # Logging
    parser.add_argument("--name", type=str, default="owt_multiscale")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default=None)

    # Warm-start / resume
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Single-field checkpoint to warm-start from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Multiscale checkpoint to resume training from")
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
