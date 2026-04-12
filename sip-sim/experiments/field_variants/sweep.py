"""Sweep field architecture variants on TinyStories.

Tests which architectural modification best draws out the field's contribution,
measured by per-chunk ablation delta (the signature we found on OWT at step 30K).

Variants:
  1. baseline     — no field (standard transformer)
  2. fixed        — current fixed ε=0.05 (what's running on OWT)
  3. learnable    — learnable per-head retention
  4. multiscale   — fast field (fixed ε=0.05) + slow field (learnable, init ε≈0.005)
  5. crosslayer   — cross-layer field coupling (field flows vertically)

Each variant trains for the same number of steps on TinyStories, then we measure:
  - Overall val loss
  - Per-chunk ablation delta (field ON vs field OFF, broken by position)
  - Learned retention values (for learnable/multiscale variants)

Usage:
    python -m experiments.field_variants.sweep
    python -m experiments.field_variants.sweep --variant learnable
    python -m experiments.field_variants.sweep --variant all --steps 3000

Run from repo root.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from sip_sim.neural import (
    GPT, get_device, get_tokenizer, estimate_loss, generate_sample,
    cosine_lr_schedule,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinystories"))
from data import get_dataloaders


VARIANTS = {
    "baseline": dict(use_field=False),
    "fixed": dict(use_field=True, evap_rate=0.05),
    "learnable": dict(use_field=True, evap_rate=0.05, learnable_retain=True),
    "multiscale": dict(use_field=True, evap_rate=0.05, multi_scale=True),
    "crosslayer": dict(use_field=True, evap_rate=0.05, cross_layer=True),
}


def build_model(variant_name, d_model, n_head, n_layer, context_length,
                chunk_size, vocab_size, device):
    """Build a GPT model for the given variant."""
    cfg = VARIANTS[variant_name]
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        dropout=0.1,
        max_len=context_length,
        chunk_size=chunk_size,
        **cfg,
    ).to(device)
    return model


def train_variant(variant_name, args, train_loader, val_loader, enc, device):
    """Train a single variant and return results."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {variant_name}")
    print(f"{'='*70}")

    model = build_model(
        variant_name, args.d_model, args.n_head, args.n_layer,
        args.context_length, args.chunk_size, enc.n_vocab, device)

    n_params = model.count_parameters()
    print(f"  {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = args.steps
    warmup_steps = min(200, total_steps // 10)
    schedule_fn = cosine_lr_schedule(warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    model.train()
    step = 0
    t0 = time.perf_counter()
    running_loss = 0.0
    train_iter = iter(train_loader)
    log = []

    while step < total_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        loss, _ = model(x, y)
        loss.backward()

        running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            avg_loss = running_loss / min(step, args.log_every)
            lr = scheduler.get_last_lr()[0]
            tps = (step * args.batch_size * args.context_length) / elapsed
            print(f"  step {step:5d}/{total_steps}  loss={avg_loss:.4f}  "
                  f"lr={lr:.2e}  {tps:.0f} tok/s")
            log.append({"step": step, "loss": avg_loss, "lr": lr})
            running_loss = 0.0

        if step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_loader, device, max_batches=50)
            print(f"  → val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")
            log.append({"step": step, "val_loss": val_loss})
            model.train()

    elapsed = time.perf_counter() - t0
    val_loss = estimate_loss(model, val_loader, device, max_batches=100)
    print(f"  Final: val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}  "
          f"({elapsed:.0f}s)")

    return model, val_loss, log


def per_chunk_ablation(model, val_loader, device, chunk_size, context_length,
                       n_batches=100):
    """Measure field ON vs OFF val loss per chunk position.

    Returns dict mapping chunk_idx -> (ppl_on, ppl_off, delta_pct).
    """
    has_field = any(
        b.attn.use_field for b in
        (model._orig_mod.blocks if hasattr(model, '_orig_mod')
         else model.blocks))
    if not has_field:
        return None

    model_ablated = copy.deepcopy(model)
    for block in (model_ablated._orig_mod.blocks
                  if hasattr(model_ablated, '_orig_mod')
                  else model_ablated.blocks):
        attn = block.attn
        if attn.use_field:
            if attn.mod_type == "additive":
                attn.w_mod.weight.data.zero_()
                if attn.multi_scale:
                    attn.w_mod_slow.weight.data.zero_()
            elif attn.mod_type == "gating":
                attn.w_gate.weight.data.zero_()
                attn.w_gate.bias.data.fill_(100.0)

    model.eval()
    model_ablated.eval()

    n_chunks = context_length // chunk_size
    chunk_loss_on = [0.0] * n_chunks
    chunk_loss_off = [0.0] * n_chunks
    count = [0] * n_chunks

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)

            logits_on = model(x)
            logits_off = model_ablated(x)

            if isinstance(logits_on, tuple):
                logits_on = logits_on[0]
            if isinstance(logits_off, tuple):
                logits_off = logits_off[0]

            for ci in range(n_chunks):
                s = ci * chunk_size
                e = min((ci + 1) * chunk_size, context_length)
                loss_on = F.cross_entropy(
                    logits_on[:, s:e].reshape(-1, logits_on.size(-1)),
                    y[:, s:e].reshape(-1), reduction="mean").item()
                loss_off = F.cross_entropy(
                    logits_off[:, s:e].reshape(-1, logits_off.size(-1)),
                    y[:, s:e].reshape(-1), reduction="mean").item()
                chunk_loss_on[ci] += loss_on
                chunk_loss_off[ci] += loss_off
                count[ci] += 1

    del model_ablated
    model.train()

    results = {}
    for ci in range(n_chunks):
        avg_on = chunk_loss_on[ci] / count[ci]
        avg_off = chunk_loss_off[ci] / count[ci]
        ppl_on = math.exp(avg_on)
        ppl_off = math.exp(avg_off)
        delta_pct = 100 * (ppl_off - ppl_on) / ppl_on if ppl_on > 0 else 0
        results[ci] = (ppl_on, ppl_off, delta_pct)

    return results


def report_learned_retention(model, variant_name):
    """Report learned retention values for learnable/multiscale variants."""
    blocks = (model._orig_mod.blocks if hasattr(model, '_orig_mod')
              else model.blocks)

    for li, block in enumerate(blocks):
        attn = block.attn
        if not attn.use_field:
            continue

        if hasattr(attn, 'retain_logit'):
            retain = torch.sigmoid(attn.retain_logit).detach()
            evap = 1 - retain
            print(f"  Layer {li} learned retain: "
                  f"{' '.join(f'{r:.4f}' for r in retain)}")
            print(f"  Layer {li} learned evap:   "
                  f"{' '.join(f'{e:.4f}' for e in evap)}")
            half_life = -1.0 / torch.log2(retain)
            print(f"  Layer {li} half-life (tokens): "
                  f"{' '.join(f'{h:.0f}' for h in half_life)}")

        if hasattr(attn, 'slow_retain_logit'):
            retain_slow = torch.sigmoid(attn.slow_retain_logit).detach()
            evap_slow = 1 - retain_slow
            print(f"  Layer {li} slow retain:    "
                  f"{' '.join(f'{r:.5f}' for r in retain_slow)}")
            half_life_slow = -1.0 / torch.log2(retain_slow)
            print(f"  Layer {li} slow half-life: "
                  f"{' '.join(f'{h:.0f}' for h in half_life_slow)}")


def main():
    parser = argparse.ArgumentParser(description="Sweep field variants on TinyStories")
    parser.add_argument("--variant", type=str, default="all",
                        choices=list(VARIANTS.keys()) + ["all"])
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments/field_variants/results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: d={args.d_model}, h={args.n_head}, L={args.n_layer}, "
          f"ctx={args.context_length}, chunk={args.chunk_size}")

    enc = get_tokenizer()
    print("Loading TinyStories...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, context_length=args.context_length,
        num_workers=0)

    variants_to_run = (list(VARIANTS.keys()) if args.variant == "all"
                       else [args.variant])

    all_results = {}

    for variant_name in variants_to_run:
        torch.manual_seed(args.seed)

        model, val_loss, log = train_variant(
            variant_name, args, train_loader, val_loader, enc, device)

        # Per-chunk ablation
        print(f"\n  Per-chunk ablation for {variant_name}:")
        ablation = per_chunk_ablation(
            model, val_loader, device, args.chunk_size, args.context_length)

        if ablation is not None:
            print(f"  {'Chunk':>6} | {'Pos':>10} | {'PPL ON':>8} | "
                  f"{'PPL OFF':>8} | {'Δ%':>7}")
            print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
            for ci, (ppl_on, ppl_off, delta_pct) in sorted(ablation.items()):
                s = ci * args.chunk_size
                e = (ci + 1) * args.chunk_size
                print(f"  {ci:6d} | {s:>4}-{e-1:<4} | {ppl_on:8.2f} | "
                      f"{ppl_off:8.2f} | {delta_pct:+6.2f}%")
        else:
            print(f"  (no field — skipping ablation)")

        # Learned retention values
        if variant_name in ("learnable", "multiscale"):
            print(f"\n  Learned retention values:")
            report_learned_retention(model, variant_name)

        # Field weight magnitudes
        if variant_name != "baseline":
            print(f"\n  Field weight magnitudes:")
            blocks = (model._orig_mod.blocks if hasattr(model, '_orig_mod')
                      else model.blocks)
            for li, block in enumerate(blocks):
                attn = block.attn
                if not attn.use_field:
                    continue
                w_mod_std = attn.w_mod.weight.data.std().item()
                w_qkv_std = attn.w_qkv.weight.data.std().item()
                extra = ""
                if attn.multi_scale:
                    w_slow_std = attn.w_mod_slow.weight.data.std().item()
                    extra = f"  w_mod_slow={w_slow_std:.5f}"
                print(f"  L{li}: w_mod={w_mod_std:.5f}  "
                      f"w_qkv={w_qkv_std:.5f}  "
                      f"ratio={w_mod_std/w_qkv_std:.3f}{extra}")

        # Sample generation
        sample = generate_sample(model, device, enc,
                                 prompt="Once upon a time", max_tokens=150)
        print(f"\n  Sample: {sample[:300]}...")

        all_results[variant_name] = {
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss),
            "n_params": model.count_parameters(),
            "ablation": {str(k): v for k, v in ablation.items()} if ablation else None,
            "log": log,
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':>12} | {'Params':>8} | {'Val PPL':>8} | "
          f"{'Chunk 0 Δ':>9} | {'Last Chunk Δ':>12}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*12}")

    for name, res in all_results.items():
        params = f"{res['n_params']/1e6:.2f}M"
        ppl = f"{res['val_ppl']:.1f}"
        if res['ablation']:
            chunk_keys = sorted(res['ablation'].keys(), key=int)
            c0_delta = f"{res['ablation'][chunk_keys[0]][2]:+.2f}%"
            last_delta = f"{res['ablation'][chunk_keys[-1]][2]:+.2f}%"
        else:
            c0_delta = "n/a"
            last_delta = "n/a"
        print(f"{name:>12} | {params:>8} | {ppl:>8} | {c0_delta:>9} | {last_delta:>12}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
