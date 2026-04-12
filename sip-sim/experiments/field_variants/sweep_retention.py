"""Sweep slow-field retention training strategies on TinyStories.

Tests whether the hierarchy collapse (all layers except L0 → hl≈1) seen on OWT
is a genuine architectural preference or an optimizer artifact.

Variants:
  1. ms_uniform     — current multiscale: uniform LR, uniform init (control)
  2. ms_low_lr      — 10x lower LR for slow_retain_logit parameters
  3. ms_fixed_hier  — fixed retain values per layer (frozen, geometric spacing)
  4. ms_diverse_init— diverse initial half-lives per layer, still learnable
  5. ms_warmup      — freeze slow field for 30% of steps, then unfreeze at 10x lower LR

All variants use multi_scale=True. We measure per-chunk ablation and
learned retention values to see which strategy produces a genuine hierarchy.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.field_variants.sweep_retention
    PYTHONUNBUFFERED=1 python -m experiments.field_variants.sweep_retention --variant ms_fixed_hier
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


VARIANT_NAMES = [
    "ms_uniform",
    "ms_low_lr",
    "ms_fixed_hier",
    "ms_diverse_init",
    "ms_warmup",
]


def hl_to_logit(half_life):
    """Convert half-life (tokens) to retain logit for sigmoid parameterization."""
    retain = math.exp(-1.0 / half_life)
    return math.log(retain / (1.0 - retain))


def build_model(n_layer, d_model, n_head, context_length, chunk_size,
                vocab_size, device):
    """Build a multiscale GPT model."""
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        dropout=0.1,
        max_len=context_length,
        chunk_size=chunk_size,
        evap_rate=0.05,
        use_field=True,
        multi_scale=True,
    ).to(device)
    return model


def set_diverse_init(model, n_layer):
    """Initialize each layer's slow_retain_logit to a different half-life.

    Geometric spacing: Layer 0 gets the longest (≈500 tokens),
    last layer gets the shortest (≈20 tokens).
    """
    hl_max, hl_min = 500.0, 20.0
    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            t = i / max(n_layer - 1, 1)
            hl = hl_max * (hl_min / hl_max) ** t
            logit_val = hl_to_logit(hl)
            block.attn.slow_retain_logit.data.fill_(logit_val)


def set_fixed_hierarchy(model, n_layer):
    """Set fixed retain values (geometric spacing) and freeze them."""
    hl_max, hl_min = 500.0, 20.0
    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            t = i / max(n_layer - 1, 1)
            hl = hl_max * (hl_min / hl_max) ** t
            logit_val = hl_to_logit(hl)
            block.attn.slow_retain_logit.data.fill_(logit_val)
            block.attn.slow_retain_logit.requires_grad_(False)


def build_optimizer(model, variant, base_lr):
    """Build optimizer with variant-specific parameter groups."""
    retain_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'slow_retain_logit' in name:
            retain_params.append(param)
        else:
            other_params.append(param)

    if variant == "ms_low_lr":
        groups = [
            {"params": other_params, "lr": base_lr, "weight_decay": 0.1},
            {"params": retain_params, "lr": base_lr / 10, "weight_decay": 0.0},
        ]
    elif variant == "ms_warmup":
        for p in retain_params:
            p.requires_grad_(False)
        groups = [
            {"params": other_params, "lr": base_lr, "weight_decay": 0.1},
        ]
    else:
        decay = [p for n, p in model.named_parameters()
                 if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in model.named_parameters()
                   if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "lr": base_lr, "weight_decay": 0.1},
            {"params": nodecay, "lr": base_lr, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    return optimizer


def unfreeze_slow_field(model, optimizer, base_lr):
    """Unfreeze slow_retain_logit params and add to optimizer at low LR."""
    new_params = []
    for block in model.blocks:
        if hasattr(block.attn, 'slow_retain_logit'):
            block.attn.slow_retain_logit.requires_grad_(True)
            new_params.append(block.attn.slow_retain_logit)

    optimizer.add_param_group({
        "params": new_params,
        "lr": base_lr / 10,
        "weight_decay": 0.0,
    })


def report_retention(model, label=""):
    """Print slow field retention for all layers."""
    if label:
        print(f"  [{label}]")
    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, 'slow_retain_logit'):
            retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
            hl = (-1.0 / torch.log(retain.clamp(min=1e-7).mean())).item()
            frozen = "" if block.attn.slow_retain_logit.requires_grad else " (frozen)"
            print(f"    L{i}: retain={retain.mean().item():.4f} hl≈{hl:.0f}{frozen}")


def per_chunk_ablation_slow(model, val_loader, device, chunk_size,
                            context_length, n_batches=80):
    """Measure slow-field-only ablation per chunk."""
    model_ablated = copy.deepcopy(model)
    for block in model_ablated.blocks:
        attn = block.attn
        if hasattr(attn, 'w_mod_slow'):
            attn.w_mod_slow.weight.data.zero_()

    model.eval()
    model_ablated.eval()

    n_chunks = context_length // chunk_size
    loss_on = [0.0] * n_chunks
    loss_off = [0.0] * n_chunks
    count = 0

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
                e = (ci + 1) * chunk_size
                lo = F.cross_entropy(
                    logits_on[:, s:e].reshape(-1, logits_on.size(-1)),
                    y[:, s:e].reshape(-1)).item()
                lf = F.cross_entropy(
                    logits_off[:, s:e].reshape(-1, logits_off.size(-1)),
                    y[:, s:e].reshape(-1)).item()
                loss_on[ci] += lo
                loss_off[ci] += lf
            count += 1

    del model_ablated
    model.train()

    results = {}
    for ci in range(n_chunks):
        on = loss_on[ci] / count
        off = loss_off[ci] / count
        results[ci] = {"loss_on": on, "loss_off": off, "delta": off - on}
    return results


def train_variant(variant, args, train_loader, val_loader, enc, device):
    """Train one retention variant and return model + results."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant}")
    print(f"{'='*70}")

    torch.manual_seed(args.seed)
    model = build_model(args.n_layer, args.d_model, args.n_head,
                        args.context_length, args.chunk_size,
                        enc.n_vocab, device)

    if variant == "ms_diverse_init":
        set_diverse_init(model, args.n_layer)
    elif variant == "ms_fixed_hier":
        set_fixed_hierarchy(model, args.n_layer)

    print(f"  {model.count_parameters()/1e6:.2f}M params")
    report_retention(model, "init")

    optimizer = build_optimizer(model, variant, args.lr)

    total_steps = args.steps
    warmup_steps = min(200, total_steps // 10)
    schedule_fn = cosine_lr_schedule(warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    unfreeze_step = int(total_steps * 0.3) if variant == "ms_warmup" else None
    unfrozen = False

    model.train()
    step = 0
    t0 = time.perf_counter()
    running_loss = 0.0
    train_iter = iter(train_loader)

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

        if variant == "ms_warmup" and step == unfreeze_step and not unfrozen:
            unfreeze_slow_field(model, optimizer, args.lr)
            unfrozen = True
            print(f"  *** Unfroze slow field at step {step} ***")
            report_retention(model, "at unfreeze")

        if step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            avg = running_loss / min(step, args.log_every)
            lr = scheduler.get_last_lr()[0]
            tps = (step * args.batch_size * args.context_length) / elapsed
            print(f"  step {step:5d}/{total_steps}  loss={avg:.4f}  "
                  f"lr={lr:.2e}  {tps:.0f} tok/s")
            running_loss = 0.0

        if step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_loader, device, max_batches=50)
            print(f"  → val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}")
            report_retention(model, f"step {step}")
            model.train()

    elapsed = time.perf_counter() - t0
    val_loss = estimate_loss(model, val_loader, device, max_batches=100)
    print(f"\n  Final: val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.1f}  "
          f"({elapsed:.0f}s)")
    report_retention(model, "final")

    # Per-chunk slow field ablation
    print(f"\n  Slow field ablation:")
    ablation = per_chunk_ablation_slow(
        model, val_loader, device, args.chunk_size, args.context_length)
    print(f"  {'Chunk':>6} {'Tokens':>10} {'Slow Δ':>10}")
    for ci in sorted(ablation.keys()):
        s = ci * args.chunk_size
        e = (ci + 1) * args.chunk_size
        d = ablation[ci]["delta"]
        print(f"  {ci:>6} {s:>4}-{e-1:<4}   {d:+.4f}")

    return model, val_loss, ablation


def main():
    parser = argparse.ArgumentParser(
        description="Sweep slow-field retention strategies")
    parser.add_argument("--variant", type=str, default="all",
                        choices=VARIANT_NAMES + ["all"])
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/field_variants/results")
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

    variants = VARIANT_NAMES if args.variant == "all" else [args.variant]

    all_results = {}

    for variant in variants:
        model, val_loss, ablation = train_variant(
            variant, args, train_loader, val_loader, enc, device)

        # Collect final retention values
        retentions = {}
        for i, block in enumerate(model.blocks):
            if hasattr(block.attn, 'slow_retain_logit'):
                retain = torch.sigmoid(block.attn.slow_retain_logit).detach()
                hl = (-1.0 / torch.log(retain.clamp(min=1e-7).mean())).item()
                retentions[i] = {"retain": retain.mean().item(), "hl": hl}

        all_results[variant] = {
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss),
            "retentions": retentions,
            "ablation": {str(k): v for k, v in ablation.items()},
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # Summary
    n_chunks = args.context_length // args.chunk_size
    last_chunk = str(n_chunks - 1)

    print(f"\n\n{'='*70}")
    print("SUMMARY — Retention Strategy Sweep")
    print(f"{'='*70}")
    print(f"{'Variant':>16} | {'PPL':>6} | {'Last Chunk Δ':>12} | {'Hierarchy':>40}")
    print(f"{'-'*16}-+-{'-'*6}-+-{'-'*12}-+-{'-'*40}")

    for name, res in all_results.items():
        ppl = f"{res['val_ppl']:.1f}"
        delta = f"{res['ablation'][last_chunk]['delta']:+.4f}"
        hls = [f"L{i}:{r['hl']:.0f}" for i, r in sorted(res['retentions'].items())]
        hier = " ".join(hls)
        print(f"{name:>16} | {ppl:>6} | {delta:>12} | {hier}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "retention_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
