"""Length generalization experiment: train short, test long.

The theoretical argument: transformers are memoryless between positions.
The feedback field adds causal memory via a scale-invariant recurrence
(decay depends on distance, not absolute position). This should help
generalize to unseen lengths.

Design:
- Train on n_train examples (fixed length)
- Test on n_test > n_train examples (longer length)
- Compare: no_field vs feedback
- Tasks: parity (hardest for attention), flipflop, cumsum

The model uses LEARNED positional embeddings, so we need max_len to
cover the test length. We train on short sequences padded/masked within
that max_len.

Actually simpler: just create models with max_len = test_len, train
on short sequences (the model sees shorter inputs), test on long ones.
"""

import sys
import os

from learn.gate_experiments import (
    VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, BATCH_SIZE, BASE_LR,
    FIXED_DROPOUT, FIXED_WD, MEASURE_EVERY,
    GPT, gen_parity, gen_flipflop, gen_cumsum,
    OUT_DIR,
)

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_and_test(vocab_size, n_train, n_test_list, seed, evap_rate,
                   use_field, field_mode, generate_fn, train_steps=5000):
    """Train on n_train examples, evaluate on multiple test lengths.

    Returns dict: {n_test: val_acc} for each test length, plus train acc.
    """
    max_n = max(n_test_list)
    max_seq = 2 * (max_n + 1) - 1  # max input length

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = GPT(vocab_size, max_seq, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                evap_rate=evap_rate, use_field=use_field,
                pe_type="learned", use_rope=False, field_mode=field_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    # Train on short sequences
    for step in range(train_steps):
        model.train()
        input_ids, targets, _ = generate_fn(
            BATCH_SIZE, n_train, vocab_size, rng=rng)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Evaluate on each test length
    model.eval()
    results = {}
    eval_rng = np.random.default_rng(9999)  # fixed eval seed

    for n_test in n_test_list:
        with torch.no_grad():
            val_in, _, val_last = generate_fn(
                512, n_test, vocab_size, rng=np.random.default_rng(9999))
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1)
            val_acc = (val_pred == val_last).float().mean().item()
            results[n_test] = val_acc

    return results


def run():
    print("=" * 70)
    print("LENGTH GENERALIZATION: Train Short, Test Long")
    print("=" * 70)

    tasks = [
        ("parity", gen_parity),
        ("flipflop", gen_flipflop),
        ("cumsum", gen_cumsum),
    ]

    modes = [
        ("no_field", False, "decay", 0.0),
        ("feedback_05", True, "feedback", 0.05),
        ("feedback_10", True, "feedback", 0.10),
        ("feedback_20", True, "feedback", 0.20),
    ]

    n_train = 16  # train on sequences of length 34
    n_test_list = [16, 32, 48, 64]  # test on 34, 66, 98, 130
    seeds = [42, 137, 256]
    train_steps = 8000  # more steps for better training

    all_results = {}
    t0 = time.perf_counter()

    for task_name, gen_fn in tasks:
        print(f"\n{'═'*70}")
        print(f"TASK: {task_name} — train n={n_train} (T={2*(n_train+1)})")
        print(f"{'═'*70}")

        task_results = {}

        for mode_name, use_field, field_mode, eps in modes:
            print(f"\n  Mode: {mode_name}")

            # Collect results across seeds for each test length
            seed_results = {n: [] for n in n_test_list}

            for seed in seeds:
                results = train_and_test(
                    VOCAB_SIZE, n_train, n_test_list, seed, eps,
                    use_field, field_mode, gen_fn, train_steps)
                for n in n_test_list:
                    seed_results[n].append(results[n])

                elapsed = time.perf_counter() - t0
                train_acc = results[n_train]
                print(f"    seed={seed}  train={train_acc:.3f}  "
                      f"test=[{', '.join(f'{results[n]:.3f}' for n in n_test_list)}]  "
                      f"({elapsed:.0f}s)")

            # Aggregate
            mode_results = {}
            for n in n_test_list:
                vals = seed_results[n]
                mode_results[n] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "vals": vals,
                }

            task_results[mode_name] = mode_results

            # Print summary
            print(f"    Mean: [{', '.join(f'{mode_results[n]['mean']:.3f}' for n in n_test_list)}]")

        all_results[task_name] = task_results

        # Print comparison table
        print(f"\n  {'Length':>8s}", end="")
        for mode_name, _, _, _ in modes:
            print(f"  {mode_name:>14s}", end="")
        print()
        for n in n_test_list:
            t_len = 2 * (n + 1)
            ood = " (OOD)" if n > n_train else ""
            print(f"  T={t_len:3d}{ood:6s}", end="")
            for mode_name, _, _, _ in modes:
                r = task_results[mode_name][n]
                print(f"  {r['mean']:.3f}±{r['std']:.3f}", end="")
            print()

    total_time = time.perf_counter() - t0
    print(f"\n{'═'*70}")
    print(f"Complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'═'*70}")

    # Plot
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]

    mode_colors = {
        "no_field": "tab:gray",
        "feedback_05": "tab:green",
        "feedback_10": "tab:blue",
        "feedback_20": "tab:orange",
    }
    mode_labels = {
        "no_field": "No field (baseline)",
        "feedback_05": "Feedback ε=0.05",
        "feedback_10": "Feedback ε=0.10",
        "feedback_20": "Feedback ε=0.20",
    }

    for ax, (task_name, _) in zip(axes, tasks):
        task_data = all_results[task_name]

        for mode_name, _, _, _ in modes:
            lengths = [2 * (n + 1) for n in n_test_list]
            vals = [task_data[mode_name][n]["mean"] for n in n_test_list]
            stds = [task_data[mode_name][n]["std"] for n in n_test_list]

            ax.errorbar(lengths, vals, yerr=stds, fmt="o-",
                        color=mode_colors[mode_name],
                        linewidth=2, markersize=8, capsize=4,
                        label=mode_labels[mode_name])

        # Mark training length
        train_t = 2 * (n_train + 1)
        ax.axvline(x=train_t, color="black", linestyle="--", alpha=0.3,
                   label=f"Train length (T={train_t})")

        # Random baselines
        ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
                   label=f"Random (1/{VOCAB_SIZE})")
        if task_name == "parity":
            ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.3,
                       label="Random (binary)")

        ax.set_xlabel("Test sequence length T")
        ax.set_ylabel("Accuracy (3 seeds)")
        ax.set_title(f"{task_name.title()}\n(trained on T={train_t})")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Length Generalization: Train Short, Test Long\n"
                 f"(V={VOCAB_SIZE}, d={D_MODEL}, L={N_LAYER}, H={N_HEAD})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lengthgen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lengthgen.png")


if __name__ == "__main__":
    run()
