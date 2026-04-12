"""Dual-channel attention experiments: social field with vector key modulation.

Tests three tasks in order:
1. Parity (1-bit state) — validate mechanism works, compare to scalar
2. Cumsum mod V (4-bit state) — test state capacity
3. Bracket matching — test geometric routing (new task)

All with RoPE (no learned PE) to enable length generalization.
Compare: standard attention, scalar feedback, dual-channel additive.
"""

import sys
import os

from learn.gate_experiments import (
    VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, BATCH_SIZE, BASE_LR,
    FIXED_DROPOUT, FIXED_WD, MEASURE_EVERY,
    GPT, gen_parity, gen_cumsum,
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


def generate_batch_brackets(batch_size, n_examples, vocab_size, rng):
    """Bracket matching task: predict nesting depth at each position.

    Uses 3 bracket types: () [] {}
    Token encoding:
      0-2: open brackets  ( [ {
      3-5: close brackets ) ] }
      6:   padding / answer token
    Answer is the nesting depth mod vocab_size (0 = balanced).

    Format: b0 d0 b1 d1 ... bn dn
    where bi is a bracket token and di is the depth after that bracket.

    The model must track: (1) nesting depth (counter), (2) which bracket
    type to match (implicit stack). This requires the social field to
    encode STRUCTURED state — depth as magnitude, type as direction.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    n_bracket_types = 3  # (), [], {}
    max_depth = vocab_size - 1  # cap depth to fit in vocab

    for b in range(batch_size):
        depth = 0
        stack = []  # stack of bracket types currently open

        for i in range(n_total):
            # Decide: open or close?
            if depth == 0 or (len(stack) > 0 and rng.random() < 0.6):
                # Open a bracket (bias toward opening to build depth)
                btype = int(rng.integers(0, n_bracket_types))
                tokens[b, 2 * i] = btype  # open bracket
                stack.append(btype)
                depth = min(depth + 1, max_depth)
            else:
                # Close the most recent bracket (correct matching)
                if len(stack) > 0:
                    btype = stack.pop()
                    tokens[b, 2 * i] = btype + n_bracket_types  # close bracket
                    depth = max(depth - 1, 0)
                else:
                    # Mismatched close (creates error, depth stays 0)
                    btype = int(rng.integers(0, n_bracket_types))
                    tokens[b, 2 * i] = btype + n_bracket_types
                    depth = 0

            tokens[b, 2 * i + 1] = depth % vocab_size

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    # Only predict depth positions (odd positions in token stream)
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]
    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


def gen_brackets(batch_size, n_examples, vocab_size, rng):
    return generate_batch_brackets(batch_size, n_examples, vocab_size, rng)


def train_and_eval(mode_name, vocab_size, n_examples, seed, evap_rate,
                   generate_fn, train_steps=5000, n_test_list=None):
    """Train a model and evaluate at training length + optional OOD lengths.

    mode_name: "baseline", "scalar_feedback", "dual_additive", "dual_gating"
    Returns dict of {n: accuracy} for each test length.
    """
    if n_test_list is None:
        n_test_list = [n_examples]
    max_n = max(n_test_list)
    max_seq = 2 * (max_n + 1) - 1

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Build model based on mode
    if mode_name == "baseline":
        model = GPT(vocab_size, max_seq, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                     evap_rate=0.0, use_field=False,
                     pe_type="none", use_rope=True)
    elif mode_name == "scalar_feedback":
        model = GPT(vocab_size, max_seq, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                     evap_rate=evap_rate, use_field=True,
                     pe_type="none", use_rope=True, field_mode="feedback")
    elif mode_name == "dual_additive":
        model = GPT(vocab_size, max_seq, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                     evap_rate=evap_rate, use_field=False,
                     pe_type="none", use_rope=True,
                     attn_type="dual_channel", mod_type="additive")
    elif mode_name == "dual_gating":
        model = GPT(vocab_size, max_seq, D_MODEL, N_HEAD, N_LAYER, FIXED_DROPOUT,
                     evap_rate=evap_rate, use_field=False,
                     pe_type="none", use_rope=True,
                     attn_type="dual_channel", mod_type="gating")
    else:
        raise ValueError(f"Unknown mode: {mode_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                   weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=train_steps)

    # Train
    for step in range(train_steps):
        model.train()
        input_ids, targets, _ = generate_fn(
            BATCH_SIZE, n_examples, vocab_size, rng=rng)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Evaluate at each test length
    model.eval()
    results = {}
    for n_test in n_test_list:
        with torch.no_grad():
            eval_rng = np.random.default_rng(9999)
            val_in, _, val_last = generate_fn(
                512, n_test, vocab_size, rng=eval_rng)
            val_logits = model(val_in)
            val_pred = val_logits[:, -1].argmax(dim=-1)
            val_acc = (val_pred == val_last).float().mean().item()
            results[n_test] = val_acc

    return results


def run_task(task_name, gen_fn, n_train=16, train_steps=5000,
             test_lengths=True, eps_values=None):
    """Run one task across all modes, return results."""
    if eps_values is None:
        eps_values = [0.05, 0.10, 0.20]

    n_test_list = [n_train]
    if test_lengths:
        n_test_list = [n_train, n_train * 2, n_train * 3]

    seeds = [42, 137, 256]

    modes = [
        ("baseline", None),
        ("scalar_feedback", "best"),  # sweep eps, pick best
        ("dual_additive", "best"),
        ("dual_gating", "best"),
    ]

    print(f"\n{'═'*70}")
    print(f"TASK: {task_name} — train n={n_train} (T={2*(n_train+1)})")
    print(f"Test lengths: {[2*(n+1) for n in n_test_list]}")
    print(f"{'═'*70}")

    task_results = {}
    t0 = time.perf_counter()

    for mode_name, eps_strategy in modes:
        print(f"\n  Mode: {mode_name}")

        if mode_name == "baseline":
            # No eps to sweep
            seed_results = {n: [] for n in n_test_list}
            for seed in seeds:
                results = train_and_eval(
                    mode_name, VOCAB_SIZE, n_train, seed, 0.0,
                    gen_fn, train_steps, n_test_list)
                for n in n_test_list:
                    seed_results[n].append(results[n])
                elapsed = time.perf_counter() - t0
                print(f"    seed={seed}  "
                      f"[{', '.join(f'{results[n]:.3f}' for n in n_test_list)}]  "
                      f"({elapsed:.0f}s)")

            task_results[mode_name] = {
                "best_eps": None,
                "results": {n: {"mean": np.mean(seed_results[n]),
                                "std": np.std(seed_results[n]),
                                "vals": seed_results[n]}
                            for n in n_test_list},
            }
        else:
            # Sweep eps, pick best by training-length accuracy
            best_eps = None
            best_train_acc = -1
            best_seed_results = None

            for eps in eps_values:
                seed_results = {n: [] for n in n_test_list}
                for seed in seeds:
                    results = train_and_eval(
                        mode_name, VOCAB_SIZE, n_train, seed, eps,
                        gen_fn, train_steps, n_test_list)
                    for n in n_test_list:
                        seed_results[n].append(results[n])

                train_acc = np.mean(seed_results[n_train])
                elapsed = time.perf_counter() - t0
                print(f"    ε={eps:.2f}  "
                      f"[{', '.join(f'{np.mean(seed_results[n]):.3f}' for n in n_test_list)}]  "
                      f"({elapsed:.0f}s)")

                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    best_eps = eps
                    best_seed_results = seed_results

            task_results[mode_name] = {
                "best_eps": best_eps,
                "results": {n: {"mean": np.mean(best_seed_results[n]),
                                "std": np.std(best_seed_results[n]),
                                "vals": best_seed_results[n]}
                            for n in n_test_list},
            }

        # Print summary for this mode
        r = task_results[mode_name]
        eps_str = f" ε={r['best_eps']:.2f}" if r['best_eps'] else ""
        vals_str = "  ".join(
            f"T={2*(n+1)}:{r['results'][n]['mean']:.3f}±{r['results'][n]['std']:.3f}"
            for n in n_test_list)
        print(f"    → Best{eps_str}:  {vals_str}")

    # Comparison table
    print(f"\n  {'Mode':>20s}", end="")
    for n in n_test_list:
        t_len = 2 * (n + 1)
        ood = "*" if n > n_train else ""
        print(f"  {'T='+str(t_len)+ood:>12s}", end="")
    print()
    for mode_name, _ in modes:
        r = task_results[mode_name]
        eps_str = f"(ε={r['best_eps']:.2f})" if r['best_eps'] else ""
        print(f"  {mode_name+eps_str:>20s}", end="")
        for n in n_test_list:
            m = r['results'][n]
            print(f"  {m['mean']:.3f}±{m['std']:.3f}", end="")
        print()

    # Delta vs baseline
    baseline = task_results["baseline"]
    for mode_name in ["scalar_feedback", "dual_additive", "dual_gating"]:
        r = task_results[mode_name]
        deltas = []
        for n in n_test_list:
            d = r['results'][n]['mean'] - baseline['results'][n]['mean']
            deltas.append(f"{d:+.3f}")
        print(f"  Δ({mode_name}): {', '.join(deltas)}")

    return task_results, n_test_list


def run():
    print("=" * 70)
    print("DUAL-CHANNEL ATTENTION: Social Field Experiments")
    print(f"V={VOCAB_SIZE}, d={D_MODEL}, H={N_HEAD}, L={N_LAYER}")
    print(f"d_f={D_MODEL//N_HEAD} (= d_h, social field dimension per head)")
    print("=" * 70)

    t0 = time.perf_counter()
    all_results = {}

    # Task 1: Parity (validate mechanism)
    parity_results, parity_lengths = run_task(
        "parity", gen_parity, n_train=16, train_steps=5000)
    all_results["parity"] = (parity_results, parity_lengths)

    # Task 2: Cumsum (test capacity)
    cumsum_results, cumsum_lengths = run_task(
        "cumsum", gen_cumsum, n_train=16, train_steps=5000)
    all_results["cumsum"] = (cumsum_results, cumsum_lengths)

    # Task 3: Bracket matching (test geometric routing)
    bracket_results, bracket_lengths = run_task(
        "brackets", gen_brackets, n_train=16, train_steps=8000)
    all_results["brackets"] = (bracket_results, bracket_lengths)

    total_time = time.perf_counter() - t0
    print(f"\n{'═'*70}")
    print(f"Complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'═'*70}")

    # Plot
    tasks = list(all_results.keys())
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]

    mode_colors = {
        "baseline": "tab:gray",
        "scalar_feedback": "tab:blue",
        "dual_additive": "tab:green",
        "dual_gating": "tab:orange",
    }
    mode_markers = {
        "baseline": "s",
        "scalar_feedback": "o",
        "dual_additive": "D",
        "dual_gating": "^",
    }
    mode_labels = {
        "baseline": "Standard attention",
        "scalar_feedback": "Scalar feedback",
        "dual_additive": "Dual-channel (additive)",
        "dual_gating": "Dual-channel (gating)",
    }

    for ax, task_name in zip(axes, tasks):
        task_data, n_test_list = all_results[task_name]

        for mode_name in mode_colors:
            if mode_name not in task_data:
                continue
            r = task_data[mode_name]
            lengths = [2 * (n + 1) for n in n_test_list]
            vals = [r['results'][n]['mean'] for n in n_test_list]
            stds = [r['results'][n]['std'] for n in n_test_list]

            eps_str = f" ε={r['best_eps']:.2f}" if r['best_eps'] else ""
            ax.errorbar(lengths, vals, yerr=stds,
                        fmt=f"{mode_markers[mode_name]}-",
                        color=mode_colors[mode_name],
                        linewidth=2, markersize=8, capsize=4,
                        label=f"{mode_labels[mode_name]}{eps_str}")

        # Training length marker
        train_t = 2 * (n_test_list[0] + 1)
        ax.axvline(x=train_t, color="black", linestyle="--", alpha=0.3,
                   label=f"Train length")

        # Random baselines
        ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
                   label=f"Random (1/{VOCAB_SIZE})")
        if task_name == "parity":
            ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.3,
                       label="Random (binary)")

        ax.set_xlabel("Sequence length T")
        ax.set_ylabel("Accuracy (3 seeds)")
        ax.set_title(f"{task_name.title()}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Dual-Channel Attention: Social Field Experiments\n"
                 f"(V={VOCAB_SIZE}, d={D_MODEL}, L={N_LAYER}, H={N_HEAD}, "
                 f"RoPE, no learned PE)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dualchannel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: dualchannel.png")


if __name__ == "__main__":
    run()
