"""
Multitask Length Generalization: State-Capacity Scaling Hypothesis.

Tests whether the stigmergic field's advantage scales with task state complexity.
Four tasks with increasing state-space size:

  1. parity        — 1-bit state (mod 2)     → should work with H=4
  2. cumsum_mod4   — 2-bit state (mod 4)     → harder, still within H=4 capacity
  3. cumsum_mod16  — 4-bit state (mod 16)    → exceeds H=4 capacity → predicted failure
  4. flipflop      — selective overwrite      → 1-of-(V-1) state, tests gating

All tasks use the same interleaved format: x0 a0 x1 a1 ... xn an
where a_i is the answer (running state) after seeing x_0..x_i.

Compares: no_field (baseline), hierarchical_feedback (our best), S4D (upper bound).

Usage:
    uv run python -u learn/multitask_lengthgen.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import infrastructure from length_gen_sweep ─────────────────

from length_gen_sweep import (
    LengthGenGPT, make_config, generate_parity_batch, n_to_seqlen,
    VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, BATCH_SIZE, BASE_LR,
    TRAIN_STEPS, MEASURE_EVERY, FIXED_DROPOUT, FIXED_WD, SEEDS,
    DEFAULT_TRAIN_N, DEFAULT_TEST_NS,
    EVAP_FAST, EVAP_SLOW,
)
from ssm_baseline import S4DModel

OUT_DIR = Path(__file__).resolve().parent


# ── Task generators ─────────────────────────────────────────────
# All follow the same contract as generate_parity_batch:
#   returns (input_ids, targets, last_targets) tensors
#   format: x0 a0 x1 a1 ... xn an   (seq len = 2*(n_examples+1))
#   targets mask input positions with -100

def generate_cumsum_mod4_batch(batch_size, n_examples, vocab_size, rng):
    """Running cumulative sum mod 4.

    Input tokens are random in [0, V-1].
    Answer tokens are in {0, 1, 2, 3}.
    a_i = (x_0 + x_1 + ... + x_i) mod 4.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        running_sum = 0
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            running_sum = (running_sum + x) % 4
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = running_sum

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    # Only predict answer positions (odd indices in input_ids → even indices in targets)
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]

    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


def generate_cumsum_mod16_batch(batch_size, n_examples, vocab_size, rng):
    """Running cumulative sum mod 16.

    Input tokens are random in [0, V-1].
    Answer tokens are in {0, 1, ..., 15}.
    a_i = (x_0 + x_1 + ... + x_i) mod 16.
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        running_sum = 0
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            running_sum = (running_sum + x) % 16
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = running_sum

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]

    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


def generate_flipflop_batch(batch_size, n_examples, vocab_size, rng):
    """Selective overwrite (flip-flop) task.

    Input tokens are random in [0, V-1].
    State starts at 0. When x_i != 0, state updates to x_i.
    When x_i == 0, state holds (no-op / "hold" signal).
    Answer is the current state value.

    a_i = x_i if x_i != 0 else a_{i-1}   (a_{-1} = 0)
    """
    n_total = n_examples + 1
    full_len = 2 * n_total
    tokens = np.zeros((batch_size, full_len), dtype=np.int64)

    for b in range(batch_size):
        state = 0
        for i in range(n_total):
            x = int(rng.integers(0, vocab_size))
            if x != 0:
                state = x
            tokens[b, 2 * i] = x
            tokens[b, 2 * i + 1] = state

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:].copy()
    for i in range(1, full_len - 1, 2):
        targets[:, i] = -100
    last_targets = tokens[:, -1]

    return (
        torch.tensor(input_ids),
        torch.tensor(targets),
        torch.tensor(last_targets),
    )


# ── Task registry ───────────────────────────────────────────────

TASKS = {
    "parity": {
        "generator": generate_parity_batch,
        "n_answers": 2,
        "description": "mod-2 running sum (1-bit state)",
    },
    "cumsum_mod4": {
        "generator": generate_cumsum_mod4_batch,
        "n_answers": 4,
        "description": "mod-4 running sum (2-bit state)",
    },
    "cumsum_mod16": {
        "generator": generate_cumsum_mod16_batch,
        "n_answers": 16,
        "description": "mod-16 running sum (4-bit state)",
    },
    "flipflop": {
        "generator": generate_flipflop_batch,
        "n_answers": VOCAB_SIZE,
        "description": "selective overwrite (gating)",
    },
}

TASK_NAMES = list(TASKS.keys())

# Configs to compare: baseline, our best, SSM upper bound
CONFIG_NAMES = ["no_field", "hierarchical_feedback", "s4d"]


# ── Model builders ──────────────────────────────────────────────

def build_model(config_name, max_pe_len=256):
    """Build a model for the given config name."""
    if config_name == "s4d":
        return S4DModel(
            VOCAB_SIZE, D_MODEL, N_LAYER,
            state_dim=64, dropout=FIXED_DROPOUT,
        )
    else:
        model_kwargs = make_config(config_name, max_pe_len=max_pe_len)
        return LengthGenGPT(**model_kwargs)


# ── Training + Evaluation ──────────────────────────────────────

def train_and_evaluate_task(task_name, config_name, train_n, test_ns,
                            seed=42, verbose=False):
    """Train on task at train_n, evaluate at each test length."""
    generator = TASKS[task_name]["generator"]
    max_test_n = max(test_ns)
    max_seq = n_to_seqlen(max_test_n) - 1

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = build_model(config_name, max_pe_len=max_seq + 16)
    n_params = model.count_params()
    train_T = n_to_seqlen(train_n)

    if verbose:
        print(f"  [{task_name}] {config_name}, params: {n_params:,}, "
              f"train n={train_n} (T={train_T}), seed={seed}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                                  weight_decay=FIXED_WD, betas=(0.85, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=TRAIN_STEPS)

    for step in range(TRAIN_STEPS):
        model.train()
        inp, tgt, labels = generator(BATCH_SIZE, train_n, VOCAB_SIZE, rng)
        logits = model(inp)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if verbose and step % (MEASURE_EVERY * 4) == 0:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1].argmax(-1)
                train_acc = (pred == labels).float().mean().item()
            print(f"    step {step:5d}  loss={loss.item():.3f}  "
                  f"train_last={train_acc:.3f}")
            model.train()

    # Evaluate at each test length
    model.eval()
    N_EVAL = 1024
    length_results = {}

    with torch.no_grad():
        for test_n in test_ns:
            eval_rng = np.random.default_rng(12345)
            eval_in, _, eval_lab = generator(N_EVAL, test_n, VOCAB_SIZE, eval_rng)
            eval_logits = model(eval_in)
            eval_pred = eval_logits[:, -1].argmax(-1)
            acc = (eval_pred == eval_lab).float().mean().item()
            length_results[test_n] = acc

            if verbose:
                test_T = n_to_seqlen(test_n)
                ratio = test_T / train_T
                ood = " (OOD)" if test_n > train_n else ""
                print(f"    n={test_n:3d} T={test_T:3d} ({ratio:.1f}x){ood}: "
                      f"acc={acc:.3f}")

    summary = {
        "task": task_name,
        "config": config_name,
        "seed": seed,
        "train_n": train_n,
        "n_params": n_params,
        **{f"acc_n{n}": a for n, a in length_results.items()},
    }

    return length_results, summary


# ── Full multitask sweep ────────────────────────────────────────

def run_multitask_sweep(train_n=DEFAULT_TRAIN_N, test_ns=None):
    if test_ns is None:
        test_ns = DEFAULT_TEST_NS

    train_T = n_to_seqlen(train_n)
    test_Ts = [n_to_seqlen(n) for n in test_ns]

    print(f"Multitask Length Generalization Sweep")
    print(f"Train: n={train_n} (T={train_T})")
    print(f"Test:  n={test_ns} -> T={test_Ts}")
    print(f"Tasks: {TASK_NAMES}")
    print(f"Configs: {CONFIG_NAMES}")
    print(f"Seeds: {SEEDS}")
    print()

    # Print model sizes
    max_pe = n_to_seqlen(max(test_ns)) + 16
    for name in CONFIG_NAMES:
        m = build_model(name, max_pe_len=max_pe)
        print(f"  {name:30s}  params={m.count_params():,}")
        del m
    print()

    all_results = []
    all_length_results = {}  # (task, config, seed) -> {n: acc}

    t0 = time.perf_counter()
    total_runs = len(TASK_NAMES) * len(CONFIG_NAMES) * len(SEEDS)
    idx = 0

    for task_name in TASK_NAMES:
        print(f"\n{'='*60}")
        print(f"Task: {task_name} — {TASKS[task_name]['description']}")
        print(f"  Answer space: {TASKS[task_name]['n_answers']} tokens")
        print(f"{'='*60}")

        for config_name in CONFIG_NAMES:
            for seed in SEEDS:
                idx += 1
                length_res, summary = train_and_evaluate_task(
                    task_name, config_name, train_n, test_ns,
                    seed=seed, verbose=True)

                all_results.append(summary)
                all_length_results[(task_name, config_name, seed)] = length_res

                elapsed = time.perf_counter() - t0
                eta = (elapsed / idx) * (total_runs - idx)

                acc_str = "  ".join(
                    f"n{n}={length_res[n]:.2f}" for n in test_ns)
                print(f"  [{idx:2d}/{total_runs}] {task_name}/{config_name} "
                      f"s={seed}  {acc_str}  "
                      f"({elapsed:.0f}s, ~{eta:.0f}s left)")
                print()

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")

    print_multitask_summary(all_results, test_ns)
    plot_multitask_results(all_results, all_length_results, train_n, test_ns)


def print_multitask_summary(results, test_ns):
    """Print summary table for each task."""
    for task_name in TASK_NAMES:
        print(f"\n{'='*70}")
        print(f"  {task_name} — {TASKS[task_name]['description']}")
        print(f"{'='*70}")

        header = f"{'Config':30s}"
        for n in test_ns:
            T = n_to_seqlen(n)
            header += f"  {'n='+str(n)+f'(T={T})':>14s}"
        print(header)
        print("-" * (30 + 16 * len(test_ns)))

        for config_name in CONFIG_NAMES:
            rows = [r for r in results
                    if r["task"] == task_name and r["config"] == config_name]
            if not rows:
                continue
            line = f"{config_name:30s}"
            for n in test_ns:
                key = f"acc_n{n}"
                vals = [r[key] for r in rows]
                m, s = np.mean(vals), np.std(vals)
                line += f"  {m:.3f}+/-{s:.3f}"
            print(line)

        # Generalization gap
        n_train = test_ns[0]
        n_max = test_ns[-1]
        print(f"\n  Gap (n={n_train} -> n={n_max}):")
        for config_name in CONFIG_NAMES:
            rows = [r for r in results
                    if r["task"] == task_name and r["config"] == config_name]
            if not rows:
                continue
            acc_train = np.mean([r[f"acc_n{n_train}"] for r in rows])
            acc_max = np.mean([r[f"acc_n{n_max}"] for r in rows])
            gap = acc_train - acc_max
            print(f"    {config_name:30s}  {acc_train:.3f} -> {acc_max:.3f}  "
                  f"(gap={gap:+.3f})")


def plot_multitask_results(results, all_length_results, train_n, test_ns):
    """Create combined plot with one subplot per task."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    colors = {"no_field": "#1f77b4", "hierarchical_feedback": "#d62728",
              "s4d": "#2ca02c"}
    markers = {"no_field": "s", "hierarchical_feedback": "D", "s4d": "^"}
    train_T = n_to_seqlen(train_n)

    for ti, task_name in enumerate(TASK_NAMES):
        ax = axes[ti]
        n_answers = TASKS[task_name]["n_answers"]
        random_baseline = 1.0 / n_answers

        for config_name in CONFIG_NAMES:
            accs_by_n = {n: [] for n in test_ns}
            for seed in SEEDS:
                key = (task_name, config_name, seed)
                if key in all_length_results:
                    for n, a in all_length_results[key].items():
                        accs_by_n[n].append(a)

            test_Ts = [n_to_seqlen(n) for n in test_ns]
            means = [np.mean(accs_by_n[n]) for n in test_ns]
            stds = [np.std(accs_by_n[n]) for n in test_ns]

            ax.errorbar(test_Ts, means, yerr=stds,
                        fmt=markers[config_name] + "-",
                        color=colors[config_name], label=config_name,
                        capsize=3, linewidth=2, markersize=7)

        ax.axvline(x=train_T, color="gray", linestyle="--", alpha=0.7)
        ax.axhline(y=random_baseline, color="gray", linestyle=":",
                   alpha=0.5, label=f"random (1/{n_answers})")
        ax.set_xlabel("Sequence length T")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task_name}\n({TASKS[task_name]['description']})")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"Multitask Length Generalization: State-Capacity Scaling\n"
        f"Train n={train_n} (T={train_T}) | "
        f"{N_LAYER}L {N_HEAD}H d={D_MODEL} V={VOCAB_SIZE} | "
        f"{len(SEEDS)} seeds",
        fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / f"multitask_lengthgen_n{train_n}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_multitask_sweep()
