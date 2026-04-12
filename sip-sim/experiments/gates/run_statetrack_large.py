"""Focused state-tracking experiment at larger sequence lengths.

Cumsum was too easy (1.000 everywhere) because the model can attend to
the previous answer position — it's Markov in the output space.

Parity and flipflop are the real tests:
- Parity: not Markov for attention, requires aggregating over ALL inputs
- Flipflop: requires remembering the LAST non-zero write, arbitrarily far back

We skip short sequences and go straight to where attention should break.
"""

import sys
import os

from learn.gate_experiments import (
    VOCAB_SIZE, TRAIN_STEPS, BATCH_SIZE, D_MODEL, N_HEAD, N_LAYER,
    FIXED_DROPOUT, FIXED_WD, MEASURE_EVERY, BASE_LR,
    gen_parity, gen_flipflop, gen_cumsum,
    train_config,
    OUT_DIR,
)

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    print("=" * 70)
    print("STATE TRACKING (LARGE): Parity + Flipflop at Long Sequences")
    print("=" * 70)

    tasks = [
        ("parity", gen_parity),
        ("flipflop", gen_flipflop),
    ]

    modes = [
        ("no_field", False, "decay"),
        ("feedback", True, "feedback"),
    ]

    # Start big — skip the trivial short lengths
    n_examples_list = [32, 64, 128]
    eps_values = [0.05, 0.10, 0.20]
    seeds = [42, 137, 256]

    # More training steps for longer sequences
    steps_by_n = {32: 5000, 64: 8000, 128: 10000}

    all_results = {}
    t0 = time.perf_counter()

    for task_name, gen_fn in tasks:
        print(f"\n{'═'*70}")
        print(f"TASK: {task_name}")
        print(f"{'═'*70}")

        task_results = {}

        for n_ex in n_examples_list:
            t_len = 2 * (n_ex + 1)
            steps = steps_by_n[n_ex]
            print(f"\n  n_examples={n_ex} (T={t_len}, steps={steps})")
            print(f"  {'─'*50}")

            length_results = {}

            for mode_name, use_field, field_mode in modes:
                if not use_field:
                    vals = []
                    for seed in seeds:
                        val, _, _ = train_config(
                            VOCAB_SIZE, n_ex, seed, 0.0,
                            use_field=False,
                            pe_type="learned", use_rope=False,
                            generate_fn=gen_fn, gen_kwargs={},
                            train_steps=steps)
                        vals.append(val)
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    length_results[mode_name] = {
                        "best_eps": None, "mean_val": mean_val,
                        "std_val": std_val, "vals": vals,
                        "all_eps": [],
                    }
                    elapsed = time.perf_counter() - t0
                    print(f"    {mode_name:10s}  val={mean_val:.3f}±{std_val:.3f}  "
                          f"[{', '.join(f'{v:.3f}' for v in vals)}]  ({elapsed:.0f}s)")
                else:
                    best_mean = -1
                    best_eps = None
                    best_vals = None
                    best_std = 0
                    eps_results = []
                    for eps in eps_values:
                        vals = []
                        for seed in seeds:
                            val, _, _ = train_config(
                                VOCAB_SIZE, n_ex, seed, eps,
                                use_field=True,
                                pe_type="learned", use_rope=False,
                                generate_fn=gen_fn, gen_kwargs={},
                                field_mode=field_mode,
                                train_steps=steps)
                            vals.append(val)
                        mean_val = np.mean(vals)
                        std_val = np.std(vals)
                        eps_results.append({
                            "eps": eps, "mean_val": mean_val,
                            "std_val": std_val, "vals": vals,
                        })
                        if mean_val > best_mean:
                            best_mean = mean_val
                            best_eps = eps
                            best_vals = vals
                            best_std = std_val
                        elapsed = time.perf_counter() - t0
                        print(f"    {mode_name:10s} ε={eps:.2f}  "
                              f"val={mean_val:.3f}±{std_val:.3f}  "
                              f"[{', '.join(f'{v:.3f}' for v in vals)}]  "
                              f"({elapsed:.0f}s)")

                    length_results[mode_name] = {
                        "best_eps": best_eps,
                        "mean_val": best_mean,
                        "std_val": best_std,
                        "vals": best_vals,
                        "all_eps": eps_results,
                    }

            # Summary
            print(f"\n  Summary (n={n_ex}, T={t_len}):")
            for mode_name, _, _ in modes:
                r = length_results[mode_name]
                eps_str = f" ε={r['best_eps']:.2f}" if r['best_eps'] else ""
                print(f"    {mode_name:10s}{eps_str:8s}  "
                      f"val={r['mean_val']:.3f}±{r['std_val']:.3f}")

            baseline = length_results["no_field"]["mean_val"]
            feedback = length_results["feedback"]["mean_val"]
            delta = feedback - baseline
            print(f"    Δ(feedback - no_field) = {delta:+.3f}")
            if baseline > 0:
                pct = delta / baseline * 100
                print(f"    Relative improvement: {pct:+.1f}%")

            task_results[n_ex] = length_results

        all_results[task_name] = task_results

    total_time = time.perf_counter() - t0
    print(f"\n{'═'*70}")
    print(f"Complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'═'*70}")

    # Plot
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]

    mode_colors = {"no_field": "tab:gray", "feedback": "tab:green"}
    mode_markers = {"no_field": "s", "feedback": "D"}
    mode_labels = {"no_field": "Standard attention", "feedback": "Feedback field"}

    for ax, (task_name, _) in zip(axes, tasks):
        task_data = all_results[task_name]

        for mode_name, _, _ in modes:
            lengths = []
            vals = []
            stds = []
            for n_ex in n_examples_list:
                r = task_data[n_ex][mode_name]
                lengths.append(2 * (n_ex + 1))
                vals.append(r["mean_val"])
                stds.append(r["std_val"])

            ax.errorbar(lengths, vals, yerr=stds,
                        fmt=f"{mode_markers[mode_name]}-",
                        color=mode_colors[mode_name],
                        linewidth=2, markersize=10, capsize=5,
                        label=mode_labels[mode_name])

        ax.axhline(y=1/VOCAB_SIZE, color="red", linestyle=":", alpha=0.3,
                   label=f"Random (1/{VOCAB_SIZE})")
        # Parity random baseline is 0.5, not 1/V
        if task_name == "parity":
            ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.3,
                       label="Random (parity)")
        ax.set_xlabel("Sequence length T")
        ax.set_ylabel("Validation Accuracy (best ε, 3 seeds)")
        ax.set_title(f"{task_name.title()}\n(V={VOCAB_SIZE}, d={D_MODEL}, "
                     f"L={N_LAYER}, H={N_HEAD})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("State Tracking at Scale: Feedback Field vs Standard Attention",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "statetrack_large.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: statetrack_large.png")


if __name__ == "__main__":
    run()
