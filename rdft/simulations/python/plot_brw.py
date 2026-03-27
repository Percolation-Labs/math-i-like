#!/usr/bin/env python3
"""
Plotting utilities for rdft-sim BRW simulation results.

Reads JSON output from the Rust simulator and produces:
  1. Log-log scaling plots: ⟨V^p⟩ vs t for each graph type
  2. Convergence plots: exponent estimate vs number of realizations
  3. Theory comparison table

Usage:
    python plot_brw.py results.json                 # Plot from file
    cargo run --release | python plot_brw.py -      # Pipe from simulator
    python plot_brw.py                              # Run simulator then plot
"""

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 120,
    'figure.facecolor': 'white',
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def load_results(path: str) -> dict:
    """Load JSON results from file or stdin."""
    if path == '-':
        return json.load(sys.stdin)
    with open(path) as f:
        return json.load(f)


def run_simulator(args: list[str] = None) -> dict:
    """Build and run the Rust simulator, return parsed JSON."""
    sim_dir = Path(__file__).parent.parent
    print("Building simulator (release mode)...")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=sim_dir,
        check=True,
    )
    binary = sim_dir / "target" / "release" / "rdft-sim"
    cmd = [str(binary)] + (args or [])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(result.stderr)
    return json.loads(result.stdout)


# ── Plotting functions ─────────────────────────────────────────────

def plot_scaling(runs: list[dict], save_path: str = None):
    """Log-log plots of ⟨V^p⟩ vs t for each graph type."""
    n_runs = len(runs)
    cols = min(3, n_runs)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, run in enumerate(runs):
        ax = axes[idx // cols][idx % cols]
        times = np.array(run['times'])
        max_p = len(run['moments'])

        for p in range(max_p):
            vals = np.array(run['moments'][p])
            mask = (times > 0) & (vals > 0)
            if mask.sum() < 2:
                continue

            t_plot = times[mask]
            v_plot = vals[mask]

            # Measured data
            ax.loglog(t_plot, v_plot, '-', color=COLORS[p % len(COLORS)],
                      linewidth=1.5, label=f'p={p+1}')

            # Theory line (if available)
            exponents = run.get('fitted_exponents', [])
            if p < len(exponents) and exponents[p].get('alpha_theory') is not None:
                alpha_th = exponents[p]['alpha_theory']
                # Anchor at midpoint
                mid = len(t_plot) // 2
                t_mid, v_mid = t_plot[mid], v_plot[mid]
                t_ref = t_plot[len(t_plot)//5 : 4*len(t_plot)//5]
                v_ref = v_mid * (t_ref / t_mid) ** alpha_th
                ax.loglog(t_ref, v_ref, '--', color=COLORS[p % len(COLORS)],
                          alpha=0.5, linewidth=1)

        ax.set_xlabel('t')
        ax.set_ylabel('⟨V^p⟩')
        ax.set_title(run['graph'])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Annotate fitted exponents
        exponents = run.get('fitted_exponents', [])
        text_lines = []
        for ex in exponents:
            th = ex.get('alpha_theory')
            th_str = f'{th:.2f}' if th is not None else '?'
            text_lines.append(f"p={ex['p']}: α={ex['alpha_measured']:.2f} (th: {th_str})")
        if text_lines:
            ax.text(0.02, 0.02, '\n'.join(text_lines),
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # Hide empty axes
    for idx in range(n_runs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('BRW Scaling: ⟨V^p⟩(t) on Various Graphs', fontsize=15, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved scaling plot to {save_path}")
    return fig


def plot_convergence(runs: list[dict], save_path: str = None):
    """Exponent estimate vs number of realizations (convergence diagnostic)."""
    n_runs = len(runs)
    cols = min(3, n_runs)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for idx, run in enumerate(runs):
        ax = axes[idx // cols][idx % cols]
        convergence = run.get('convergence', [])
        exponents = run.get('fitted_exponents', [])

        for p, conv_data in enumerate(convergence):
            if not conv_data:
                continue
            ns = [c[0] for c in conv_data]
            alphas = [c[1] for c in conv_data]

            ax.plot(ns, alphas, 'o-', color=COLORS[p % len(COLORS)],
                    markersize=3, linewidth=1, label=f'p={p+1}')

            # Theory horizontal line
            if p < len(exponents) and exponents[p].get('alpha_theory') is not None:
                alpha_th = exponents[p]['alpha_theory']
                ax.axhline(y=alpha_th, color=COLORS[p % len(COLORS)],
                           linestyle='--', alpha=0.4)

        ax.set_xlabel('N (surviving realizations)')
        ax.set_ylabel('α (fitted exponent)')
        ax.set_title(run['graph'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    for idx in range(n_runs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('Convergence of Exponent Estimates', fontsize=15, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved convergence plot to {save_path}")
    return fig


def plot_theory_comparison(runs: list[dict], save_path: str = None):
    """Bar chart comparing measured vs theoretical exponents."""
    labels = []
    measured = []
    theory = []
    errors = []

    for run in runs:
        for ex in run.get('fitted_exponents', []):
            if ex.get('alpha_theory') is None:
                continue
            labels.append(f"{run['graph']}\np={ex['p']}")
            measured.append(ex['alpha_measured'])
            theory.append(ex['alpha_theory'])
            errors.append(abs(ex['alpha_measured'] - ex['alpha_theory']))

    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    bars1 = ax.bar(x - width/2, theory, width, label='Theory', color='#4a90d9', alpha=0.8)
    bars2 = ax.bar(x + width/2, measured, width, label='Measured', color='#e8833a', alpha=0.8)

    ax.set_ylabel('Exponent α')
    ax.set_title('Theory vs Simulation: BRW Scaling Exponents')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved comparison plot to {save_path}")
    return fig


def print_summary_table(runs: list[dict]):
    """Print a formatted comparison table."""
    print()
    print(f"{'Graph':<32} {'p':>3} {'Theory':>10} {'Measured':>10} {'Error':>10}")
    print("─" * 70)
    for run in runs:
        for ex in run.get('fitted_exponents', []):
            th = ex.get('alpha_theory')
            th_str = f"{th:.3f}" if th is not None else "?"
            err = abs(ex['alpha_measured'] - th) if th is not None else float('nan')
            mark = "✓" if err < 0.15 else ("~" if err < 0.3 else "✗")
            print(f"  {run['graph']:<30} {ex['p']:>3} {th_str:>10} "
                  f"{ex['alpha_measured']:>10.3f} {err:>9.3f} {mark}")
    print()
    for run in runs:
        print(f"  {run['graph']}: {run['n_survived']}/{run['n_realizations']} survived, "
              f"{run['wall_time_secs']:.1f}s")


# ── Main ───────────────────────────────────────────────────────────

def main():
    # Load or run
    if len(sys.argv) > 1:
        data = load_results(sys.argv[1])
    else:
        data = run_simulator()

    runs = data.get('runs', [data])  # Handle both suite and single-run output

    # Print summary
    print_summary_table(runs)

    # Generate plots
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    plot_scaling(runs, str(out_dir / "scaling.png"))
    plot_convergence(runs, str(out_dir / "convergence.png"))
    plot_theory_comparison(runs, str(out_dir / "comparison.png"))

    plt.show()


if __name__ == '__main__':
    main()
