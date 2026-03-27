#!/usr/bin/env python3
"""
Plotting utilities for rdft-sim BRW simulation results.

Reads JSON output from the Rust simulator and produces:
  1. Log-log scaling plots: ⟨V^p⟩ vs t for each graph type
  2. Convergence plots: exponent estimate vs number of realizations
  3. Theory comparison bar chart

Usage:
    python plot_brw.py results.json                 # Plot from saved file
    cargo run --release | python plot_brw.py -      # Pipe from simulator
    python plot_brw.py                              # Build, run, and plot
"""

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
    if path == '-':
        return json.load(sys.stdin)
    with open(path) as f:
        return json.load(f)


def run_simulator(args: list = None) -> dict:
    sim_dir = Path(__file__).parent.parent
    print("Building simulator (release mode)...")
    subprocess.run(["cargo", "build", "--release"], cwd=sim_dir, check=True)
    binary = sim_dir / "target" / "release" / "rdft-sim"
    cmd = [str(binary)] + (args or [])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(result.stderr)
    return json.loads(result.stdout)


# ── Plot 1: Scaling (log-log) ─────────────────────────────────────

def plot_scaling(runs: list, save_path: str = None):
    """Log-log plots of ⟨V^p⟩ and ⟨V^p|surv⟩ vs t."""
    n_runs = len(runs)
    cols = min(3, n_runs)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)

    for idx, run in enumerate(runs):
        ax = axes[idx // cols][idx % cols]
        times = np.array(run['times'])
        max_p = len(run.get('moments_surviving', run.get('moments', [])))

        for p in range(max_p):
            # Conditional moments (cleaner signal)
            key = 'moments_surviving' if 'moments_surviving' in run else 'moments'
            vals = np.array(run[key][p])
            mask = (times > 1) & (vals > 0)
            if mask.sum() < 2:
                continue

            t_plot = times[mask]
            v_plot = vals[mask]

            ax.loglog(t_plot, v_plot, '-', color=COLORS[p % len(COLORS)],
                      linewidth=1.5, label=f'p={p+1} (surv)' if key == 'moments_surviving' else f'p={p+1}')

            # Theory reference line
            exps = run.get('fitted_exponents', [])
            if p < len(exps):
                alpha_th = exps[p].get('alpha_theory_cond', exps[p].get('alpha_theory'))
                if alpha_th is not None and alpha_th != 0:
                    mid = len(t_plot) // 2
                    t_mid, v_mid = t_plot[mid], v_plot[mid]
                    t_ref = t_plot[len(t_plot)//4 : 3*len(t_plot)//4]
                    v_ref = v_mid * (t_ref / t_mid) ** alpha_th
                    ax.loglog(t_ref, v_ref, '--', color=COLORS[p % len(COLORS)],
                              alpha=0.4, linewidth=1)

        ax.set_xlabel('t')
        ax.set_ylabel('⟨V^p | survived⟩')
        d_s = run.get('d_s', '?')
        ax.set_title(f"{run['graph']} (d_s={d_s})")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, which='both')

        # Annotate exponents
        exps = run.get('fitted_exponents', [])
        lines = []
        for ex in exps:
            ac = ex.get('alpha_cond', ex.get('alpha_surviving', ex.get('alpha_measured', '?')))
            th = ex.get('alpha_theory_cond', ex.get('alpha_theory'))
            th_s = f'{th:.2f}' if th is not None else '?'
            lines.append(f"p={ex['p']}: {ac:.2f} (th {th_s})")
        if lines:
            ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=7, va='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    for idx in range(n_runs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('BRW Scaling: ⟨V^p | survived⟩(t) on Various Graphs', fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ── Plot 2: Convergence ──────────────────────────────────────────

def plot_convergence(runs: list, save_path: str = None):
    """Exponent estimate vs number of realizations."""
    n_runs = len(runs)
    cols = min(3, n_runs)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for idx, run in enumerate(runs):
        ax = axes[idx // cols][idx % cols]
        convergence = run.get('convergence', [])
        exps = run.get('fitted_exponents', [])

        for p, conv_data in enumerate(convergence):
            if not conv_data:
                continue
            ns = [c[0] for c in conv_data]
            alphas = [c[1] for c in conv_data]

            ax.plot(ns, alphas, 'o-', color=COLORS[p % len(COLORS)],
                    markersize=3, linewidth=1, label=f'p={p+1}')

            # Theory line
            if p < len(exps):
                th = exps[p].get('alpha_theory_uncond', exps[p].get('alpha_theory'))
                if th is not None:
                    ax.axhline(y=th, color=COLORS[p % len(COLORS)],
                               linestyle='--', alpha=0.4, linewidth=0.8)

        ax.set_xlabel('N (realizations)')
        ax.set_ylabel('α (fitted exponent)')
        d_s = run.get('d_s', '?')
        ax.set_title(f"{run['graph']} (d_s={d_s})")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    for idx in range(n_runs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('Convergence of Exponent Estimates over N Realizations', fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ── Plot 3: Theory comparison ────────────────────────────────────

def plot_comparison(runs: list, save_path: str = None):
    """Bar chart: measured vs theoretical conditional exponents."""
    labels, measured, theory = [], [], []

    for run in runs:
        for ex in run.get('fitted_exponents', []):
            th = ex.get('alpha_theory_cond', ex.get('alpha_theory'))
            meas = ex.get('alpha_cond', ex.get('alpha_surviving', ex.get('alpha_measured')))
            if th is None or meas is None:
                continue
            graph_short = run['graph'].split('(')[0].strip()
            labels.append(f"{graph_short}\np={ex['p']}")
            measured.append(meas)
            theory.append(th)

    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5))
    ax.bar(x - width/2, theory, width, label='Theory (p·d_s/2)', color='#4a90d9', alpha=0.8)
    ax.bar(x + width/2, measured, width, label='Simulation', color='#e8833a', alpha=0.8)

    ax.set_ylabel('Conditional exponent α_cond')
    ax.set_title('Theory vs Simulation: BRW Conditional Scaling Exponents')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ── Summary table ─────────────────────────────────────────────────

def print_summary(runs: list):
    print()
    print("CONDITIONAL EXPONENTS (surviving walks): α_cond/p → d_s/2")
    print(f"{'Graph':<35} {'d_s':>5} {'p':>3} {'α_cond':>8} {'Theory':>8} {'α/p':>8}")
    print("-" * 72)
    for run in runs:
        for ex in run.get('fitted_exponents', []):
            th = ex.get('alpha_theory_cond', '?')
            ac = ex.get('alpha_cond', ex.get('alpha_surviving', '?'))
            ratio = ac / ex['p'] if isinstance(ac, (int, float)) else '?'
            d_s = run.get('d_s', '?')
            print(f"  {run['graph']:<33} {d_s:>5} {ex['p']:>3} {ac:>8.3f} {th:>8.3f} {ratio:>8.3f}")

    print()
    print("TIMING")
    for run in runs:
        surv = run.get('n_survived', '?')
        tot = run.get('n_realizations', '?')
        wt = run.get('wall_time_secs', '?')
        print(f"  {run['graph']:<35} {surv}/{tot} survived, {wt:.1f}s")


# ── Main ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        data = load_results(sys.argv[1])
    else:
        data = run_simulator()

    runs = data.get('runs', [data])

    print_summary(runs)

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    plot_scaling(runs, str(out_dir / "scaling.png"))
    plot_convergence(runs, str(out_dir / "convergence.png"))
    plot_comparison(runs, str(out_dir / "comparison.png"))

    plt.show()


if __name__ == '__main__':
    main()
