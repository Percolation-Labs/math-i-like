#!/usr/bin/env python3
"""
Plot prion propagation simulation results.
Generates figures for the paper.
"""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def plot_R0_scan(csv_path: str, output_dir: str = 'results'):
    """Plot survival fraction and density exponent vs R0."""
    R0s, betas, survivals, alphas = [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            R0s.append(float(row['R0']))
            betas.append(float(row['beta']))
            surv = int(row['survived']) if row['survived'] else 0
            n = int(row['n_total'])
            survivals.append(surv / n)
            alpha = float(row['alpha_p1']) if row['alpha_p1'] else np.nan
            alphas.append(alpha)

    R0s = np.array(R0s)
    survivals = np.array(survivals)
    alphas = np.array(alphas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Survival fraction vs R0
    ax1.plot(R0s, survivals, 'ko-', markersize=5)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_xlabel(r'$R_0 = \beta\lambda/(\mu_H\mu_M)$', fontsize=12)
    ax1.set_ylabel('Survival fraction', fontsize=12)
    ax1.set_title('(a) Absorbing-state transition', fontsize=12)

    # Mark mean-field critical point
    ax1.axvline(1.0, color='blue', linestyle='--', alpha=0.5, label=r'$R_0^{\rm MF}=1$')

    # Estimate true critical R0 (where survival first appears)
    for i in range(len(survivals)):
        if survivals[i] > 0.01:
            R0c = R0s[i]
            ax1.axvline(R0c, color='red', linestyle='--', alpha=0.5,
                       label=f'$R_0^c \\approx {R0c:.1f}$')
            break
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, max(R0s) * 1.05)

    # Panel (b): Volume exponent vs R0
    valid = ~np.isnan(alphas)
    ax2.plot(R0s[valid], alphas[valid], 'rs-', markersize=5)
    ax2.axhline(-0.159, color='green', linestyle='--', alpha=0.7,
                label=r'DP: $\delta=0.159$')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel(r'$R_0$', fontsize=12)
    ax2.set_ylabel(r'$\alpha_1$ (volume exponent)', fontsize=12)
    ax2.set_title('(b) Scaling exponent', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, max(R0s) * 1.05)

    plt.tight_layout()
    out = Path(output_dir) / 'prion_1d_scan.pdf'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f'Saved {out} and {out.with_suffix(".png")}')
    plt.close()


def plot_density_decay(json_path: str, output_dir: str = 'results'):
    """Plot M density vs time for a single run."""
    with open(json_path) as f:
        data = json.load(f)

    times = np.array(data['times'])
    # Population is tracked species (M for prion)
    # Moments are volume moments; population is in the raw realization data
    # For now use the volume moments as proxy
    moments = data['moments']

    fig, ax = plt.subplots(figsize=(6, 4))

    for p_idx, label in enumerate([r'$\langle V \rangle$', r'$\langle V^2 \rangle$', r'$\langle V^3 \rangle$']):
        vals = np.array(moments[p_idx])
        mask = (times > 1) & (vals > 0)
        ax.loglog(times[mask], vals[mask], '-', linewidth=1.5, label=label)

    # Theory lines
    d_s = data.get('d_s', 1.0)
    for p in [1, 2, 3]:
        alpha = (p * d_s - 2) / 2
        if alpha > -2:
            t_th = np.logspace(1, np.log10(times.max()), 50)
            ax.loglog(t_th, t_th**alpha * 0.5, '--', color='gray', alpha=0.5)

    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel(r'$\langle V^p \rangle$', fontsize=12)
    ax.set_title(f'Prion on {data.get("graph", "lattice")} ($d_s$={d_s})', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = Path(output_dir) / 'prion_scaling.pdf'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path.endswith('.csv'):
            plot_R0_scan(path)
        else:
            plot_density_decay(path)
    else:
        # Default: look for scan CSV
        if Path('results/prion_1d_scan.csv').exists():
            plot_R0_scan('results/prion_1d_scan.csv')
        else:
            print("Usage: python plot_prion.py results/prion_1d_scan.csv")
