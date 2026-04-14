"""
Generate figures for the canonical ant paper:
  (a) snapshots of canonical S/R ant at four densities (low → high)
  (b) f_R(n): naive MF vs local MF vs lattice
  (c) the correlation illustration: <phi|S> vs <phi>_global
  (d) trail/order-parameter transition as lambda sweeps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from math import sqrt

# Import the canonical sim
import sys
sys.path.insert(0, os.path.dirname(__file__))
from cfac_ants_standalone import run_lattice, mean_field_f_A


FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..',
                      'rdft', 'paper', 'wip', 'figures_ants')
FIGDIR = os.path.abspath(FIGDIR)
os.makedirs(FIGDIR, exist_ok=True)

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 11


# ─────────────────────────────────────────────────────────────────
# Snapshot runs: capture final state
# ─────────────────────────────────────────────────────────────────
def snapshot_run(n, L=80, n_steps=2000, seed=42,
                 k_plus=0.01, k_minus=0.05, chi=1.0,
                 delta=2.0, lam=0.10, sigma=0.005):
    """Same as run_lattice but returns the final field/state grid."""
    rng = np.random.default_rng(seed)
    occ = rng.random((L, L)) < n
    is_A = rng.random((L, L)) < 0.5
    is_A = is_A & occ
    is_B = occ & ~is_A
    phi = np.zeros((L, L), dtype=np.float64)

    def laplacian(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    for t in range(n_steps):
        rate_BA = k_plus * (1.0 + chi * phi)
        p_BA = np.clip(rate_BA, 0.0, 1.0)
        convert_BA = is_B & (rng.random((L, L)) < p_BA)
        convert_AB = is_A & (rng.random((L, L)) < k_minus)
        new_is_A = (is_A & ~convert_AB) | convert_BA
        new_is_B = (is_B & ~convert_BA) | convert_AB
        is_A, is_B = new_is_A, new_is_B
        phi = phi + delta * is_A.astype(np.float64)
        phi = phi * (1.0 - lam)
        phi = phi + sigma * laplacian(phi)

    n_A = int(is_A.sum())
    n_B = int(is_B.sum())
    return {
        'n': n, 'L': L,
        'is_R': is_A.copy(),
        'is_S': is_B.copy(),
        'phi': phi.copy(),
        'n_R': n_A, 'n_S': n_B,
        'f_R': n_A / (n_A + n_B) if (n_A + n_B) > 0 else 0.0,
    }


def make_snapshots_figure():
    """4 densities × 2 rows (ant map / field heatmap)."""
    densities = [0.05, 0.15, 0.30, 0.50]
    snaps = []
    for d in densities:
        print(f"  snapshot n={d}...", flush=True)
        snaps.append(snapshot_run(d))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5),
                              constrained_layout=True)

    for col, snap in enumerate(snaps):
        L = snap['L']
        # Top: agent map
        ax = axes[0, col]
        rgb = np.ones((L, L, 3)) * 0.94   # empty → very light grey
        rgb[snap['is_R']] = [0.85, 0.2, 0.2]   # R = red
        rgb[snap['is_S']] = [0.2, 0.4, 0.85]   # S = blue
        ax.imshow(rgb, origin='upper', interpolation='nearest')
        ax.set_title(f"n = {snap['n']:.2f}\n"
                     f"$f_R$ = {snap['f_R']:.3f}   "
                     f"(R={snap['n_R']}, S={snap['n_S']})")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Agents\n(red=R, blue=S, grey=empty)")

        # Bottom: field heatmap
        ax = axes[1, col]
        phi = snap['phi']
        vmax = max(phi.max(), 0.5)
        im = ax.imshow(phi, cmap='hot', origin='upper',
                       interpolation='nearest',
                       vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$\langle\phi\rangle$={phi.mean():.2f}   "
                     rf"max={phi.max():.1f}")
        if col == 0:
            ax.set_ylabel(r"Pheromone $\phi$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(r"Canonical ant ($S\rightleftharpoons R$ with "
                 r"linear rate $k_+(1+\chi\phi)$): snapshots across "
                 r"densities $n$",
                 fontsize=12)
    out = os.path.join(FIGDIR, 'canonical_ant_snapshots.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────
# Figure B: naive vs local MF vs lattice
# ─────────────────────────────────────────────────────────────────
def make_mf_comparison_figure():
    """Plot f_R vs n for the three curves + gap."""
    densities = np.array([0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40])
    mf_naive = np.array([mean_field_f_A(n, 0.01, 0.05, 1.0, 2.0, 0.10)
                          for n in densities])

    print("\nRunning lattice for MF comparison figure...")
    lat = []
    phi_mean = []
    phi_S = []
    for n in densities:
        print(f"  n={n:.2f}...", flush=True)
        runs = [run_lattice(n, seed=42 + s * 13) for s in range(3)]
        lat.append(np.mean([r['f_A_mean'] for r in runs]))
        phi_mean.append(np.mean([r['phi_mean'] for r in runs]))
        phi_S.append(np.mean([r['phi_B'] for r in runs]))
    lat = np.array(lat)
    phi_mean = np.array(phi_mean)
    phi_S = np.array(phi_S)

    # Local MF from measured <phi|S>
    k_p, k_m, chi = 0.01, 0.05, 1.0
    rate_local = k_p * (1 + chi * phi_S)
    mf_local = rate_local / (rate_local + k_m)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                              constrained_layout=True)

    # Left: f_R curves
    ax = axes[0]
    ax.plot(densities, mf_naive, 'o--', color='#d62728',
            label=r'Naive MF ($\langle\phi\rangle$)', lw=1.6, ms=6)
    ax.plot(densities, mf_local, 's-', color='#2ca02c',
            label=r'Local MF ($\langle\phi|S\rangle$)', lw=1.8, ms=6)
    ax.plot(densities, lat, 'kD', label='Lattice (3-seed mean)',
            ms=6.5, mfc='k')
    ax.set_xlabel(r'Density $n$')
    ax.set_ylabel(r'Recruiter fraction $f_R$')
    ax.set_title('Tree-level MF vs lattice: canonical ant')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.3)

    # Right: <phi|S> vs <phi>
    ax = axes[1]
    ax.plot(densities, phi_mean, 'o-', color='#1f77b4',
            label=r'$\langle\phi\rangle$ (global)', lw=1.8, ms=6)
    ax.plot(densities, phi_S, 's-', color='#ff7f0e',
            label=r'$\langle\phi\mid S\rangle$ (conditional)',
            lw=1.8, ms=6)
    ax.set_xlabel(r'Density $n$')
    ax.set_ylabel(r'Pheromone average')
    ax.set_title(r'Conditional correlation: $\langle\phi|S\rangle\gg\langle\phi\rangle$')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.3)

    out = os.path.join(FIGDIR, 'canonical_ant_mf_comparison.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")

    # Print error summary
    err_naive = 100 * np.abs(lat - mf_naive) / mf_naive
    err_local = 100 * np.abs(lat - mf_local) / mf_local
    print(f"\nMean |err|: naive={err_naive.mean():.1f}%, "
          f"local={err_local.mean():.1f}%")
    return out


# ─────────────────────────────────────────────────────────────────
# Figure C: λ sweep — order parameter / field max vs λ (evaporation)
# ─────────────────────────────────────────────────────────────────
def make_lambda_sweep_figure():
    """Show how pheromone organisation changes with evaporation λ."""
    lams = np.array([0.02, 0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.60, 0.90])
    n_fixed = 0.15

    print("\nRunning λ sweep...")
    f_R, phi_m, phi_max, concentration, phi_cond_S = [], [], [], [], []
    for lam in lams:
        print(f"  λ={lam:.2f}...", flush=True)
        rs = [run_lattice(n_fixed, lam=lam, seed=42 + s * 13)
              for s in range(3)]
        fR = np.mean([r['f_A_mean'] for r in rs])
        pm = np.mean([r['phi_mean'] for r in rs])
        # Need phi.max from a snapshot run
        snap = snapshot_run(n_fixed, lam=lam, seed=42, n_steps=2000)
        px = snap['phi'].max()
        concentration.append(px / pm if pm > 0.01 else 1.0)
        f_R.append(fR)
        phi_m.append(pm)
        phi_max.append(px)
        phi_cond_S.append(np.mean([r['phi_B'] for r in rs]))

    f_R = np.array(f_R); phi_m = np.array(phi_m); phi_max = np.array(phi_max)
    concentration = np.array(concentration); phi_cond_S = np.array(phi_cond_S)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8),
                              constrained_layout=True)

    ax = axes[0]
    ax.semilogx(lams, f_R, 'ko-', lw=1.8, ms=6)
    ax.set_xlabel(r'Evaporation $\lambda$ (MSR mass)')
    ax.set_ylabel(r'Recruiter fraction $f_R$')
    ax.set_title(r'State balance vs $\lambda$')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.loglog(lams, phi_m, 'o-', color='#1f77b4',
              label=r'$\langle\phi\rangle$', lw=1.8, ms=6)
    ax.loglog(lams, phi_max, 's-', color='#d62728',
              label=r'$\phi_{\max}$', lw=1.8, ms=6)
    ax.loglog(lams, phi_cond_S, '^-', color='#ff7f0e',
              label=r'$\langle\phi|S\rangle$', lw=1.8, ms=6)
    # Reference: 1/λ scaling
    ax.loglog(lams, 1.0 / lams * phi_m[0] * lams[0], 'k:', alpha=0.5,
              label=r'$\sim 1/\lambda$')
    ax.set_xlabel(r'Evaporation $\lambda$')
    ax.set_ylabel(r'Field magnitude')
    ax.set_title(r'Field vs $\lambda$: mass-like scaling')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    ax.semilogx(lams, concentration, 'mo-', lw=1.8, ms=6)
    ax.set_xlabel(r'Evaporation $\lambda$')
    ax.set_ylabel(r'Concentration $\phi_{\max}/\langle\phi\rangle$')
    ax.set_title(r'Spatial organisation vs $\lambda$')
    ax.grid(alpha=0.3, which='both')
    ax.axhline(y=1, color='grey', ls='--', alpha=0.5)

    out = os.path.join(FIGDIR, 'canonical_ant_lambda_sweep.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")

    # Dump table
    print("\nλ-sweep table:")
    print(f"{'λ':>6s} {'f_R':>6s} {'<φ>':>7s} {'<φ|S>':>7s} "
          f"{'φ_max':>7s} {'conc':>6s}")
    for i, L in enumerate(lams):
        print(f"{L:>6.2f} {f_R[i]:>6.3f} {phi_m[i]:>7.3f} "
              f"{phi_cond_S[i]:>7.3f} {phi_max[i]:>7.2f} "
              f"{concentration[i]:>6.2f}")
    return out


if __name__ == '__main__':
    print("=" * 72)
    print("CFAC ANT PAPER FIGURES")
    print("=" * 72)
    print(f"\nOutput dir: {FIGDIR}\n")

    make_snapshots_figure()
    make_mf_comparison_figure()
    make_lambda_sweep_figure()

    print("\nDONE.")
