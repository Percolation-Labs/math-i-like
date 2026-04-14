"""
Saddle-node β exponent in the cooperative canonical ant
(k_+ chi phi^2 coupling, NO baseline — so the disordered f_R=0
state is a fixed point and the ordered branch emerges at a
tangent bifurcation).

MF equation (quadratic branch):
  f_R(k_+ g f_R^2 + k_-) = k_+ g f_R^2
  => f_R (1 - f_R) = 1/(alpha g)   for f_R > 0
  => f_R^2 - f_R + 1/(alpha g) = 0

Solutions:
  f_R^{(±)} = [1 ± sqrt(1 - 4/(alpha g))] / 2
  + OFF: f_R = 0 (always a root)

Bistability: (alpha g) > 4
Saddle-node: alpha g_c = 4, f_R^{(sn)} = 1/2.

Near saddle-node: (f_R^{(+)} - f_R^{(sn)}) = (1/2) sqrt(1 - 4/(alpha g))
  = (1/2) sqrt( (alpha g - 4)/(alpha g) )
  ~ (1/2) sqrt( (alpha g - alpha g_c)/(alpha g_c) )
  ~ [(alpha g - alpha g_c)/alpha g_c]^{1/2}

=> CFAC AC-layer prediction: β = 1/2 (Puiseux exponent of
the quadratic DSE).

This script verifies beta = 1/2 numerically from the MF curves
AND confirms stability of the ON branch on the lattice.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def mf_roots_nobaseline(g, alpha):
    """Roots of alpha g f^2 - alpha g f + 1 = 0 → f^2 - f + 1/(ag) = 0
    (plus f=0 always).
    """
    if alpha * g < 4:
        return [0.0]
    disc = 1 - 4 / (alpha * g)
    return [0.0, (1 - np.sqrt(disc)) / 2, (1 + np.sqrt(disc)) / 2]


def run_nobaseline(n, L=80, k_plus=0.01, k_minus=0.05, chi=1.0,
                    delta=2.0, lam=0.10, sigma=0.005,
                    n_steps=4000, burn_in=2000, seed=42,
                    init_f_R=0.7):
    """Cooperative ant: k_+ chi phi^2 coupling, NO baseline.
    Start with init_f_R fraction of agents in R (test ON branch stability).
    """
    rng = np.random.default_rng(seed)
    occ = rng.random((L, L)) < n
    is_R = rng.random((L, L)) < init_f_R
    is_R = is_R & occ
    is_S = occ & ~is_R

    # Seed the field at R-sites so we're on the ON branch to start
    phi = np.zeros((L, L), dtype=np.float64)
    phi[is_R] = delta / lam  # near the saturation value

    def laplacian(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    f_R_trace = []
    for t in range(n_steps):
        # No baseline: rate k_+ chi phi^2
        rate_SR = k_plus * chi * phi * phi
        p_SR = np.clip(rate_SR, 0.0, 1.0)
        convert_SR = is_S & (rng.random((L, L)) < p_SR)
        convert_RS = is_R & (rng.random((L, L)) < k_minus)
        new_R = (is_R & ~convert_RS) | convert_SR
        new_S = (is_S & ~convert_SR) | convert_RS
        is_R, is_S = new_R, new_S
        phi = phi + delta * is_R.astype(np.float64)
        phi = phi * (1.0 - lam)
        phi = phi + sigma * laplacian(phi)
        if t >= burn_in:
            nR, nS = int(is_R.sum()), int(is_S.sum())
            if nR + nS > 0:
                f_R_trace.append(nR / (nR + nS))

    if not f_R_trace:
        return {'n': n, 'f_R_mean': 0, 'f_R_std': 0}
    return {'n': n, 'f_R_mean': float(np.mean(f_R_trace)),
            'f_R_std': float(np.std(f_R_trace))}


def main():
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    # Parameters
    k_plus, k_minus, chi, delta, lam = 0.01, 0.05, 1.0, 2.0, 0.10
    alpha = k_plus / k_minus  # 0.2

    # g = chi * (delta n / lam)^2 = 400 n^2 (with these params)
    # Saddle-node: alpha g_c = 4 → g_c = 20 → n_c^2 = 0.05 → n_c = 0.2236
    g_c = 4.0 / alpha
    n_c = lam / delta * np.sqrt(g_c / chi)
    print(f"MF saddle-node: g_c = {g_c:.2f}, n_c = {n_c:.4f}")

    # MF curves
    n_range = np.linspace(0.01, 0.5, 300)
    f_OFF = np.zeros_like(n_range)
    f_LOW = np.full_like(n_range, np.nan)
    f_HI = np.full_like(n_range, np.nan)
    for i, n in enumerate(n_range):
        g = chi * (delta * n / lam) ** 2
        roots = mf_roots_nobaseline(g, alpha)
        f_OFF[i] = 0.0
        if len(roots) == 3:
            f_LOW[i] = roots[1]
            f_HI[i] = roots[2]

    # Fit β from the high-branch MF: (f_HI - 1/2) vs (n - n_c)/n_c
    # Restrict to near-threshold for clean Puiseux exponent
    m = ~np.isnan(f_HI) & (n_range > n_c * 1.0001) & (n_range < n_c * 1.05)
    dn = (n_range[m] - n_c) / n_c
    y = f_HI[m] - 0.5
    beta_mf, _ = np.polyfit(np.log(dn), np.log(y), 1)
    print(f"MF numerical β (upper branch, ε<5%): {beta_mf:.4f}  "
          f"(CFAC Puiseux = 0.5)")

    # Lattice sweep
    print("\nLattice sweep around n_c (starting on the ON branch)...")
    n_sweep = np.array([0.22, 0.24, 0.27, 0.30, 0.35, 0.40, 0.50])
    lattice_f_R = []
    lattice_sd = []
    for n in n_sweep:
        rs = [run_nobaseline(n, seed=42 + s * 17) for s in range(3)]
        fm = np.mean([r['f_R_mean'] for r in rs])
        fs = np.std([r['f_R_mean'] for r in rs])
        lattice_f_R.append(fm)
        lattice_sd.append(fs)
        print(f"  n = {n:.3f}:  lattice f_R = {fm:.3f} ± {fs:.3f}")
    lattice_f_R = np.array(lattice_f_R)
    lattice_sd = np.array(lattice_sd)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                              constrained_layout=True)

    # Panel 1: Bifurcation diagram (MF + lattice)
    ax = axes[0]
    ax.plot(n_range, f_OFF, 'b-', lw=2, label=r'MF: $f_R=0$ (OFF)')
    ax.plot(n_range, f_LOW, 'r--', lw=1.5,
            label=r'MF: unstable mid branch')
    ax.plot(n_range, f_HI, 'g-', lw=2, label=r'MF: $f_R^{(+)}$ (ON)')
    ax.errorbar(n_sweep, lattice_f_R, yerr=lattice_sd, fmt='ko',
                ms=7, mfc='k', capsize=4, label='lattice (ON start)')
    ax.axvline(n_c, color='grey', ls=':', alpha=0.6,
               label=rf'$n_c = {n_c:.3f}$')
    ax.set_xlabel(r'Density $n$')
    ax.set_ylabel(r'Recruiter fraction $f_R$')
    ax.set_title(r'Cooperative canonical ant: saddle-node bifurcation')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: log-log near saddle-node
    ax = axes[1]
    ax.loglog(dn, y, 'g-', lw=2,
              label=rf'MF analytic: $\beta={beta_mf:.3f}$')
    # lattice points above n_c
    m2 = n_sweep > n_c
    dn_lat = (n_sweep[m2] - n_c) / n_c
    y_lat = lattice_f_R[m2] - 0.5
    mm = y_lat > 0
    if mm.sum() >= 3:
        beta_lat = np.polyfit(np.log(dn_lat[mm]),
                              np.log(y_lat[mm]), 1)[0]
        ax.loglog(dn_lat[mm], y_lat[mm], 'ko', ms=8, mfc='none',
                  mew=2, label=rf'lattice: $\beta={beta_lat:.3f}$')
    else:
        beta_lat = np.nan
    # Reference β=1/2
    ref_x = np.array([dn.min(), dn.max()])
    ax.loglog(ref_x, 0.5 * ref_x ** 0.5, 'k--', alpha=0.5,
              label=r'CFAC tree: $\beta = 1/2$')
    ax.set_xlabel(r'$(n - n_c)/n_c$')
    ax.set_ylabel(r'$f_R^{(+)} - f_R^{\rm (sn)}$')
    ax.set_title(r'Order parameter exponent $\beta$ at saddle-node')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.grid(alpha=0.3, which='both')

    out = os.path.join(figdir, 'canonical_ant_saddle_node.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")

    print(f"\n{'='*60}")
    print(f"COOPERATIVE CANONICAL ANT (k_+ χ φ² coupling)")
    print(f"  Saddle-node: n_c = {n_c:.4f}, g_c = {g_c:.2f}")
    print(f"  CFAC AC-layer prediction: β = 1/2 (Puiseux, quadratic DSE)")
    print(f"  MF numerical fit β: {beta_mf:.4f}")
    if not np.isnan(beta_lat):
        print(f"  Lattice β (ON branch): {beta_lat:.4f}")
    print(f"{'='*60}")

    return n_c, beta_mf


if __name__ == '__main__':
    main()
