"""
Why is the cusp messy?  Diagnostic.

Claim: the cusp appears at g_1 < 0 (inhibitory linear coupling).
For physically activating ant couplings (g_1 > 0, g_2 > 0),
there is NO cusp — just a generic saddle-node.

We scan the (g_1, g_2) plane and mark 1-root vs 3-root regions,
then find the discriminant curve and any cusps.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def mf_poly(alpha, g1, g2):
    return [alpha * g2,
            -alpha * (g2 - g1),
            -(alpha * g1 - alpha - 1),
            -alpha]


def discriminant(alpha, g1, g2):
    a3, a2, a1, a0 = mf_poly(alpha, g1, g2)
    return (18 * a3 * a2 * a1 * a0 - 4 * a2**3 * a0 + a2**2 * a1**2
            - 4 * a3 * a1**3 - 27 * a3**2 * a0**2)


def count_roots_in_01(alpha, g1, g2):
    coeffs = mf_poly(alpha, g1, g2)
    rs = np.roots(coeffs)
    return sum(1 for r in rs if abs(r.imag) < 1e-8 and -0.01 < r.real < 1.01)


def main():
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    alpha = 0.2

    # Full scan of (g_1, g_2)
    g1_range = np.linspace(-8, 8, 200)
    g2_range = np.linspace(1, 60, 200)
    G1, G2 = np.meshgrid(g1_range, g2_range)
    n_roots = np.zeros_like(G1)
    disc = np.zeros_like(G1)

    for i in range(len(g2_range)):
        for j in range(len(g1_range)):
            n_roots[i, j] = count_roots_in_01(alpha, G1[i, j], G2[i, j])
            disc[i, j] = discriminant(alpha, G1[i, j], G2[i, j])

    # Cusp candidates: points on disc = 0 where the gradient is also 0
    # (i.e., stationary points of the disc=0 curve) OR where the
    # inflection condition {val=0, deriv=0} holds.
    inflection_val = np.zeros_like(G1)
    for i in range(len(g2_range)):
        for j in range(len(g1_range)):
            g1, g2 = G1[i, j], G2[i, j]
            a3, a2, a1, a0 = mf_poly(alpha, g1, g2)
            if abs(a3) < 1e-10:
                inflection_val[i, j] = np.nan
                continue
            f_inflect = -a2 / (3 * a3)
            if not (0 < f_inflect < 1):
                inflection_val[i, j] = np.nan
                continue
            val = a3 * f_inflect**3 + a2 * f_inflect**2 + a1 * f_inflect + a0
            der = 3 * a3 * f_inflect**2 + 2 * a2 * f_inflect + a1
            inflection_val[i, j] = val**2 + der**2

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax1 = axes[0]
    ax1.pcolormesh(G1, G2, n_roots, shading='auto', cmap='viridis',
                    vmin=0, vmax=3)
    ax1.set_xlabel(r'$g_1$ (linear cooperativity)')
    ax1.set_ylabel(r'$g_2$ (quadratic cooperativity)')
    ax1.set_title(r'Number of real roots in $[0,1]$')
    ax1.contour(G1, G2, n_roots, levels=[1.5, 2.5], colors='white', linewidths=1)
    ax1.axvline(0, color='red', lw=1, alpha=0.5,
                label='physically activating: $g_1>0$')
    ax1.legend(loc='upper right')

    ax2 = axes[1]
    # Plot inflection-vanishing (cusp candidates) as dark spots
    logv = np.log10(inflection_val + 1e-15)
    ax2.pcolormesh(G1, G2, logv, shading='auto', cmap='plasma_r',
                    vmin=-8, vmax=0)
    ax2.set_xlabel(r'$g_1$')
    ax2.set_ylabel(r'$g_2$')
    ax2.set_title(r'$\log_{10}(F(f_*)^2 + F\,''(f_*)^2)$ — cusp candidates (darker = closer to cusp)')
    ax2.contour(G1, G2, n_roots, levels=[1.5, 2.5], colors='white',
                 linewidths=1, alpha=0.7)
    ax2.axvline(0, color='red', lw=1, alpha=0.5)
    # Identify the cusp = minimum of inflection_val over valid region
    mask = ~np.isnan(inflection_val)
    if mask.any():
        idx = np.unravel_index(np.nanargmin(inflection_val), inflection_val.shape)
        ax2.plot(G1[idx], G2[idx], 'r*', ms=16, label=f'cusp at ({G1[idx]:.2f}, {G2[idx]:.2f})')
        ax2.legend(loc='upper right')
        print(f"Cusp at (g1, g2) = ({G1[idx]:.3f}, {G2[idx]:.3f})")
        print(f"  Inside physical region (g1, g2 > 0)?  "
              f"{G1[idx] > 0 and G2[idx] > 0}")

    out = os.path.join(figdir, 'canonical_ant_cusp_diagram.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")

    # Now: verify Puiseux 1/3 at THIS cusp by crossing it carefully
    g1_c, g2_c = G1[idx], G2[idx]
    print(f"\nVerifying Puiseux exponent at cusp ({g1_c:.3f}, {g2_c:.3f})...")
    # Scan points crossing discriminant boundary
    # Direction: move in g2 with g1 fixed at cusp value
    eps_vals = np.geomspace(1e-6, 1e-1, 100)
    max_spread = []
    for eps in eps_vals:
        best_spread = np.nan
        for s in [+1, -1]:
            # Also try a couple of g1 offsets
            for dg1 in [0, 0.1 * eps, -0.1 * eps]:
                g1n = g1_c + dg1
                g2n = g2_c + s * eps
                coeffs = mf_poly(alpha, g1n, g2n)
                rs = np.roots(coeffs)
                real = sorted([r.real for r in rs if abs(r.imag) < 1e-9
                                and -0.01 < r.real < 1.01])
                if len(real) >= 3:
                    sp = real[-1] - real[0]
                    if np.isnan(best_spread) or sp > best_spread:
                        best_spread = sp
        max_spread.append(best_spread)
    max_spread = np.array(max_spread)
    m = ~np.isnan(max_spread) & (max_spread > 1e-6)
    print(f"  3-root hits: {m.sum()}/{len(eps_vals)}")
    if m.sum() >= 5:
        slope = np.polyfit(np.log(eps_vals[m]), np.log(max_spread[m]), 1)[0]
        print(f"  Puiseux β fitted: {slope:.4f}   CFAC prediction: 1/3 = {1/3:.4f}")


if __name__ == '__main__':
    main()
