"""
Tunable Puiseux exponent from cooperativity — the NON-1/2 CFAC prediction.

For a canonical ant with cooperativity degree n:
  S -> R at rate k_+ (1 + χ φ^n)    (sum of monomial up to φ^n)
  R -> S at rate k_-
  R -> R + φ at rate δ
  φ decay at λ

The MF self-consistent equation is:
  f_R = k_+(1 + χ φ^n) / [k_+(1 + χ φ^n) + k_-]
  φ = δ ρ f_R / λ
  =>  α g^n f_R^{n+1} - α g^n f_R^n + (1+α) f_R - α = 0
  with g = χ^{1/n} (δ ρ / λ),  α = k_+/k_-.

This is degree (n+1) in f_R.  The CFAC AC layer reads off the
Puiseux exponent at the branch point DIRECTLY from the
polynomial degree:

  Tangent bifurcation (saddle-node): β = 1/2 (generic, any n >= 2)
  Pitchfork:                         β = 1/2 (generic)
  Cusp (tricritical):                β = 1/3 (triple-root merger)
  Degenerate higher-order:           β = 1/k (k-fold merger)

For a CLEAN non-1/2 exponent, we need to tune to the CUSP where
two saddle-nodes coalesce.  This occurs when the first TWO
derivatives vanish simultaneously.

Parametrise: include both LINEAR and cooperative couplings:
  S -> R at rate k_+(1 + χ_1 φ + χ_2 φ²)

MF (dividing by k_-): α(1 + g_1 f + g_2 f²) = f(1+α+α g_1 f + α g_2 f²)
  α g_2 f^3 - α g_2 f^2 + α g_1 f^2 - α g_1 f + (1+α)f - α = 0   [wait, redo]

  α(1 + g_1 f + g_2 f²) = f + α f(1 + g_1 f + g_2 f²)
  α(1+g_1 f+g_2 f²)(1 - f) = f
  α(1-f) + α g_1 f(1-f) + α g_2 f²(1-f) = f
  α - α f + α g_1 f - α g_1 f² + α g_2 f² - α g_2 f³ = f
  -α g_2 f^3 + (α g_2 - α g_1) f^2 + (α g_1 - α - 1) f + α = 0

  Or: α g_2 f^3 - α(g_2 - g_1) f^2 - (α g_1 - α - 1) f - α = 0

At a CUSP: both F(f)=0 AND F'(f)=0 AND F''(f)=0 simultaneously
(triple real root).  The CFAC Puiseux exponent is then 1/3.

We will NUMERICALLY:
  (1) Scan (g_1, g_2) space to find cusp point (g_1^*, g_2^*)
  (2) Approach cusp along a path ε = (g_2 - g_2^*) with g_1 = g_1^*
  (3) Measure the scaling of the triple-root as it splits
  (4) Verify exponent = 1/3 (CFAC Puiseux)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def mf_poly(alpha, g1, g2):
    """Return coefficients [a3, a2, a1, a0] for a3 f^3 + a2 f^2 + a1 f + a0 = 0."""
    a3 = alpha * g2
    a2 = -alpha * (g2 - g1)
    a1 = -(alpha * g1 - alpha - 1)
    a0 = -alpha
    return [a3, a2, a1, a0]


def real_roots_in_01(coeffs):
    """Return sorted list of real roots in [−0.01, 1.01]."""
    r = np.roots(coeffs)
    real = [x.real for x in r
            if abs(x.imag) < 1e-8 and -0.01 <= x.real <= 1.01]
    return sorted(real)


def find_cusp(alpha=0.2):
    """The cusp is where F = F' = F'' = 0, i.e. the discriminant
    AND its derivative both vanish.  Find by 2D search."""
    # Cubic discriminant: Δ = 18 a3 a2 a1 a0 - 4 a2^3 a0 + a2^2 a1^2
    #                      - 4 a3 a1^3 - 27 a3^2 a0^2
    # Δ = 0 separates 1-real from 3-real region.
    # Cusp is where Δ = 0 AND ∂Δ/∂g1 (along the boundary) = 0.
    # Easier: cusp has triple root; triple root of cubic a3 f^3+...
    # means f_triple = -a2/(3 a3) and plugging into the cubic
    # gives 0.  So: -a2/(3 a3) is the triple root.

    # Scan (g1, g2) and find where the cubic has a triple real root
    # in (0,1).

    best = None
    best_dev = np.inf
    for g1 in np.linspace(-10, 10, 200):
        for g2 in np.linspace(0.5, 100, 200):
            a3, a2, a1, a0 = mf_poly(alpha, g1, g2)
            if abs(a3) < 1e-12:
                continue
            f_triple = -a2 / (3 * a3)
            if not (0.01 < f_triple < 0.99):
                continue
            # Check cubic value at triple root
            val = a3 * f_triple**3 + a2 * f_triple**2 + a1 * f_triple + a0
            # Check derivative at triple root (should be 0 too)
            deriv = 3 * a3 * f_triple**2 + 2 * a2 * f_triple + a1
            dev = val**2 + deriv**2
            if dev < best_dev:
                best_dev = dev
                best = (g1, g2, f_triple, val, deriv)

    if best is None:
        return None
    return best


def test_puiseux_at_cusp(alpha=0.2):
    """Near the cusp, scale each branch vs the tuning parameter."""
    cusp = find_cusp(alpha)
    if cusp is None:
        print("No cusp found.")
        return
    g1_c, g2_c, f_c, _, _ = cusp
    print(f"  Cusp found at (g1, g2) = ({g1_c:.4f}, {g2_c:.4f}),"
          f"  triple root f* = {f_c:.4f}")

    # Refine cusp by iterative local optimization
    best = (g1_c, g2_c, f_c)
    dev_best = np.inf
    # Multi-scale refinement
    for scale in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        g1r, g2r, fr = best
        for dg1 in np.linspace(-scale, scale, 21):
            for dg2 in np.linspace(-scale*10, scale*10, 21):
                g1n, g2n = g1r + dg1, g2r + dg2
                a3, a2, a1, a0 = mf_poly(alpha, g1n, g2n)
                if abs(a3) < 1e-12:
                    continue
                f_try = -a2 / (3 * a3)
                if not (0 < f_try < 1):
                    continue
                val = a3 * f_try**3 + a2 * f_try**2 + a1 * f_try + a0
                der = 3*a3*f_try**2 + 2*a2*f_try + a1
                dev = val**2 + der**2
                if dev < dev_best:
                    dev_best = dev
                    best = (g1n, g2n, f_try)
    g1_c, g2_c, f_c = best
    print(f"  Refined cusp: ({g1_c:.6f}, {g2_c:.6f}), f* = {f_c:.6f}")

    # Move along g_2 direction (we know 3-root is for g_2 slightly
    # SMALLER than cusp value at fixed g_1).
    # Use fine scan with geometric eps.
    eps_vals = np.geomspace(1e-7, 1e-1, 80)
    spreads = []
    chosen_sign = None
    for eps in eps_vals:
        # Try both directions
        for sign in [-1, +1]:
            g2n = g2_c + sign * eps
            roots = real_roots_in_01(mf_poly(alpha, g1_c, g2n))
            if len(roots) >= 3:
                if chosen_sign is None:
                    chosen_sign = sign
                if sign == chosen_sign:
                    spread = roots[-1] - roots[0]
                    spreads.append(spread)
                    break
        else:
            spreads.append(np.nan)

    spreads = np.array(spreads)

    spreads = np.array(spreads)
    m = ~np.isnan(spreads) & (spreads > 1e-6)
    print(f"  3-root points found: {m.sum()}/{len(eps_vals)}  "
          f"(direction sign = {chosen_sign})")
    if m.sum() < 5:
        print("  Not enough points to fit.")
        return None
    slope, intercept = np.polyfit(np.log(eps_vals[m]),
                                   np.log(spreads[m]), 1)
    print(f"\n  Puiseux exponent fitted: β = {slope:.4f}")
    print(f"  CFAC AC-layer prediction at cusp: β = 1/3 = {1/3:.4f}")
    print(f"  Error: {100*abs(slope - 1/3)/(1/3):.2f}%")

    # Make plot
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2), constrained_layout=True)
    ax.loglog(eps_vals[m], spreads[m], 'o', color='purple', ms=6,
              mfc='none', mew=2, label='MF numerical')
    eref = eps_vals[m]
    ax.loglog(eref, np.exp(intercept) * eref**slope, '-', color='purple',
              alpha=0.6, label=rf'fit $\beta={slope:.3f}$')
    ax.loglog(eref, np.exp(intercept) * eref**(1/3), 'k--',
              label=r'CFAC Puiseux: $\beta=1/3$')
    ax.set_xlabel(r'$\varepsilon = |g_2 - g_2^{\rm cusp}|$')
    ax.set_ylabel(r'Spread of three roots')
    ax.set_title(r'Tunable CFAC exponent: cusp gives $\beta = 1/3$ (not $1/2$)')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.3, which='both')
    out = os.path.join(figdir, 'canonical_ant_cusp.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")

    return slope, g1_c, g2_c


def simple_puiseux_at_saddle(alpha=0.2, g2=30):
    """Also verify standard saddle-node gives 1/2 (for contrast)."""
    # Pure g2, find saddle-node
    # Cubic in f: α g2 f³ - α g2 f² + (1+α) f - α = 0
    # Saddle at αg = 4 (derived earlier analytically for g1=0)
    # Actually this corresponds to α g_2 = 4, so g_2 = 20 for α=0.2
    # NO WAIT: the original "cooperative no baseline" I solved was
    # with no constant term.  Here with baseline, different.
    # Let's just numerically find saddle-node of the cubic.

    # Find g2_sn: value above which cubic has 3 real roots
    g_low, g_high = 1.0, 1000.0
    while abs(g_high - g_low) > 1e-6:
        gm = (g_low + g_high) / 2
        n_roots = len(real_roots_in_01(mf_poly(alpha, 0, gm)))
        if n_roots >= 3:
            g_high = gm
        else:
            g_low = gm
    g_sn = (g_low + g_high) / 2
    print(f"  Saddle-node at g2 = {g_sn:.4f} (for g1 = 0)")
    if g_sn >= 999:
        print("  Hmm, may not exist at this α")
        return None

    # Scale: take g2 just above g_sn and measure root spread ~ (g-g_sn)^{1/2}
    eps_vals = np.geomspace(1e-6, 1e-1, 30)
    spreads = []
    for eps in eps_vals:
        roots = real_roots_in_01(mf_poly(alpha, 0, g_sn + eps))
        if len(roots) >= 3:
            spreads.append(roots[-1] - roots[1])
        else:
            spreads.append(np.nan)
    spreads = np.array(spreads)
    m = ~np.isnan(spreads) & (spreads > 1e-4)
    if m.sum() < 5:
        return None
    slope, _ = np.polyfit(np.log(eps_vals[m]), np.log(spreads[m]), 1)
    print(f"  Puiseux at saddle-node: slope = {slope:.4f}"
          f"  (CFAC: 1/2 = 0.5)")
    return slope


if __name__ == '__main__':
    import sys

    # Stub scipy-like refine function (local bisection)
    class scipy_like_opt:
        @staticmethod
        def refine(alpha, g1, g2, f):
            best = (g1, g2, f)
            dev_best = np.inf
            for dg1 in np.linspace(-0.5, 0.5, 31):
                for dg2 in np.linspace(-5, 5, 31):
                    g1n, g2n = g1 + dg1, g2 + dg2
                    a3, a2, a1, a0 = mf_poly(alpha, g1n, g2n)
                    if abs(a3) < 1e-10:
                        continue
                    f_try = -a2 / (3*a3)
                    if not (0 < f_try < 1):
                        continue
                    val = a3 * f_try**3 + a2 * f_try**2 + a1 * f_try + a0
                    der = 3*a3*f_try**2 + 2*a2*f_try + a1
                    dev = val**2 + der**2
                    if dev < dev_best:
                        dev_best = dev
                        best = (g1n, g2n, f_try)
            return best
    sys.modules[__name__].scipy_like_opt = scipy_like_opt

    print("=" * 72)
    print("TUNABLE CFAC PUISEUX EXPONENTS")
    print("=" * 72)

    print("\n(1) Standard saddle-node (cooperative + baseline):")
    simple_puiseux_at_saddle(alpha=0.2)

    print("\n(2) Cusp / tricritical point in the (g1, g2) 2-parameter family:")
    test_puiseux_at_cusp(alpha=0.2)
