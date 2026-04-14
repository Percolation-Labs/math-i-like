"""
Experiment 4: bridge function for the cubic vertex (triangle bubble).

The CFAC scalar-bubble bridge is mass-independent at one loop in d = 4 - eps:
    B_2(m1, m2) = int d^d k / [(k^2 + m1^2)(k^2 + m2^2)]
                = Gamma(2 - d/2) (4pi)^{-d/2} int_0^1 [x m1^2 + (1-x) m2^2]^{d/2-2} dx
The 1/eps pole has residue = 2 (4pi)^{-2}, independent of (m1, m2).
This is the bridge_scalar() = 1 result from the 0-d skeleton.

For the cubic vertex (a triangle diagram at zero external momentum):
    B_3(m1, m2, m3) = int d^d k / [(k^2 + m1^2)(k^2 + m2^2)(k^2 + m3^2)]
The cubic-theory upper critical dimension is d_c = 6, so we work at d = 6 - eps.
Mass-independence of the 1/eps pole at the cube-root cusp would be the
analogue of bridge_scalar() = 1 for the rank-3 cusp.

Test: compute eps * B_3(m1, m2, m3) at small eps for a range of (m1, m2, m3).
If mass-independent (relative deviation < 1e-3), the cubic-vertex bridge
collapses to a constant — we have a "bridge_cubic = 1" result analogous to
the scalar case, and the anomalous-dimension calculation at the C_3 cusp
proceeds with one closed-form bridge function.

If NOT mass-independent, we need the explicit f_3(r1, r2, r3) bridge function
for the rank-3 vertex — a two-variable analogue of f(r) = ln(r)/(r-1).
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import gamma


def B3(m1: float, m2: float, m3: float, eps: float) -> float:
    """Triangle bubble at zero external momentum, d = 6 - eps."""
    d = 6 - eps
    # Two Feynman parameters: x, y, with z = 1 - x - y
    # Integrand: 2 * (x m1^2 + y m2^2 + (1-x-y) m3^2)^{d/2 - 3}
    def integrand(y, x):
        z = 1 - x - y
        if z <= 0 or x < 0 or y < 0:
            return 0.0
        m_eff_sq = x * m1**2 + y * m2**2 + z * m3**2
        if m_eff_sq <= 0:
            return 0.0
        return 2 * m_eff_sq**(d/2 - 3)
    integral, _ = dblquad(integrand, 0, 1, 0, lambda x: 1 - x, epsabs=1e-10, epsrel=1e-8)
    prefactor = gamma(3 - d/2) * (4 * np.pi)**(-d/2)
    return prefactor * integral


def main():
    eps = 1e-4

    # Predicted residue if mass-independent: 2 (4pi)^{-3}.
    # Derivation: at d = 6 - eps, Gamma(3 - d/2) = Gamma(eps/2) ~ 2/eps;
    # Feynman parametrisation has prefactor 2 from Gamma(3) = 2!;
    # simplex integral int_0^1 dx int_0^{1-x} dy = 1/2.
    # So eps * B_3 -> eps * (2/eps) * (4pi)^{-3} * 2 * 1/2 = 2 (4pi)^{-3}.
    predicted = 2 * (4 * np.pi)**(-3)

    print(f'Predicted (mass-independent) pole residue: {predicted:.6e}')
    print(f'(formula: 2 * (4 pi)^{{-3}}, the rank-3 analogue of the scalar 2*(4pi)^{{-2}})')
    print()
    print(f'{"m1":>8} {"m2":>8} {"m3":>8} {"eps*B3":>16} {"ratio to pred":>14} {"rel.dev":>12}')

    rel_devs = []
    cases = [
        (1, 1, 1),
        (1, 2, 3),
        (1, 1, 5),
        (0.5, 2, 4),
        (0.1, 1, 10),
        (1, 1, 100),
        (0.01, 1, 100),
        (0.5, 0.5, 5),
        (3, 0.3, 3),
    ]
    for m1, m2, m3 in cases:
        v = eps * B3(m1, m2, m3, eps)
        rel = (v - predicted) / predicted
        rel_devs.append(abs(rel))
        print(f'{m1:>8g} {m2:>8g} {m3:>8g} {v:>16.6e} {v/predicted:>14.6f} {rel:>+12.2e}')

    max_rel = max(rel_devs)
    print()
    print(f'Max relative deviation across all mass combinations: {max_rel:.2e}')
    print()
    if max_rel < 1e-3:
        print('=> CUBIC VERTEX BRIDGE IS MASS-INDEPENDENT to 1e-3')
        print('   bridge_cubic() = 1 is the rank-3 analogue of bridge_scalar() = 1')
        print('   The C_3 cusp anomalous dimension gamma_3(d) at one loop is therefore')
        print('   a simple closed-form rational of the simplex measure.')
    else:
        print('=> CUBIC VERTEX BRIDGE IS NOT mass-independent.  Pole residue depends on')
        print('   the mass ratios; we need an explicit two-variable bridge function')
        print('   f_3(r12, r13) analogous to f(r) = ln(r)/(r-1) for the rank-2 case.')

    # If mass-independent, predict gamma_3(d):
    # gamma_3(eps) = (counting factor) * (bridge=1) * eps + O(eps^2)
    # The counting factor for the cube-root cusp at the canonical CRN is...
    # phi(G) = (1+G)^3 + beta G; the cubic vertex has counting weight = 1 (single coupling)
    # vs the scalar bubble (n=0 SAW had counting (n+2)/(4(n+8)) = 2/32 = 1/16).
    # By analogy: gamma_3(d) ~ k * eps * bridge / (loop factor).  Need explicit calc.
    print()
    print('Predicted spatial exponent (if mass-independent):')
    print(f'  tau_3(d) = 4/3 + gamma_3(d), with gamma_3(d) computable from one extra')
    print(f'  loop integral.  d_c = 6 for cubic interactions => eps = 6 - d.')
    print(f'  At d = 1 (SAW-like spatial extension): eps = 5, OUTSIDE perturbative range.')
    print(f'  At d = 4 (DP universal): eps = 2, perturbative window.')
    print(f'  At d = 6 (mean-field): eps = 0, exact mean-field.')


if __name__ == '__main__':
    main()
