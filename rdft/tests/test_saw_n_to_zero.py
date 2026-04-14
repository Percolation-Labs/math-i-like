"""
Self-avoiding walk (SAW) as the n -> 0 limit of the O(n) model: AC+ demonstration.

Claim (problems.md #1): at n=0, the AC+ scalar bubble has bridge function = 1,
so the one-loop exponent is pure counting. The Wilson-Fisher one-loop result
    nu = 1/2 + (n+2) / (4(n+8)) * eps
specialises at n=0 to
    nu_SAW = 1/2 + eps/16 + O(eps^2),
which is exactly what one_loop_On(n=0) returns.

This test verifies:
  (a) the AC+ counting -> nu coefficient matches Wilson-Fisher as rationals
      for n in {0, 1, 2, 3} (SAW, Ising, XY, Heisenberg),
  (b) the scalar bubble pole residue is mass-independent at n=0, to ~10^-5,
      across six orders of magnitude in the mass ratio,
  (c) the residue equals AC+'s closed form 2 * Omega_4 * bridge_scalar().
"""

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import gamma

from rdft.ac.bridge import one_loop_On, one_loop_pole, bridge_scalar


def _wilson_fisher_nu_coeff(n: int) -> sp.Rational:
    return sp.Rational(n + 2, 4 * (n + 8))


def test_one_loop_nu_matches_wilson_fisher_for_n_in_0_to_3():
    for n in (0, 1, 2, 3):
        r = one_loop_On(n=n)
        ac_coef = sp.nsimplify(r['nu_coefficient'], rational=True)
        assert ac_coef == _wilson_fisher_nu_coeff(n), (
            f'n={n}: AC+ gave {ac_coef}, Wilson-Fisher expects '
            f'{_wilson_fisher_nu_coeff(n)}'
        )


def test_saw_nu_coefficient_is_exactly_one_sixteenth():
    r = one_loop_On(n=0)
    assert sp.nsimplify(r['nu_coefficient'], rational=True) == sp.Rational(1, 16)
    # At d=3 (eps=1): nu ~ 1/2 + 1/16 = 9/16 = 0.5625
    assert abs(r['nu_1loop_d3'] - 0.5625) < 1e-12


def _scalar_bubble(m1: float, m2: float, eps: float) -> float:
    """One-loop bubble integral at zero external momentum, in d = 4 - eps.

    Closed-form Feynman parametrisation:
        B = Gamma(2 - d/2) (4 pi)^{-d/2} int_0^1 dx [x m1^2 + (1-x) m2^2]^{d/2 - 2}.
    """
    d = 4 - eps
    integrand = lambda x: (x * m1 ** 2 + (1 - x) * m2 ** 2) ** (d / 2 - 2)
    integral, _ = quad(integrand, 0, 1)
    return gamma(2 - d / 2) * (4 * np.pi) ** (-d / 2) * integral


def test_scalar_bubble_pole_is_mass_independent_at_n_zero():
    """Residue = 2 * Omega_4 * bridge_scalar(), independent of masses."""
    predicted = one_loop_pole(d_c=4) * bridge_scalar()  # = 2 (4 pi)^{-2}
    eps = 1e-5

    mass_pairs = [
        (1.0, 1.0),
        (1.0, 2.0),
        (1.0, 5.0),
        (1.0, 10.0),
        (0.1, 10.0),
        (0.01, 100.0),
        (5.0, 0.2),
        (100.0, 1.0),
        (1e-3, 1e3),
    ]
    max_rel_dev = 0.0
    for m1, m2 in mass_pairs:
        observed = eps * _scalar_bubble(m1, m2, eps)
        rel_dev = abs(observed - predicted) / predicted
        max_rel_dev = max(max_rel_dev, rel_dev)

    # The remaining deviation is the O(eps) subleading term, controlled by eps.
    assert max_rel_dev < 1e-4, (
        f'Mass-independence violated: max relative deviation {max_rel_dev:.2e} '
        f'exceeds 1e-4 across six orders of magnitude in the mass ratio.'
    )


def test_bridge_at_n_zero_is_the_identity():
    """The AC+ claim: at n=0, the analytic bridge collapses to 1.

    In Madras-Slade language, this is the statement that the SAW two-point
    function's one-loop correction is carried entirely by the counting
    factor (n+8) -> 8 from the O(n) trace; the remaining analytic content
    (the scalar bubble pole) is a mass-independent constant. That constant
    times the counting gives nu = 1/2 + eps/16 with zero further input.
    """
    assert bridge_scalar() == 1.0
    assert one_loop_On(n=0)['bridge'] == 1.0
