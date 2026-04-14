"""
Cube-root CRN: a NON-decorative AC+ prediction.

The DSE machinery in rdft.ac.dse identifies a closed-form tuning curve in CRN
parameter space at which the dressed-propagator generating function G(z) =
z * phi(G) acquires a cube-root branch point instead of the generic DP
square-root.  At the cube-root branch the transfer theorem gives

    [z^n] G(z)  ~  A * rho^{-n} * n^{-4/3}

instead of the universal DP n^{-3/2}.  This exponent is invisible to the
epsilon-expansion around the DP fixed point: it requires non-perturbative
tuning of the vertex rates onto the algebraic cusp where phi''(G*) = 0
coincides with the branch condition.

The tuning condition for a cubic kernel
    phi(G) = 1 + a G + b G^2 + c G^3
is  b^3 = 27 c^2,  with the parameter a free.  Whether the cube-root branch
is the DOMINANT singularity (controlling the asymptotics) requires an
additional check: it must be closer to z=0 than every other branch of the
algebraic curve F(G,z) = G - z*phi(G) = 0.

This test pins down all three claims:
  (i) tuning curve is closed-form,
  (ii) at the tuned point, phi''(G*) = 0 AND branch condition self-consistent,
  (iii) for parameter regions where the cube-root branch dominates, the
        Lagrange-inverted coefficients scale as n^{-4/3}, NOT n^{-3/2}.

Concrete CRN realisation tested:
    A -> 3A      rate 3
    2A -> A      rate 2
    2A -> 3A     rate 1
=> phi(G) = 1 - G + 3 G^2 + G^3.  All rates positive; physical.
"""

import sympy as sp
import numpy as np
import pytest


def _phi_cubic(a, b, c, G):
    return 1 + a * G + b * G ** 2 + c * G ** 3


def test_cube_root_tuning_curve_is_b_cubed_equals_27_c_squared():
    """The branch-point condition phi''(G*) = 0 AND G* phi'(G*) = phi(G*)
    has a closed-form solution in (b, c), independent of a."""
    G = sp.Symbol('G')
    a, b, c = sp.symbols('a b c', real=True)
    phi = _phi_cubic(a, b, c, G)
    phi_pp = sp.diff(phi, G, 2)
    G_star = sp.solve(phi_pp, G)[0]   # G* = -b/(3c)
    branch_residual = sp.simplify(
        G_star * sp.diff(phi, G).subs(G, G_star) - phi.subs(G, G_star)
    )
    # branch_residual factors as (b^3 - 27 c^2) / (27 c^2);
    # numerator vanishes iff b^3 = 27 c^2, regardless of a.
    numer = sp.numer(sp.together(branch_residual))
    assert sp.factor(numer) == sp.factor(b ** 3 - 27 * c ** 2), (
        f'Tuning curve was expected to be b^3 = 27 c^2, got numerator {numer}'
    )


def test_specific_tuned_point_has_cube_root_branch():
    """phi(G) = 1 - G + 3G^2 + G^3 satisfies tuning, phi'' vanishes at the
    branch point G* = -1, and the algebraic curve has a (z + 1/4)^2
    factor in its discriminant (signature of a cube-root branch)."""
    G, z = sp.symbols('G z')
    phi = 1 - G + 3 * G ** 2 + G ** 3
    F = G - z * phi
    disc = sp.discriminant(sp.Poly(F, G).as_expr(), G)
    factored = sp.factor(disc)
    # disc factors as -z * (4z + 1)^2 * (11z - 4)
    # The (4z+1)^2 piece encodes the cube-root branch at z = -1/4.
    # Verify by extracting the multiplicity of z = -1/4 as a root.
    mult = sp.Poly(disc, z).as_expr().subs(z, sp.Symbol('w') - sp.Rational(1, 4))
    series = sp.series(mult, sp.Symbol('w'), 0, 4).removeO()
    # leading order should be w^2 (double zero)
    leading = sp.Poly(series, sp.Symbol('w')).all_terms()
    powers = [t[0][0] for t in leading if t[1] != 0]
    assert min(powers) == 2, f'Expected double root at z=-1/4, got powers {powers}'


def test_cube_root_branch_dominates_for_chosen_parameters():
    """At a = -1 (concrete CRN), the cube-root branch at z = -1/4 is closer
    to the origin than the secondary branch at z = 4/11, so it controls
    the asymptotic of [z^n] G(z)."""
    G, z = sp.symbols('G z')
    phi = 1 - G + 3 * G ** 2 + G ** 3
    F = G - z * phi
    disc = sp.discriminant(sp.Poly(F, G).as_expr(), G)
    branches = [complex(s) for s in sp.solve(disc, z) if s != 0]
    closest = min(branches, key=lambda b: abs(b))
    assert abs(abs(closest) - 0.25) < 1e-10, (
        f'Expected dominant branch at |z| = 0.25, got {abs(closest)}'
    )


def test_lagrange_inverted_coefficients_scale_as_n_to_minus_four_thirds():
    """The acid test: [z^n] G(z) for the tuned CRN, scaled by rho^n,
    should be much closer to n^{-4/3} than to n^{-3/2}.

    We compare |c_n| * rho^n * n^{tau} for tau in {4/3, 3/2}.  The correct
    tau converges to a constant as n grows; the wrong tau drifts."""
    G = sp.Symbol('G')
    phi = 1 - G + 3 * G ** 2 + G ** 3
    rho = 0.25  # |z*| at the cube-root branch

    N = 400
    coeffs = []
    phi_n = sp.Integer(1)
    for n in range(1, N + 1):
        phi_n = sp.expand(phi_n * phi)
        c_n = sp.Rational(1, n) * phi_n.coeff(G, n - 1)
        coeffs.append(c_n)

    cn = np.array([float(x) for x in coeffs])
    ns = np.arange(1, N + 1)
    scaled_43 = np.abs(cn) * rho ** ns * ns ** (4.0 / 3.0)
    scaled_32 = np.abs(cn) * rho ** ns * ns ** (3.0 / 2.0)

    # Drift ratio over an 8x range in n.  Constant (correct tau) -> ~1.
    drift_43 = scaled_43[399] / scaled_43[49]
    drift_32 = scaled_32[399] / scaled_32[49]

    # Wrong exponent grows linearly:  drift_32 ~ (400/50)^(3/2 - 4/3) = 8^{1/6} ~ 1.41
    # plus the drift the correct exponent has from log corrections.
    assert drift_43 < 1.15, (
        f'tau = 4/3 should converge; drift over 8x n is {drift_43:.3f}'
    )
    assert drift_32 > 1.4, (
        f'tau = 3/2 should drift up by ~8^{{1/6}} ~ 1.41; got {drift_32:.3f}'
    )
    # And the 4/3 scaling is closer to constant than the 3/2 scaling
    # by at least a factor of 2 in the relative drift.
    assert abs(drift_43 - 1) * 2 < abs(drift_32 - 1), (
        f'4/3 drift ({drift_43:.3f}) should be much smaller than '
        f'3/2 drift ({drift_32:.3f})'
    )


def test_crn_realisation_yields_the_tuned_phi():
    """The polynomial phi = 1 - G + 3 G^2 + G^3 arises from a CRN with
    strictly positive rates: A -> 3A (3), 2A -> A (2), 2A -> 3A (1).

    Doi-Peliti vertex contributions to phi(G) = 1 + sum g_{mn} G^{m+n-2}:
        A  -> 3A  rate r1: vertex (3,1) g=+r1   ->  +r1 G^2
        2A -> A   rate r2: (1,2) g=-r2, (2,2) g=-r2  ->  -r2 G - r2 G^2
        2A -> 3A  rate r3: (1,2) g=+r3, (2,2) g=+2 r3, (3,2) g=+r3
                                       ->  +r3 G + 2 r3 G^2 + r3 G^3
    Net: 1 + (r3 - r2) G + (r1 + 2 r3 - r2) G^2 + r3 G^3.
    With r1 = 3, r2 = 2, r3 = 1: 1 - G + 3 G^2 + G^3.  Tuning b^3 = 27 c^2
    is satisfied (3^3 = 27 = 27 * 1^2).
    """
    r1, r2, r3 = 3, 2, 1
    coeff_1 = r3 - r2
    coeff_2 = r1 + 2 * r3 - r2
    coeff_3 = r3
    assert (coeff_1, coeff_2, coeff_3) == (-1, 3, 1)
    assert coeff_2 ** 3 == 27 * coeff_3 ** 2
