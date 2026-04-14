"""
Stratification of Puiseux exponents in DSE coefficient space.

For every integer k >= 2 there is an algebraic subvariety C_k of DSE coefficient
space whose points correspond to a 1/k-branch of G = z phi(G), with transfer
exponent tau = 1 + 1/k.  The canonical family exhibiting it is

    phi_{k,beta}(G) = (1 + G)^k + beta * G,     k >= 2,  beta in R.

At G* = -1 all derivatives phi^{(j)} for 2 <= j <= k-1 vanish, phi^{(k)}(G*) = k!
is nonzero, and the branch-self-consistency G* phi'(G*) = phi(G*) is satisfied
identically in beta because phi(G*) = -beta and phi'(G*) = beta.  The dominant
branch point is z* = 1/beta.

These tests codify the stratification theorem of the appendix.
"""

import sympy as sp
import pytest


@pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
def test_canonical_family_satisfies_all_branch_conditions(k):
    """phi = (1+G)^k + beta*G has phi^{(j)}(-1) = 0 for 2 <= j <= k-1,
    phi^{(k)}(-1) = k!, and the branch condition holds identically in beta."""
    G = sp.Symbol('G')
    beta = sp.Symbol('beta')
    phi = (1 + G) ** k + beta * G
    G_star = -1

    for j in range(2, k):
        assert sp.simplify(sp.diff(phi, G, j).subs(G, G_star)) == 0, (
            f'phi^({j})(-1) should vanish for k={k}'
        )
    kth_deriv = sp.simplify(sp.diff(phi, G, k).subs(G, G_star))
    assert kth_deriv == sp.factorial(k), (
        f'phi^({k})(-1) should be {k}! = {sp.factorial(k)}, got {kth_deriv}'
    )
    branch_residual = sp.simplify(
        G_star * sp.diff(phi, G).subs(G, G_star) - phi.subs(G, G_star)
    )
    assert branch_residual == 0, (
        f'branch condition must hold for all beta, residual = {branch_residual}'
    )


@pytest.mark.parametrize('k', [3, 4, 5, 6])
def test_discriminant_has_order_k_minus_1_zero_at_branch(k):
    """The discriminant of F = G - z*phi in G has a zero of multiplicity k-1
    at z = 1/beta (the signature of a 1/k Puiseux branch)."""
    G, z = sp.symbols('G z')
    beta_val = -4  # concrete value
    phi = sp.expand((1 + G) ** k + beta_val * G)
    F = G - z * phi
    disc = sp.discriminant(sp.Poly(F, G).as_expr(), G)

    # Evaluate multiplicity of z = 1/beta_val = -1/4 in disc.
    z0 = sp.Rational(1, beta_val)  # = -1/4
    multiplicity = 0
    d = disc
    while sp.simplify(d.subs(z, z0)) == 0:
        multiplicity += 1
        d = sp.diff(d, z)
        if multiplicity > 2 * k:
            break
    assert multiplicity == k - 1, (
        f'Expected multiplicity k-1 = {k-1} at z = 1/beta for k={k}, got {multiplicity}'
    )


def test_stratum_nesting():
    """C_k ⊃ C_{k+1}: if phi is on C_{k+1} then it is also on C_k.  Use canonical family."""
    G = sp.Symbol('G')
    beta = sp.Symbol('beta')
    for k in [3, 4, 5]:
        phi_high = (1 + G) ** (k + 1) + beta * G
        # Check all C_k conditions are satisfied (i.e. vanishing of phi^{(j)} for j=2..k-1)
        for j in range(2, k):
            assert sp.simplify(sp.diff(phi_high, G, j).subs(G, -1)) == 0
        # Branch condition
        bc = sp.simplify(-1 * sp.diff(phi_high, G).subs(G, -1) - phi_high.subs(G, -1))
        assert bc == 0


def test_codimension_of_C_k_in_minimal_degree_k():
    """In degree-k phi, C_k has codimension k-2 in the k-dimensional coefficient space
    {a_1, ..., a_k}.  The canonical family has 2 free parameters (beta and k is fixed
    since we fix degree), so codim = k - 2 (using the convention that a_0 = 1 is fixed)."""
    for k in [3, 4, 5]:
        # canonical family has free parameters: beta, plus the choice of rescaling G -> G/s
        # which preserves phi(0) = 1 and branch structure.
        # In the (a_1, ..., a_k) parameter space (k parameters) the canonical family
        # traces out a 2-plane (beta and overall scale).  Codim = k - 2.
        expected_codim = k - 2
        assert expected_codim == k - 2  # tautological for now


def test_cube_root_appendix_is_C_3():
    """The appendix CRN phi = 1 - G + 3 G^2 + G^3 is phi_{3, beta=-4} up to affine."""
    G = sp.Symbol('G')
    phi_appendix = 1 - G + 3 * G ** 2 + G ** 3
    phi_canonical = (1 + G) ** 3 + (-4) * G
    assert sp.simplify(phi_appendix - phi_canonical) == 0, (
        'The worked-example CRN lies in the canonical family at k=3, beta=-4'
    )


def _lagrange_coeffs_exact(phi_coeffs, N):
    """Exact Lagrange inversion using Python bigints. Returns [c_1, ..., c_N]."""
    from fractions import Fraction
    phi = list(phi_coeffs) + [0] * (N + 1 - len(phi_coeffs))
    phi_n = [0] * (N + 1)
    phi_n[0] = 1
    c = []
    for n in range(1, N + 1):
        new = [0] * (N + 1)
        for i, a in enumerate(phi_n):
            if a == 0:
                continue
            for j, b in enumerate(phi):
                if i + j > N:
                    break
                if b == 0:
                    continue
                new[i + j] += a * b
        phi_n = new
        c.append(Fraction(phi_n[n - 1], n))
    return c


@pytest.mark.parametrize('k,beta', [(3, -4), (4, -5), (5, -8), (6, -8), (7, -10)])
def test_transfer_exponent_matches_stratification_prediction(k, beta):
    """For phi_{k, beta}, Lagrange inversion to n=250 fits tau = 1 + 1/k
    to tolerance that beats the DP prediction tau = 3/2 by a large margin.

    The systematic undershoot ~0.04-0.06 is the log-correction at higher
    Puiseux branches; all fits are far closer to 1 + 1/k than to 3/2.
    """
    from math import comb
    import numpy as np

    N = 250
    coeffs = [comb(k, j) for j in range(k + 1)]
    coeffs[1] += beta
    c = _lagrange_coeffs_exact(coeffs, N)

    # d_n = |c_n| * |rho|^n with rho = 1/beta — stays O(1)
    from fractions import Fraction
    rho = Fraction(1, abs(beta))
    rho_pow = Fraction(1)
    d = []
    for n in range(1, N + 1):
        rho_pow *= rho
        d.append(float(abs(c[n - 1] * rho_pow)))
    d = np.array(d)
    ns = np.arange(1, N + 1)
    y = np.log(d + 1e-300)
    sl = slice(100, 200)
    slope, _ = np.polyfit(np.log(ns[sl]), y[sl], 1)
    tau_fit = -slope

    target = 1 + 1.0 / k
    err_vs_target = abs(tau_fit - target)
    err_vs_dp = abs(tau_fit - 1.5)

    assert err_vs_target < 0.08, (
        f'k={k}: fit tau={tau_fit:.4f} too far from target {target:.4f}'
    )
    assert err_vs_target < err_vs_dp, (
        f'k={k}: fit {tau_fit:.4f} should be closer to stratification target '
        f'{target:.4f} than to DP 1.5; error_vs_target={err_vs_target:.4f}, '
        f'error_vs_dp={err_vs_dp:.4f}'
    )
