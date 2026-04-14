"""
Tests for rdft.ac.stratification.

Locks in the Puiseux-order, discriminant, canonical-family, and CRN
translator API used by the experiment scripts.
"""

import numpy as np
import pytest

from rdft.ac.stratification import (
    discriminant_in_z,
    discriminant_roots,
    puiseux_order,
    lambda_scgf,
    on_stratum_C_k,
    canonical_family,
    canonical_z_star,
    is_dyadic,
    banderier_drmota_status,
    phi_from_reactions,
)


def test_canonical_family_satisfies_C_k_for_k_2_through_7():
    for k in range(2, 8):
        phi = canonical_family(k, beta=-4)
        assert on_stratum_C_k(phi, k), f'k={k}: should be on C_{k}'


def test_canonical_family_z_star_is_one_over_beta_when_dominant():
    """For canonical phi_{k, beta}, z_star = 1/beta when the 1/k branch is
    dominant.  Dominance threshold beta scales as: k=3 needs |beta|>=4,
    k=4 >=5, k=5 >=8, k=6 >=8, k=7 >=10 (from numerical scan in Exp 5)."""
    cases = [(3, -4), (4, -5), (5, -8), (6, -8), (7, -10)]
    for k, beta in cases:
        phi = canonical_family(k, beta)
        _, z = puiseux_order(phi)
        # higher k incurs numpy.roots precision loss
        tol = 5e-3 if k >= 6 else 1e-4
        assert abs(abs(z) - abs(1 / beta)) < tol, (
            f'k={k}, beta={beta}: |z*|={abs(z):.4f}, expected 1/|beta|={abs(1/beta):.4f}'
        )


def test_321_CRN_phi_with_proper_DP_extraction():
    """A->3A:3, 2A->A:2, 2A->3A:1 with full Doi-Peliti vertex extraction gives
    phi = 1 + 8 G + 3 G^2 + G^3.

    HISTORICAL NOTE: An earlier derivation in the cube-root CRN paper claimed
    this CRN gives phi = 1 - G + 3 G^2 + G^3 by under-counting vertices (only
    keeping the highest-j vertex per reaction, instead of summing the full
    binomial expansion of (z+1)^l - (z+1)^k).  The correct DP expansion gives
    +8 G in the linear coefficient, not -G.
    """
    reactions = [(1, 3, 3.0), (2, 1, 2.0), (2, 3, 1.0)]
    phi = phi_from_reactions(reactions)
    expected = [1.0, 8.0, 3.0, 1.0]
    assert phi == pytest.approx(expected)


def test_321_CRN_is_on_C3_algebraically_but_NOT_dominant():
    """For phi = 1 + 8G + 3G^2 + G^3: b=3, c=1 satisfies b^3=27c^2 (on C_3
    algebraic locus), but the dominant branch is square-root (k=2), NOT
    cube-root.  This is the same Banderier-Drmota mechanism as the trap case
    in test_BD_null: positive systems can SIT on C_k locus but cannot HAVE it
    as their leading asymptotic.

    The actual cube-root universality requires the canonical family
    phi_{3, -4}(G) = (1+G)^3 - 4G = 1 - G + 3 G^2 + G^3, with NEGATIVE linear
    coefficient.  No positive-rate single-species CRN of A->kA reactions
    realises this.
    """
    reactions = [(1, 3, 3.0), (2, 1, 2.0), (2, 3, 1.0)]
    phi = phi_from_reactions(reactions)
    # On C_3 algebraically (b^3 = 27 c^2)
    assert on_stratum_C_k(phi, 3)
    # But dominant branch is square-root
    k, z = puiseux_order(phi)
    assert k == 2, f'expected k_dom=2 (square-root), got k={k}'


def test_canonical_phi_3_minus4_IS_cube_root_dominant():
    """The canonical family at (k=3, beta=-4): phi = (1+G)^3 - 4G has cube-root
    dominance, |z*|=1/4.  This is the actual mathematical cube-root CRN — it
    requires signed structure to realise as a CRN."""
    phi = canonical_family(3, -4)
    k, z = puiseux_order(phi)
    assert k == 3, f'expected k_dom=3 (cube-root), got k={k}'
    assert abs(abs(z) - 0.25) < 1e-6


def test_positive_truncation_of_cube_root_CRN_falls_to_C_2():
    """phi = 1 + G + 3 G^2 + G^3 sits algebraically on C_3 (b^3 = 27 c^2 holds)
    but the dominant branch is C_2 (square-root) because a closer-to-origin
    branch wins.  Banderier-Drmota mechanism."""
    phi_pos = [1.0, 1.0, 3.0, 1.0]
    assert on_stratum_C_k(phi_pos, 3), 'C_3 algebraic locus IS satisfied'
    k, z = puiseux_order(phi_pos)
    assert k == 2, f'but dominant branch should be square-root, got k={k}'


def test_dyadic_status():
    assert is_dyadic(2)
    assert is_dyadic(4)
    assert is_dyadic(8)
    assert not is_dyadic(3)
    assert not is_dyadic(5)
    assert not is_dyadic(6)
    assert not is_dyadic(7)


def test_BD_forbidden_strings():
    assert 'FORBIDDEN' in banderier_drmota_status(3)
    assert 'FORBIDDEN' in banderier_drmota_status(5)
    assert 'generic' in banderier_drmota_status(2)
    assert 'allowed' in banderier_drmota_status(4)


def test_lambda_scgf_for_canonical_cube_root():
    """For canonical phi_{3, -4}, |z*| = 1/4, so lambda = log 4."""
    phi = canonical_family(3, -4)
    lam = lambda_scgf(phi)
    assert abs(lam - np.log(4)) < 1e-6


def test_discriminant_factors_correctly_for_canonical_cube_root():
    """disc_G(F) for phi_{3,-4} = 1 - G + 3 G^2 + G^3 has roots {0, -1/4, 4/11}
    with -1/4 a double root (signature of cube-root branch)."""
    phi = canonical_family(3, -4)
    roots = discriminant_roots(phi)
    abs_roots = sorted(set(round(abs(r), 4) for r in roots))
    assert 0.25 in abs_roots
    assert any(abs(abs(r) - 4 / 11) < 1e-4 for r in roots)
