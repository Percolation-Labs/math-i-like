"""
Tests for rdft.ac.tilted — current-tilted DSE + SCGF.
"""

import numpy as np
import pytest

from rdft.ac.tilted import (
    tilt_reaction_rates,
    tilted_phi,
    scgf,
    scgf_with_branch,
    detect_dynamical_phase_transition,
)


CUBE_ROOT_CRN = [(1, 3, 3.0), (2, 1, 2.0), (2, 3, 1.0)]


def test_tilt_at_zero_recovers_original():
    """exp(0) = 1, so tilt at s=0 must reproduce the bare phi."""
    from rdft.ac.stratification import phi_from_reactions
    bare = phi_from_reactions(CUBE_ROOT_CRN)
    tilted = tilted_phi(CUBE_ROOT_CRN, [0], 0.0)
    assert tilted == pytest.approx(bare)


def test_tilt_increases_branching_decreases_z_star():
    """Tilting the A->3A rate up (s>0) should pull the branch point closer
    to the origin (smaller |z*|, larger lambda)."""
    from rdft.ac.stratification import lambda_scgf
    s_arr, lams = scgf(CUBE_ROOT_CRN, [0], np.linspace(-0.3, 0.3, 13))
    # lam should be monotone-ish increasing in s (more current = more growth)
    lam_left = lams[0]
    lam_right = lams[-1]
    assert lam_right > lam_left, f'lam(0.3)={lam_right} should exceed lam(-0.3)={lam_left}'


def test_dynamical_phase_transition_at_C_3_canonical_phi():
    """For the canonical phi_{3, beta=-4}, tilting beta off -4 moves the
    system off C_3 (the cube-root branch persists but the dominance shifts).
    Theorem A.2 + Exp 1 predict d^2 lambda / d s^2 has structure near s=0."""
    from rdft.ac.stratification import lambda_scgf, canonical_family

    s_vals = np.linspace(-0.4, 0.4, 81)
    lams = np.array([lambda_scgf(canonical_family(3, -4 * np.exp(s))) for s in s_vals])
    d_lam = np.gradient(lams, s_vals)
    d2_lam = np.gradient(d_lam, s_vals)

    # At s=0 the system is exactly at C_3 (cube-root dominant).  For s>0,
    # |beta| grows so cube-root remains dominant.  For s<0, |beta| shrinks
    # below the dominance threshold ~4, and the branch order can change.
    # We expect d^2 lambda / d s^2 to have a feature near or below s=0.
    median = np.median(np.abs(d2_lam) + 1e-12)
    max_d2 = np.max(np.abs(d2_lam))
    assert max_d2 > 1.5 * median, (
        f'expected d2_lam to have a feature; max={max_d2:.3f} vs median={median:.3f}'
    )


def test_canonical_family_branch_order_changes_with_tilt():
    """For phi_{3, beta} with beta varying via s: at sufficient |beta| the
    cube-root branch dominates, otherwise it does not."""
    from rdft.ac.stratification import puiseux_order, canonical_family
    # beta = -4, -8, -16: all dominant cube-root
    for s in [0, np.log(2), np.log(4)]:
        phi = canonical_family(3, -4 * np.exp(s))
        k, _ = puiseux_order(phi)
        assert k == 3, f's={s}, beta={-4*np.exp(s):.3f}: expected k=3, got {k}'
    # beta = -2 (s = -log(2)): below dominance threshold, should not be cube-root
    phi_below = canonical_family(3, -2.0)
    k_below, _ = puiseux_order(phi_below)
    assert k_below == 2, f'below threshold, expected k=2, got {k_below}'
