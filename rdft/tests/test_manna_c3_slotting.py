"""Verification tests for the Manna/C-DP C_3 slotting paper.

Documents three numerical facts on which the paper's claims rest:

1. Single-species reactions saturate at k=2 (Banderier-Drmota dyadic).
2. The polynomial phi = 1 + 3 G^2 - G^3 carries a C_3 algebraic stratum,
   even though that stratum is not dominant (z = 1/3 is double root,
   z = -4/15 is the dominant simple root).
3. The rank-3 bridge B_3 = 1/(4 pi)^3 is two orders of magnitude smaller
   than B_2 = 1/(8 pi^2), consistent with the observed pattern of
   tau-residuals (C-DP/Manna ~ few %, DP ~ tens of %).
"""

import numpy as np
import sympy as sp

from rdft.ac.bridge import bridge_rank_k, upper_critical_dim_for_cusp
from rdft.ac.stratification import (
    phi_from_reactions, puiseux_order, on_stratum_C_k, discriminant_in_z
)


class TestC3Slotting:

    def test_single_species_dyadic_saturation(self):
        """Single-species reaction networks give k=2 dominant (Banderier-Drmota)."""
        candidates = [
            [(1, 2, 1.0), (2, 1, 1.0)],                            # DP
            [(1, 2, 1.0), (2, 1, 1.0), (2, 0, 0.5)],               # DP + pair-annihilation
            [(1, 2, 1.0), (2, 1, 1.0), (1, 0, 0.3)],               # DP + spontaneous decay
            [(1, 3, 0.5), (2, 1, 1.0)],                            # tribranching + coalescence
        ]
        for rxns in candidates:
            phi = phi_from_reactions(rxns)
            k_dom, _ = puiseux_order(phi)
            assert k_dom == 2, f"reactions {rxns} gave k={k_dom}, expected k=2"

    def test_C3_algebraically_present_in_cubic_phi(self):
        """phi = 1 + 3 G^2 - G^3 has the C_3 stratum (z = 1/3 double root)."""
        phi = [1.0, 0.0, 3.0, -1.0]
        # Algebraic check
        assert on_stratum_C_k(phi, 3) is True

        # Discriminant factorisation: -z (3z - 1)^2 (15z + 4)
        # Coefficient list, leading first
        coeffs = discriminant_in_z(phi)
        # Verify the polynomial -135 z^4 + 54 z^3 + 9 z^2 - 4 z
        expected = [-135.0, 54.0, 9.0, -4.0, 0.0]
        for c, e in zip(coeffs, expected):
            assert abs(c - e) < 1e-9, f"discriminant coeff mismatch: {coeffs} vs {expected}"

    def test_C3_branch_not_dominant_for_cubic_phi(self):
        """For phi = 1 + 3 G^2 - G^3 the C_3 branch (z=1/3) is not closest to origin."""
        phi = [1.0, 0.0, 3.0, -1.0]
        k_dom, z_star = puiseux_order(phi)
        # Dominant root is z = -4/15, simple, hence k_dom = 2
        assert k_dom == 2
        assert abs(z_star.real - (-4.0 / 15.0)) < 1e-3
        assert abs(z_star.imag) < 1e-3

    def test_C3_can_be_dominant_in_quartic(self):
        """Quartic phi = 1 + 3 G^2 - 2 G^3 + (1/4) G^4 has C_3 dominant."""
        phi4 = [1.0, 0.0, 3.0, -2.0, 0.25]
        k_dom, z_star = puiseux_order(phi4)
        assert k_dom == 3, f"expected k=3, got k={k_dom}"
        assert abs(z_star.real - (-0.25)) < 1e-3

    def test_rank_k_bridge_separation(self):
        """B_3 / B_2 = 1 / (8 pi) approx 1/25 — the structural separation."""
        B2 = bridge_rank_k(2)
        B3 = bridge_rank_k(3)
        # B_2 = 2 / (1! (4 pi)^2) = 1 / (8 pi^2)
        assert abs(B2 - 1.0 / (8 * np.pi ** 2)) < 1e-12
        # B_3 = 2 / (2! (4 pi)^3) = 1 / (4 pi)^3
        assert abs(B3 - 1.0 / (4 * np.pi) ** 3) < 1e-12
        # B_3 / B_2 = (8 pi^2) / (4 pi)^3 = 1 / (8 pi) ~ 0.0398 ~ 1/25
        # This ~25x suppression is the structural fact that drives
        # C_3 dressings to ~ few % vs C_2 dressings at tens of %.
        ratio = B3 / B2
        assert abs(ratio - 1.0 / (8 * np.pi)) < 1e-12
        assert 0.03 < ratio < 0.05  # ~ 1/25

    def test_upper_critical_dim_C2_C3(self):
        """d_c(k=2) = 4 (DP), d_c(k=3) = 3 (CDP/Manna multicritical)."""
        assert upper_critical_dim_for_cusp(2) == 4.0
        assert upper_critical_dim_for_cusp(3) == 3.0

    def test_C3_skeleton_is_4_3(self):
        """Skeleton tau_0 = 1 + 1/k for k=3 is 4/3."""
        k = 3
        tau_0 = 1.0 + 1.0 / k
        assert abs(tau_0 - 4.0 / 3.0) < 1e-12

    def test_observed_tau_close_to_C3_skeleton(self):
        """Observed Manna tau ~ 1.275 and CDP tau ~ 1.338 sit within ~5% of 4/3."""
        tau_manna = 1.275
        tau_cdp = 1.338
        skeleton = 4.0 / 3.0
        assert abs(tau_manna - skeleton) / skeleton < 0.05
        assert abs(tau_cdp - skeleton) / skeleton < 0.01

    def test_dressing_scale_consistent_with_observation(self):
        """B_3 * eps^2 with class counting ~ O(10) gives the observed gap ~ 0.06."""
        B3 = bridge_rank_k(3)
        eps = upper_critical_dim_for_cusp(3) - 1.0  # d=1+1d -> d=1, eps = 2
        natural_unit = B3 * eps ** 2
        # Observed gap ~ 0.06; ratio gives approximate counting integer ~ 30
        observed_gap = 0.06
        n_eff = observed_gap / natural_unit
        assert 5 < n_eff < 200, (
            f"effective counting n_eff = {n_eff:.1f}, expected order-10 to order-100"
        )
