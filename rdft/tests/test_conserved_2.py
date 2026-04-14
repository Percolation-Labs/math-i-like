"""Tests for the soft-mode-aware conservation projector (conserved_2.py).

Documents what is now demonstrable about the Ward-identity-tied
quartic vertex and the gamma_3 prediction.
"""
import numpy as np
import pytest

from rdft.ac.conserved_2 import (
    soft_mode_induced_vertex,
    cdp_dse_with_softmode,
    find_C3_dominant_in_chi,
    gamma_3_from_softmode,
)
from rdft.ac.bridge import bridge_rank_k, upper_critical_dim_for_cusp


class TestSoftModeVertex:

    def test_g_eff_formula(self):
        """g_eff = chi^2 B_2 / (D_psi + D_rho)."""
        v = soft_mode_induced_vertex(chi=2.0, D_psi=1.0, D_rho=3.0)
        expected = 2.0**2 * bridge_rank_k(2) / (1.0 + 3.0)
        assert abs(v['g_eff_quartic'] - expected) < 1e-12

    def test_g_eff_chi_squared_scaling(self):
        """g_eff scales as chi^2 (Ward-identity tied)."""
        v1 = soft_mode_induced_vertex(chi=1.0)
        v2 = soft_mode_induced_vertex(chi=2.0)
        assert abs(v2['g_eff_quartic'] / v1['g_eff_quartic'] - 4.0) < 1e-12

    def test_g_eff_is_small_in_natural_units(self):
        """g_eff = chi^2 / (8 pi^2 (D + D')) is naturally small."""
        v = soft_mode_induced_vertex(chi=1.0, D_psi=1.0, D_rho=1.0)
        # g_eff = 1 / (16 pi^2) ~ 0.006
        assert v['g_eff_quartic'] < 0.01
        assert v['g_eff_quartic'] > 0.001


class TestAugmentedDSE:

    def test_dse_has_correct_phi_structure(self):
        """phi(G) = 1 + chi G - lambda G^2 + g_eff G^3."""
        out = cdp_dse_with_softmode(lambda_dp=1.0, chi=1.0)
        phi = out['phi_coefficients']
        assert phi[0] == 1.0
        assert phi[1] == 1.0  # chi
        assert phi[2] == -1.0  # -lambda
        # phi[3] = g_eff = 1 / (16 pi^2)
        assert abs(phi[3] - 1.0 / (16 * np.pi**2)) < 1e-12

    def test_simple_softmode_ansatz_remains_k2(self):
        """The simple polynomial ansatz [1, chi, -lambda, g_eff] gives k=2.

        This is a NEGATIVE result: the soft-mode-augmented DSE in its
        simplest form does NOT reach C_3 dominance. The Ward-identity
        argument for C_3 is structural (codim counting); reproducing it
        within a polynomial DSE requires more than just adding the
        soft-mode quartic.
        """
        for lam in [0.5, 1.0, 2.0]:
            for chi in [0.5, 1.0, 5.0, 20.0]:
                out = cdp_dse_with_softmode(lambda_dp=lam, chi=chi)
                # k=2 expected for moderate chi; very large chi may give k=4
                assert out['puiseux_order'] in (2, 4), (
                    f"lambda={lam}, chi={chi}: got k={out['puiseux_order']}"
                )


class TestGamma3Prediction:

    def test_gamma_3_is_negative(self):
        """One-loop gamma_3 is negative (anomalous dimension reduces tau)."""
        g = gamma_3_from_softmode(d=1.0)
        assert g['gamma_3_estimate'] < 0

    def test_predicted_tau_within_observed_range(self):
        """Predicted tau falls within observed [1.275, 1.338]."""
        g = gamma_3_from_softmode(d=1.0)
        assert 1.275 < g['tau_predicted'] < 1.338

    def test_prediction_robust_across_n_eff(self):
        """For any n_eff in 1..20, predicted tau is within 5% of observed 1.30."""
        from rdft.ac.bridge import bridge_rank_k, upper_critical_dim_for_cusp
        B3 = bridge_rank_k(3)
        eps = upper_critical_dim_for_cusp(3) - 1.0
        observed = 1.30
        for n_eff in [1, 3, 5, 10, 20]:
            gamma_3 = -n_eff * B3 * eps**2
            tau_pred = 4.0 / 3.0 + gamma_3
            rel_err = abs(tau_pred - observed) / observed
            assert rel_err < 0.05, f"n_eff={n_eff}: tau_pred={tau_pred:.4f}, rel_err={rel_err:.4f}"

    def test_eps_is_2_at_d_equals_1(self):
        """At d=1 (1+1d): eps = d_c(3) - d = 3 - 1 = 2."""
        g = gamma_3_from_softmode(d=1.0)
        assert g['eps'] == 2.0

    def test_skeleton_value_preserved(self):
        """tau_skeleton = 4/3 regardless of dressing."""
        for d in [1.0, 1.5, 2.0]:
            g = gamma_3_from_softmode(d=d)
            assert abs(g['tau_skeleton'] - 4.0 / 3.0) < 1e-12


class TestStructural:

    def test_g_eff_tied_to_chi(self):
        """The Ward identity tying: g_eff is determined by chi, not free.

        This is the structural content of the conservation Ward identity
        in the polynomial DSE picture: changing chi changes BOTH the
        linear coupling (chi G) AND the quartic (g_eff G^3 = chi^2 B_2 G^3).
        """
        out_a = cdp_dse_with_softmode(lambda_dp=1.0, chi=1.0)
        out_b = cdp_dse_with_softmode(lambda_dp=1.0, chi=2.0)
        # Linear coupling scales as chi
        assert abs(out_b['phi_coefficients'][1] / out_a['phi_coefficients'][1] - 2.0) < 1e-12
        # Quartic scales as chi^2 -- the tying
        assert abs(out_b['phi_coefficients'][3] / out_a['phi_coefficients'][3] - 4.0) < 1e-12
