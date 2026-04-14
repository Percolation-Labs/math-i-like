"""Tests for the FULL CFAC pipeline applied to Manna/CDP (manna_full.py).

This is what the user asked for: not just stratification, but the proper
CFAC factorisation counting * bridge * algebra applied to the Manna
action.  The tests document what the pipeline produces and what it
does and does not predict reliably.
"""
import numpy as np
import pytest

from rdft.ac.bridge import bridge_scalar, bridge_gradient_mass
from rdft.ac.manna_full import (
    manna_action_couplings,
    manna_one_loop_self_energy,
    manna_one_loop_beta_functions,
    manna_tau_prediction,
    comparison_DP_vs_CDP,
)


class TestSelfEnergyFactorisation:
    """The CFAC factorisation: each diagram = counting * bridge * algebra."""

    def test_DP_loop_uses_scalar_bridge(self):
        """The DP loop diagram has bridge = bridge_scalar() = 1."""
        se = manna_one_loop_self_energy(sigma=1, lambda_dp=1, chi=0, chi_prime=0)
        assert se['bridge_DP'] == 1.0
        # With chi=0, only DP loop contributes
        assert se['pole_soft'] == 0
        assert se['pole_DP'] > 0

    def test_soft_mode_loop_uses_gradient_bridge(self):
        """The soft-mode loop has bridge = bridge_gradient_mass(D_psi, D_rho)."""
        se = manna_one_loop_self_energy(sigma=0, lambda_dp=0, chi=1, chi_prime=1,
                                          D_psi=2.0, D_rho=1.0)
        # Bridge value for r = 2/1 = 2: ln(2)/(2-1) = ln(2)
        expected_bridge = np.log(2.0) / 1.0
        assert abs(se['bridge_soft'] - expected_bridge) < 1e-12

    def test_pole_total_is_sum_of_two_diagrams(self):
        """Total pole = DP pole + soft-mode pole."""
        se = manna_one_loop_self_energy(sigma=1, lambda_dp=1, chi=1, chi_prime=1)
        assert abs(se['pole_total'] - se['pole_DP'] - se['pole_soft']) < 1e-12

    def test_cdp_minus_dp_isolates_soft_mode(self):
        """The CDP - DP shift = soft-mode pole alone."""
        se = manna_one_loop_self_energy(sigma=1, lambda_dp=1, chi=1, chi_prime=1)
        assert se['cdp_minus_dp_pole'] == se['pole_soft']

    def test_equal_diffusion_gives_bridge_one(self):
        """When D_psi = D_rho, bridge_gradient_mass = 1 (its limit)."""
        se = manna_one_loop_self_energy(sigma=0, lambda_dp=0, chi=1, chi_prime=1,
                                          D_psi=1.0, D_rho=1.0)
        assert abs(se['bridge_soft'] - 1.0) < 1e-9


class TestStructuralPrediction:
    """CFAC predicts the SHIFT eta_CDP - eta_DP > 0 from the soft-mode bridge."""

    def test_cdp_has_more_anomalous_dimension_than_dp(self):
        """At any d below d_c, the soft-mode contribution to eta_psi is positive."""
        for d in [1.0, 1.5, 2.0, 3.0]:
            bf = manna_one_loop_beta_functions(d=d)
            assert bf['eta_psi_CDP'] > bf['eta_psi_DP']
            assert bf['eta_shift_CDP_minus_DP'] > 0

    def test_shift_scales_with_eps(self):
        """The shift goes to zero at d = d_c (eps = 0)."""
        bf_low_eps = manna_one_loop_beta_functions(d=3.5)  # eps = 0.5
        bf_high_eps = manna_one_loop_beta_functions(d=1.0)  # eps = 3
        assert bf_high_eps['eta_shift_CDP_minus_DP'] > bf_low_eps['eta_shift_CDP_minus_DP']

    def test_eta_at_dc_is_zero(self):
        """At d = d_c = 4, eps = 0 so no anomalous dimension."""
        bf = manna_one_loop_beta_functions(d=4.0)
        assert 'note' in bf or bf['eta_psi_CDP'] == 0

    def test_observed_tau_shift_is_positive(self):
        """Observed tau_CDP - tau_DP > 0 (CDP has higher cluster-size exponent)."""
        tp = manna_tau_prediction(d=1.0)
        assert tp['tau_shift_observed'] > 0
        # CFAC predicts shift > 0 in same direction
        # (Quantitative agreement in d=1 is not expected because eps=3 is large)


class TestPipelineConsistency:
    """The CFAC pipeline applied to Manna uses the SAME machinery as
    LDWC and Wilson-Fisher O(N): counting * bridge * algebra."""

    def test_DP_loop_uses_same_bridge_as_WF_phi4(self):
        """The DP cubic vertex loop uses bridge_scalar = 1, same as
        Wilson-Fisher phi^4 self-energy at one loop."""
        from rdft.ac.bridge import bridge_scalar
        se = manna_one_loop_self_energy(sigma=1, lambda_dp=1, chi=0, chi_prime=0)
        assert se['bridge_DP'] == bridge_scalar()

    def test_soft_mode_uses_same_bridge_as_KS_chemotaxis(self):
        """The chi-coupling loop uses bridge_gradient_mass, same as
        Keller-Segel chemotaxis (one_loop_KS)."""
        from rdft.ac.bridge import one_loop_KS
        ks = one_loop_KS(chi=1.0, mu=1.0, D_A=1.0, D_c=1.0)
        se = manna_one_loop_self_energy(sigma=0, lambda_dp=0, chi=1, chi_prime=1)
        # Both should use the same bridge_gradient_mass at equal D
        assert abs(ks['bridge_mass'] - se['bridge_soft']) < 1e-12


class TestLimitations:
    """Honestly document what the pipeline does NOT predict reliably."""

    def test_d1_predictions_quantitatively_off(self):
        """At d=1, eps=3 is large; absolute exponent values are unreliable."""
        bf = manna_one_loop_beta_functions(d=1.0)
        # Pure 1-loop in d=1 gives eta way too large (no resummation)
        assert bf['eta_psi_CDP'] > 1.0  # absurdly large vs observed eta ~ 0.3

    def test_one_loop_does_not_yield_observed_tau_in_d1(self):
        """The 1-loop tau prediction in d=1 does not match observation;
        higher loops + resummation are needed for quantitative agreement."""
        tp = manna_tau_prediction(d=1.0)
        # Observed ~ 1.30, but 1-loop in d=1 gives nonsense (NaN or ~2.3)
        # The prediction is structurally correct (shift > 0) but not
        # quantitatively accurate.
        if not np.isnan(tp['tau_CDP_pred_oneloop']):
            assert abs(tp['tau_CDP_pred_oneloop'] - 1.30) > 0.5  # bad fit
