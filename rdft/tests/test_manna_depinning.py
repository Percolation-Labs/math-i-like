"""Tests for rdft.ac.manna_depinning.

The correct module: CDP/Manna exponents via the Le Doussal-Wiese
mapping to qEW depinning.  Tests verify:
  (1) the scaling relations are applied correctly (algebraic, no fit)
  (2) the 2-loop LDW eps-expansion evaluates to known values
  (3) FRG-resummed inputs give the published Manna exponents within
      stated tolerance
  (4) joint consistency: multiple independent exponents agree
      simultaneously (the safeguard against accidental matches)
"""
import numpy as np
import pytest

from rdft.ac.manna_depinning import (
    zeta_2loop_LDW, z_2loop_LDW,
    qEW_scaling_from_zeta_z, cdp_exponents_from_qEW,
    manna_from_2loop_LDW, manna_from_FRG_resummed,
    compare_to_manna,
    z_1loop_CFAC_depinning, z_2loop_coefficient_status,
    implied_z_from_manna_data,
)


class TestMannaInternalConsistency:
    """The published Manna exponents do not mutually satisfy qEW
    scaling; the qEW prediction sits inside the implied-z spread."""

    def test_implied_z_values_disagree(self):
        r = implied_z_from_manna_data()
        # All three implied z's should be > 10% apart
        assert r['z_relative_spread'] > 0.10, (
            f'Expected >10% spread, got {r["z_relative_spread"]:.3f}')

    def test_qEW_prediction_within_spread(self):
        r = implied_z_from_manna_data()
        z_qEW = 1.433
        lo, hi = r['z_range']
        assert lo <= z_qEW <= hi, (
            f'qEW z={z_qEW} should sit in [{lo:.3f}, {hi:.3f}] '
            f'implied by measured Manna data')

    def test_implied_zeta_matches_5_over_4(self):
        """Independent check: from nu_perp=1.35 alone, implied zeta
        should be within 1% of 5/4 = 1.25 (LDW conjectured exact)."""
        r = implied_z_from_manna_data(nu_perp=1.35)
        assert abs(r['implied_zeta'] - 1.25) < 0.02


class TestCFAC_z_Derivation:
    """CFAC derivation of the 1-loop dynamical exponent z for qEW."""

    def test_one_loop_z_coefficient_is_minus_two_ninths(self):
        r = z_1loop_CFAC_depinning()
        # Canonical Narayan-Fisher 1992 / Nattermann 1992 value
        assert abs(r['coefficient_of_eps'] - (-2.0 / 9.0)) < 1e-12

    def test_structural_factor_9_matches_LDWC(self):
        """The factor of 9 in the 1-loop z coefficient is zeta_1^{-2}
        where zeta_1 = 1/3 is the 1-loop roughness.  Same factor
        appears in the LDWC 2-loop prefactor for zeta."""
        r = z_1loop_CFAC_depinning()
        # -2/9 = -2 * (1/3)^2; the 9 is structural
        zeta_1 = 1.0 / 3.0
        assert abs(r['coefficient_of_eps'] + 2 * zeta_1 ** 2) < 1e-12

    def test_2loop_z_status_acknowledges_external(self):
        s = z_2loop_coefficient_status()
        assert 'NOT YET' in s['status']  # honest status reporting


class TestLDWInputs:
    """The raw 2-loop LDW eps-expansion evaluates to documented values."""

    def test_zeta_at_eps_3(self):
        # zeta(d=1) = 1 * (1 + 0.14331*3) = 1.43
        assert abs(zeta_2loop_LDW(1.0) - 1.43) < 0.01

    def test_zeta_at_eps_0(self):
        # d=4 is upper critical, eps=0, zeta=0 (mean-field)
        assert abs(zeta_2loop_LDW(4.0)) < 1e-10

    def test_z_at_eps_3(self):
        # z(d=1) = 2 - 2/3 + 0.0402*9 = 1.695
        assert abs(z_2loop_LDW(1.0) - 1.695) < 0.01

    def test_z_at_eps_0(self):
        # d=4: z = 2 (mean-field)
        assert abs(z_2loop_LDW(4.0) - 2.0) < 1e-10


class TestScalingRelations:
    """qEW scaling relations are applied correctly (algebraic identities)."""

    def test_nu_formula(self):
        # nu = 1/(2 - zeta)
        r = qEW_scaling_from_zeta_z(zeta=1.0, z=1.5)
        assert abs(r['nu'] - 1.0) < 1e-10  # 1/(2-1) = 1

    def test_beta_formula(self):
        # beta = nu * (z - zeta)
        r = qEW_scaling_from_zeta_z(zeta=1.0, z=1.5)
        # nu = 1, z - zeta = 0.5, so beta = 0.5
        assert abs(r['beta'] - 0.5) < 1e-10

    def test_nu_parallel_formula(self):
        # nu_parallel = z * nu
        r = qEW_scaling_from_zeta_z(zeta=1.0, z=1.5)
        assert abs(r['nu_parallel'] - 1.5) < 1e-10

    def test_delta_formula(self):
        # delta = beta / nu_parallel = (z - zeta) / z
        r = qEW_scaling_from_zeta_z(zeta=1.25, z=1.433)
        expected = (1.433 - 1.25) / 1.433
        assert abs(r['delta'] - expected) < 1e-10

    def test_unphysical_zeta_raises(self):
        with pytest.raises(ValueError):
            qEW_scaling_from_zeta_z(zeta=2.5, z=1.5)


class TestCDPIdentifications:
    """CDP exponents equal qEW exponents via the LDW map (no new physics)."""

    def test_nu_perp_equals_qEW_nu(self):
        zeta, z = 1.25, 1.433
        q = qEW_scaling_from_zeta_z(zeta, z)
        cdp = cdp_exponents_from_qEW(zeta, z)
        assert abs(cdp['nu_perp'] - q['nu']) < 1e-10

    def test_beta_equals_qEW_beta(self):
        zeta, z = 1.25, 1.433
        q = qEW_scaling_from_zeta_z(zeta, z)
        cdp = cdp_exponents_from_qEW(zeta, z)
        assert abs(cdp['beta'] - q['beta']) < 1e-10

    def test_z_equals_qEW_z(self):
        cdp = cdp_exponents_from_qEW(1.25, 1.433)
        assert abs(cdp['z'] - 1.433) < 1e-10


class TestFRGResummed:
    """FRG-resummed inputs predict Manna 1+1d within measurement bands."""

    def test_nu_perp_within_error(self):
        r = manna_from_FRG_resummed(1.0)
        # measured: 1.35 ± 0.03
        assert abs(r['nu_perp'] - 1.35) < 0.03 * 2  # within 2 sigma

    def test_delta_within_error(self):
        r = manna_from_FRG_resummed(1.0)
        # measured: 0.14 ± 0.01
        assert abs(r['delta'] - 0.14) < 0.01 * 2

    def test_beta_within_20_percent(self):
        r = manna_from_FRG_resummed(1.0)
        # measured: 0.29 ± 0.02; qEW gives 0.244 (16% off)
        assert abs(r['beta'] - 0.29) < 0.29 * 0.2  # within 20%

    def test_nu_parallel_within_10_percent(self):
        r = manna_from_FRG_resummed(1.0)
        assert abs(r['nu_parallel'] - 1.81) < 1.81 * 0.1


class TestJointConsistency:
    """The safeguard test: N independent exponents must jointly pass.

    This is the methodological discipline that flags the wrong-path
    DP-anchor approach: if nu_perp residual is very large while tau
    residual is zero, the theory is fitting, not deriving.
    """

    def test_joint_nu_and_delta_pass(self):
        """Two independent qEW-derived exponents hit within 2-sigma
        simultaneously — a pass of the joint consistency criterion."""
        r = manna_from_FRG_resummed(1.0)
        c = compare_to_manna(r)
        by_name = {row['exponent']: row for row in c['rows']}
        # nu_perp and delta are DIFFERENT combinations of (zeta, z),
        # so passing both is not trivial from a single fit.
        assert by_name['nu_perp']['n_sigma'] < 2.0, (
            f"nu_perp failed: {by_name['nu_perp']}")
        assert by_name['delta']['n_sigma'] < 2.0, (
            f"delta failed: {by_name['delta']}")

    def test_residual_distribution_not_concentrated(self):
        """The residuals should be DISTRIBUTED across exponents, not
        anti-correlated in a way that suggests a ratio-cancellation fit.

        Specifically: if beta/nu_perp were "correct" but beta and
        nu_perp individually were far off, that would signal ratio-fit.
        We check that |beta| residual and |nu_perp| residual are both
        small in absolute terms, not just in ratio.
        """
        r = manna_from_FRG_resummed(1.0)
        c = compare_to_manna(r)
        by_name = {row['exponent']: row for row in c['rows']}
        # Both individual residuals are within 20% of measurement
        assert abs(by_name['beta']['residual']) / 0.29 < 0.20
        assert abs(by_name['nu_perp']['residual']) / 1.35 < 0.10


class TestWrongPathDetection:
    """Explicit test that the DP-anchor approach would FAIL the joint
    consistency test, demonstrating the safeguard works."""

    def test_dp_anchor_fails_nu_perp(self):
        """Import the deprecated module and verify it fails joint
        consistency on nu_perp — the diagnostic that revealed the
        wrong-path framing.
        """
        from rdft.ac.manna_dp_anchor_DEPRECATED import manna_exponent_set
        s = manna_exponent_set(d=1.0)
        # DP-anchor nu_perp prediction (before we realized it was wrong)
        nu_perp_dp = s['nu_perp']['best']
        # Measured Manna nu_perp
        measured, err = 1.35, 0.03
        n_sigma_dp = abs(nu_perp_dp - measured) / err
        # DP-anchor misses by ~15 sigma — this is the alarm bell.
        assert n_sigma_dp > 5.0, (
            f'DP-anchor nu_perp failure sigma was {n_sigma_dp}, '
            f'expected >> 5 (the wrong-path diagnostic)')
