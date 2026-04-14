"""Tests for rdft.ac.signed_projector — microscopic C_3 from CDP action.

Verifies:
  (1) The chi*chi' bubble produces a SIGNED (negative) a_3 coefficient
      in the effective DSE, breaking N-algebraic positivity.
  (2) For specific (sigma, lambda, chi, chi') values, the effective
      polynomial DSE has C_3 as the dominant branch.
  (3) The canonical Manna example phi = 1 + 3G^2 - 2G^3 + 0.25G^4 is
      reproduced from action parameters (sigma=3, lambda=1, chi=chi'=1,
      D_rho=1).
  (4) When chi*chi' = 0 (no conservation coupling), a_3 = 0 and the
      system cannot reach non-dyadic strata.
"""
import numpy as np
import pytest

from rdft.ac.signed_projector import (
    effective_DSE_from_CDP_action, scan_action_space_for_C3,
)


class TestSignedCoefficient:
    """The chi-chi' bubble produces a negative a_3 coefficient."""

    def test_a3_negative_for_positive_chi_coupling(self):
        r = effective_DSE_from_CDP_action(sigma=1.0, lambda_dp=1.0,
                                             chi=1.0, chi_prime=1.0,
                                             D_rho=1.0, higher_order=0.25)
        assert r['phi_eff_coefficients'][3] < 0
        assert r['breaks_N_algebraic_positivity']

    def test_a3_zero_when_conservation_absent(self):
        """Setting chi or chi' to zero (no conservation coupling)
        should give a_3 = 0, restoring N-algebraic positivity."""
        r = effective_DSE_from_CDP_action(sigma=1.0, lambda_dp=1.0,
                                             chi=0.0, chi_prime=1.0,
                                             D_rho=1.0, higher_order=0.25)
        assert r['phi_eff_coefficients'][3] == 0.0
        assert not r['breaks_N_algebraic_positivity']


class TestCanonicalC3Reproduction:
    """Reproduce the Manna paper's canonical quartic from action params."""

    def test_canonical_coefficients(self):
        """sigma=3, lambda=1, chi=chi'=1, D_rho=1, higher_order=0.25 gives
        phi_eff = 1 + 3G^2 - 2G^3 + 0.25 G^4."""
        r = effective_DSE_from_CDP_action(sigma=3.0, lambda_dp=1.0,
                                             chi=1.0, chi_prime=1.0,
                                             D_rho=1.0, higher_order=0.25)
        assert r['phi_eff_coefficients'] == [1.0, 0.0, 3.0, -2.0, 0.25]

    def test_canonical_hits_C3(self):
        r = effective_DSE_from_CDP_action(sigma=3.0, lambda_dp=1.0,
                                             chi=1.0, chi_prime=1.0,
                                             D_rho=1.0, higher_order=0.25)
        assert r['on_C3_stratum']
        assert r['k_dom'] == 3
        assert abs(r['tau_0'] - 4.0/3.0) < 1e-10

    def test_canonical_z_star_is_minus_one_quarter(self):
        """The Manna paper reports z_star approx -0.25 for the canonical
        quartic example."""
        r = effective_DSE_from_CDP_action(sigma=3.0, lambda_dp=1.0,
                                             chi=1.0, chi_prime=1.0,
                                             D_rho=1.0, higher_order=0.25)
        assert abs(r['z_star'].real - (-0.25)) < 1e-3


class TestCodimensionOneStratum:
    """C_3 is a codimension-1 stratum in action parameter space."""

    def test_C3_is_measure_zero(self):
        """Scanning a continuous range of sigma should find C_3 only
        at a discrete point (codimension-1)."""
        s = scan_action_space_for_C3(sigma_values=np.linspace(1.0, 5.0, 41))
        # Should find C_3 at exactly one (or at most a few) values.
        # NOT a continuous range.
        assert s['C3_window_count'] <= 3

    def test_C3_found_at_sigma_three(self):
        """The canonical Manna C_3 point is at sigma = 3 (given our
        normalisation)."""
        s = scan_action_space_for_C3(sigma_values=np.linspace(2.8, 3.2, 9))
        # Should find it in the narrow window
        assert s['C3_window_count'] >= 1
        # The sigma value near 3.0
        found_sigmas = [r['sigma'] for r in s['C3_window']]
        assert any(abs(sig - 3.0) < 0.1 for sig in found_sigmas)
