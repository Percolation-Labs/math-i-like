"""Tests for rdft.ac.log_corrections — algebraic-log singularity transfer.

Verifies:
  (1) Pure algebraic case (beta=0) reduces to standard Flajolet-Odlyzko.
  (2) Log-corrected transfer matches textbook sqrt(1-z)·log(1/(1-z)) case.
  (3) Leading log factor scales correctly.
  (4) Potts q=4 schematic produces a visible log correction.
"""
import numpy as np
import pytest

from rdft.ac.log_corrections import (
    transfer_theorem_log_corrected, known_case_log_half_asymptotic,
    potts_q4_marginal_demo,
)


class TestPureAlgebraic:
    """Log-corrected formula with beta=0 should reproduce standard
    transfer theorem for pure algebraic branches."""

    def test_alpha_half_beta_zero(self):
        """[z^n] sqrt(1-z) ~ -1/(2 sqrt pi) * n^{-3/2}"""
        n = 1000
        result = transfer_theorem_log_corrected(0.5, 0, 1.0, 1.0, n)
        # Standard: n^{-3/2} / Gamma(-1/2) = n^{-3/2} / (-2 sqrt pi)
        expected = -1.0 / (2 * np.sqrt(np.pi)) * n**(-1.5)
        assert abs(result - expected) / abs(expected) < 1e-10

    def test_alpha_third_beta_zero(self):
        """[z^n] (1-z)^{1/3} ~ 1/Gamma(-1/3) * n^{-4/3} (C_3 stratum)"""
        from scipy.special import gamma
        n = 1000
        result = transfer_theorem_log_corrected(1.0/3.0, 0, 1.0, 1.0, n)
        expected = (1.0 / gamma(-1.0/3.0)) * n**(-4.0/3.0)
        assert abs(result - expected) / abs(expected) < 1e-10


class TestLogCorrectedCase:
    """sqrt(1-z) * log(1/(1-z)) — the classic log-corrected singularity
    whose textbook asymptotic is [z^n] ~ -1/(2 sqrt pi) * n^{-3/2}
    * (log n + euler_gamma - 2)."""

    def test_leading_log_agrees_at_large_n(self):
        """Leading log behaviour: (result / textbook) * (log n + c) / log n -> -1."""
        euler_gamma = 0.5772156649015329
        for n in [10_000, 100_000]:
            transfer = transfer_theorem_log_corrected(0.5, 1, 1.0, 1.0, n)
            textbook = known_case_log_half_asymptotic(n)
            # transfer_thm: result = n^{-3/2} log n / Gamma(-1/2) = -log n / (2 sqrt pi) n^{-3/2}
            # textbook:    -1/(2 sqrt pi) * n^{-3/2} * (log n + euler_gamma - 2)
            # Ratio:        log n / (log n + euler_gamma - 2)  -> 1 as n -> inf
            ratio = transfer / textbook
            theoretical_ratio = np.log(n) / (np.log(n) + euler_gamma - 2)
            assert abs(ratio - theoretical_ratio) < 0.01, (
                f'n={n}: ratio {ratio:.4f} vs theoretical {theoretical_ratio:.4f}')

    def test_log_factor_ratio_scales_as_log_n(self):
        """With log vs without log: ratio should be log(n) exactly."""
        for n in [100, 1000, 10000]:
            with_log = transfer_theorem_log_corrected(0.5, 1, 1.0, 1.0, n)
            without = transfer_theorem_log_corrected(0.5, 0, 1.0, 1.0, n)
            ratio = with_log / without
            assert abs(ratio - np.log(n)) < 1e-10


class TestHigherLogPowers:
    """beta = 2 gives (log n)^2 factor, etc."""

    def test_beta_two(self):
        n = 1000
        r1 = transfer_theorem_log_corrected(0.5, 1, 1.0, 1.0, n)
        r2 = transfer_theorem_log_corrected(0.5, 2, 1.0, 1.0, n)
        # r2 / r1 = log n
        assert abs(r2 / r1 - np.log(n)) < 1e-10


class TestInvalidInputs:
    def test_negative_integer_alpha_raises(self):
        # alpha non-negative integer is undefined branch (Gamma pole)
        with pytest.raises(ValueError):
            transfer_theorem_log_corrected(0, 1, 1.0, 1.0, 100)


class TestPottsDemo:
    """Potts q=4 schematic produces correct scaling form."""

    def test_demo_runs(self):
        r = potts_q4_marginal_demo()
        assert 'asymptotics' in r
        assert len(r['asymptotics']) == 4

    def test_log_factor_grows_with_n(self):
        r = potts_q4_marginal_demo()
        log_factors = [a['log_factor'] for a in r['asymptotics']]
        # Should be increasing (log grows)
        assert all(log_factors[i+1] > log_factors[i]
                   for i in range(len(log_factors) - 1))
