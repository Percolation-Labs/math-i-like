"""Tests for rdft.ac.admissible — transcendental/Hayman DSE extension.

Verifies:
  (1) Poisson-offspring CRN at criticality gives the Flajolet-Sedgewick
      canonical asymptotic [z^n] G ~ n^{-3/2}/sqrt(2 pi).
  (2) Generic admissible phi lands on C_2 (tau = 3/2).
  (3) Stable-tree widening gives non-dyadic tau for alpha in (1, 2].
  (4) Subcritical/supercritical regimes are correctly identified.
  (5) Asymptotic agrees numerically with exact enumeration at finite n.
"""
import numpy as np
import pytest

from rdft.ac.admissible import (
    find_critical_point, admissible_asymptotics, coefficient_asymptotic,
    stable_tree_tau, poisson_branching_demo, cosh_offspring_demo,
)


class TestPoissonBranching:
    """Critical Poisson-offspring branching is the canonical test case."""

    def test_critical_fixed_point(self):
        r = poisson_branching_demo(lambda_offspring=1.0)
        assert r['critical_point_found']
        assert abs(r['G_star'] - 1.0) < 1e-8
        assert abs(r['z_star'] - 1.0) < 1e-8

    def test_critical_amplitude_is_FS_value(self):
        """At criticality, the amplitude of n^{-3/2} should be 1/sqrt(2 pi)
        (Flajolet-Sedgewick Example I.5.13, Proposition III.6)."""
        r = poisson_branching_demo(lambda_offspring=1.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(r['amplitude_coef_asymptotic'] - expected) < 1e-10

    def test_tau_is_three_halves(self):
        r = poisson_branching_demo(lambda_offspring=1.0)
        assert r['tau'] == 1.5
        assert r['stratum'] == 'C_2'

    def test_subcritical_has_zstar_above_one(self):
        r = poisson_branching_demo(lambda_offspring=0.5)
        assert r['z_star'] > 1.0
        assert r['regime'] == 'subcritical'

    def test_supercritical_regime_detected(self):
        """For supercritical branching (lambda > 1) the DSE for total
        progeny still has z* > 1 because extinction probability < 1;
        we only check regime labelling, not z* < 1 (that would be a
        different generating function)."""
        r = poisson_branching_demo(lambda_offspring=1.5)
        assert r['regime'] == 'supercritical'
        assert r['z_star'] > 0.5  # finite and positive


class TestCoshOffspring:
    """cosh admissible kernel — different entire function, same stratum."""

    def test_admissible(self):
        r = cosh_offspring_demo()
        assert r['admissible']

    def test_still_lands_on_C2(self):
        """Drmota-Lalley-Woods: any admissible phi with finite variance
        gives tau = 3/2 regardless of the specific entire function."""
        r = cosh_offspring_demo()
        assert r['tau'] == 1.5
        assert r['stratum'] == 'C_2'

    def test_critical_point_values(self):
        """G*·sinh(G*) = cosh(G*) gives G* ~ 1.1997."""
        r = cosh_offspring_demo()
        assert abs(r['G_star'] - 1.1997) < 1e-3


class TestStableTreeWidening:
    """Admissible extension includes stable trees with alpha < 2."""

    def test_alpha_equals_2_gives_3_halves(self):
        # Finite variance recovers DLW
        assert abs(stable_tree_tau(2.0) - 1.5) < 1e-12

    def test_alpha_3_halves_gives_5_thirds(self):
        # Stable index 3/2 -> tau = 1 + 2/3 = 5/3
        assert abs(stable_tree_tau(1.5) - 5/3) < 1e-12

    def test_non_dyadic_tau_reachable(self):
        """The whole point: admissible+heavy-tail reaches non-dyadic tau."""
        tau = stable_tree_tau(1.3)
        # Not an element of the dyadic ladder {3/2, 4/3, 5/4, ...}
        dyadic_values = {1 + 1/k for k in [2, 3, 4, 5, 6]}
        assert min(abs(tau - d) for d in dyadic_values) > 0.03

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            stable_tree_tau(0.5)
        with pytest.raises(ValueError):
            stable_tree_tau(2.5)


class TestAsymptoticAccuracyAtFiniteN:
    """The asymptotic formula should agree with exact enumeration at
    moderate n.  For critical Poisson branching with lambda=1, the
    exact coefficient is [z^n] G(z) = n^{n-1}/n! by Lagrange inversion
    on G = z exp(G-1), which we can compute for small n and compare
    against n^{-3/2}/sqrt(2 pi)."""

    def test_asymptotic_matches_exact_at_large_n(self):
        """For critical Poisson branching with lambda=1, the exact
        coefficient c_n = e^{-n} n^{n-1} / n! (Lagrange inversion on
        G = z e^{G-1}).  By Stirling this equals n^{-3/2}/sqrt(2 pi)
        asymptotically.  We verify the asymptotic formula agrees with
        the exact c_n to ~1/n relative precision at moderate n."""
        from scipy.special import gammaln
        r = poisson_branching_demo(lambda_offspring=1.0)
        for n in [100, 1000, 10000]:
            # Exact in log space: log c_n = -n + (n-1)*log(n) - log(n!)
            log_exact = -n + (n - 1) * np.log(n) - gammaln(n + 1)
            exact_c_n = np.exp(log_exact)
            asymp = coefficient_asymptotic(n, r)
            rel_err = abs(asymp - exact_c_n) / exact_c_n
            # Leading correction to Stirling is 1/(12n); tolerate 2/n.
            assert rel_err < 2.0 / n + 1e-4, (
                f'n={n}: asymptotic {asymp:.6e} vs exact {exact_c_n:.6e}, '
                f'rel err {rel_err:.4f}')
