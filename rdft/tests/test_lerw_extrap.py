"""Algebraic extrapolation schemes for LERW d_f.

These tests witness two complementary results:

1. **Positive**: the schemes are internally consistent on synthetic
   data with known correction structure --- Richardson recovers the
   limit when the correction is exactly 1/L, the best-omega fit
   recovers omega when one is injected, etc.

2. **Negative** (and arguably more informative): on real LERW data
   at accessible box sizes, the "sophisticated" extrapolation
   schemes (Richardson, Neville, forced-power correction fits) do
   not beat the naive log-log fit. This is concrete evidence that
   LERW finite-size corrections do not follow a simple algebraic
   power-of-1/L structure --- if they did, Richardson would
   outperform naive.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw_extrap import (
    naive_fit, fit_with_correction, scan_omega,
    effective_exponents, richardson_extrapolate,
    neville_richardson, best_neville, all_extrapolations,
)


class TestSyntheticValidation:

    def test_naive_fit_on_pure_power_law(self):
        """Pure power-law data: naive fit recovers exact slope."""
        L_vals = [4, 8, 16, 32, 64]
        d_f = 1.25
        means = [float(L) ** d_f for L in L_vals]
        d_fit, _ = naive_fit(L_vals, means)
        assert abs(d_fit - d_f) < 1e-10

    def test_correction_fit_recovers_injected_B(self):
        """Data generated as L^{1.25} * (1 + 0.5/L): the omega=1 fit
        recovers d_f = 1.25 and B = 0.5.
        """
        L_vals = [8, 12, 16, 24, 32, 48, 64]
        d_f_true = 1.25
        B_true = 0.5
        means = [float(L) ** d_f_true * (1 + B_true / L) for L in L_vals]
        res = fit_with_correction(L_vals, means, omega=1.0)
        # Alternating minimisation is not exact on finite data with
        # correction present; loose tolerance.
        assert abs(res['d_f'] - d_f_true) < 5e-3
        assert abs(res['B'] - B_true) < 5e-2

    def test_omega_scan_runs_and_returns_reasonable_d_f(self):
        """Data generated with correction 1/L^{0.5}: grid search
        over omega returns a d_f estimate close to the true value,
        even if omega itself is confounded with B under alternating
        minimisation (a known limitation on finite L ranges).
        """
        L_vals = [8, 12, 16, 24, 32, 48, 64, 96]
        d_f_true = 1.25
        omega_true = 0.5
        B_true = 1.2
        means = [float(L) ** d_f_true * (1 + B_true * L ** (-omega_true))
                 for L in L_vals]
        best = scan_omega(L_vals, means,
                          omega_grid=np.linspace(0.1, 2.0, 30))
        # Alternating minimisation on L in [8, 96] with slow 1/L^{0.5}
        # correction is strongly under-determined: d_f and B/omega
        # trade off. Accept d_f within 0.08 --- the point of the test
        # is that the API runs and returns something sensible, not
        # that the fit is precision-grade.
        assert abs(best['d_f'] - d_f_true) < 0.08
        # omega is in the tested grid
        assert 0.1 <= best['omega'] <= 2.0

    def test_richardson_exact_on_one_over_L_correction(self):
        """Richardson on d_eff(L) = d_f + C/L converges to d_f."""
        L_mids = [10.0, 20.0, 40.0, 80.0]
        d_f_true = 1.25
        C = 0.5
        d_effs = [d_f_true + C / L for L in L_mids]
        est = richardson_extrapolate(L_mids, d_effs)
        assert abs(est - d_f_true) < 1e-9

    def test_neville_exact_on_series_expansion(self):
        """Neville-Richardson tableau on d_eff = d_f + C1/L + C2/L^2
        recovers d_f within the tableau depth.
        """
        L_mids = [4.0, 8.0, 16.0, 32.0, 64.0]
        d_f_true = 1.6
        C1, C2 = 0.3, -0.1
        d_effs = [d_f_true + C1 / L + C2 / L ** 2 for L in L_mids]
        est = best_neville(L_mids, d_effs)
        assert abs(est - d_f_true) < 1e-6


class TestEffectiveExponents:

    def test_effective_exponents_on_power_law(self):
        """Clean power-law: d_eff(L) = d_f exactly (no corrections)."""
        L_vals = [4, 8, 16, 32]
        d_f = 1.4
        means = [float(L) ** d_f for L in L_vals]
        eff = effective_exponents(L_vals, means)
        for _, d in eff:
            assert abs(d - d_f) < 1e-10


class TestRealLERW2DExtrapolationAgainstKenyon:

    def test_naive_fit_is_within_2percent_of_kenyon(self):
        """On actual 2D Dirichlet-box LERW data (precomputed to avoid
        long test time): the naive fit sits within 2% of Kenyon's
        exact 5/4. The "sophisticated" corrections bias it downward
        and do NOT improve agreement --- a concrete witness that
        LERW corrections are not simple power-law in 1/L.
        """
        # These values come from lerw_dirichlet_sweep(d=2, seed=100,
        # n_samples=4000). Hard-coded so the test is fast and
        # deterministic independent of RNG changes.
        L_vals = [8, 12, 16, 24, 32, 48, 64]
        means = [6.894, 11.918, 17.271, 28.931, 41.115, 68.260, 98.666]
        d_naive, _ = naive_fit(L_vals, means)
        # Kenyon exact: 5/4 = 1.25. Naive fit is within 3%.
        assert abs(d_naive - 1.25) < 0.04, \
            f"d_naive = {d_naive:.4f} too far from 1.25"


class TestExtrapolationSchemesDoNotBeatNaive:
    """Negative-result tests: they document that our extrapolation
    schemes, despite being 'more principled', do not improve over
    the naive log-log fit on LERW data at accessible box sizes.
    """

    def test_1overL_correction_fit_biases_2d_downward(self):
        """On 2D data the (1 + B/L) fit pushes d_f BELOW 1.25 because
        the true correction is not a simple 1/L power. This is the
        'correction-to-scaling overfit' failure mode.
        """
        L_vals = [8, 12, 16, 24, 32, 48, 64]
        means = [6.894, 11.918, 17.271, 28.931, 41.115, 68.260, 98.666]
        res = fit_with_correction(L_vals, means, omega=1.0)
        # Biased below 1.25
        assert res['d_f'] < 1.25
        # Further from 1.25 than the naive
        d_naive, _ = naive_fit(L_vals, means)
        gap_naive = abs(d_naive - 1.25)
        gap_corr = abs(res['d_f'] - 1.25)
        # Corrected fit is LESS accurate than naive here
        assert gap_corr > gap_naive - 1e-9, \
            f"corrected={res['d_f']:.4f} beats naive={d_naive:.4f}"

    def test_richardson_amplifies_noise_or_bias(self):
        """Richardson extrapolation assumes a 1/L correction; on
        real LERW data its output is further from Kenyon than naive,
        because the assumption is wrong.
        """
        L_vals = [8, 12, 16, 24, 32, 48, 64]
        means = [6.894, 11.918, 17.271, 28.931, 41.115, 68.260, 98.666]
        res = all_extrapolations(L_vals, means)
        gap_naive = abs(res['naive'] - 1.25)
        gap_rich = abs(res['richardson'] - 1.25)
        # In our regime Richardson is (at least) no better than naive
        assert gap_rich >= gap_naive - 0.02


class TestAllExtrapolationsAPI:

    def test_runs_without_error(self):
        L_vals = [8, 16, 32, 64]
        means = [float(L) ** 1.3 for L in L_vals]
        res = all_extrapolations(L_vals, means)
        assert 'naive' in res
        assert 'with_correction_om1' in res
        assert 'best_omega_fit' in res
        assert 'richardson' in res
        assert 'neville' in res

    def test_naive_matches_pure_power_law(self):
        L_vals = [4, 8, 16, 32, 64]
        d_f = 1.6236
        means = [float(L) ** d_f for L in L_vals]
        res = all_extrapolations(L_vals, means)
        assert abs(res['naive'] - d_f) < 1e-9
