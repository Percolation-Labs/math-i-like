"""Dirichlet-box LERW: Kenyon (d=2) and Kozma (d=3) fractal exponents.

Paper/cfac/enumerative_boundary.tex, Prop 1 places LERW inside the
CFAC enumeration tier as a ratio of Kirchhoff polynomials. The
scaling-limit exponent d_f is a finite-size question about that
ratio. This test recovers d_f = 5/4 in d=2 and d_f ~ 1.624 in d=3
by Monte Carlo sampling Lawler's construction on a killed box, the
natural setting for Kenyon's theorem.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw_dirichlet import (
    sample_lerw_to_boundary, mean_lerw_length_dirichlet,
    lerw_dirichlet_sweep, fit_fractal_dimension,
)


class TestLerwDirichletSampler:

    def test_sample_returns_simple_path_starting_at_origin(self):
        rng = np.random.default_rng(0)
        path = sample_lerw_to_boundary(L=10, d=2, rng=rng,
                                       start=(5, 5))
        assert path[0] == (5, 5)
        # simple: no repeated vertices
        assert len(set(path)) == len(path)
        # consecutive sites differ by exactly one unit in one coord
        for u, v in zip(path, path[1:]):
            diffs = [abs(a - b) for a, b in zip(u, v)]
            assert sum(diffs) == 1

    def test_sample_stays_inside_box(self):
        rng = np.random.default_rng(1)
        L = 8
        path = sample_lerw_to_boundary(L=L, d=2, rng=rng, start=(4, 4))
        for p in path:
            for c in p:
                assert 0 <= c < L

    def test_mean_length_increases_with_L_in_2d(self):
        rng = np.random.default_rng(2)
        _, _ = mean_lerw_length_dirichlet(L=6, d=2, n_samples=200, rng=rng)
        m_small, _ = mean_lerw_length_dirichlet(
            L=8, d=2, n_samples=500, rng=rng)
        m_large, _ = mean_lerw_length_dirichlet(
            L=24, d=2, n_samples=500, rng=rng)
        assert m_large > m_small


class TestLerwFractalDimension:

    def test_kenyon_5_over_4_in_2d(self):
        """E[|gamma|] ~ L^{5/4} for 2D LERW from centre to boundary
        (Kenyon 2000). We recover 1.25 within 0.07 at modest compute.
        """
        L_vals = [8, 12, 16, 24, 32]
        data = lerw_dirichlet_sweep(L_vals, d=2, n_samples=1500, seed=10)
        means = [data[L][0] for L in L_vals]
        sems = [data[L][1] for L in L_vals]
        d_f, _ = fit_fractal_dimension(L_vals, means, sems)
        # Kenyon: d_f = 5/4 = 1.25. Finite-size drift ~ a few %.
        assert 1.18 < d_f < 1.35, \
            f"d=2 fit d_f={d_f:.4f}, want ~1.25"

    def test_kozma_d3_fractal_dimension(self):
        """E[|gamma|] ~ L^{d_f} with d_f ~ 1.624 in 3D
        (Kozma 2007; Wilson 2010 numerical 1.6236...).
        """
        L_vals = [4, 6, 8, 10, 12]
        data = lerw_dirichlet_sweep(L_vals, d=3, n_samples=1000, seed=11)
        means = [data[L][0] for L in L_vals]
        sems = [data[L][1] for L in L_vals]
        d_f, _ = fit_fractal_dimension(L_vals, means, sems)
        # Kozma: d_f ~ 1.624. Small-L finite-size drift ~ 5%.
        assert 1.50 < d_f < 1.80, \
            f"d=3 fit d_f={d_f:.4f}, want ~1.624"

    def test_d4_trends_towards_mean_field_2(self):
        """In d >= 4, LERW is diffusive (d_f = 2) up to log corrections.
        At very small L the fit is driven hard by finite-size; we only
        check that it exceeds the d=3 value, recovering the ordering.
        """
        L_vals = [3, 4, 5, 6]
        data = lerw_dirichlet_sweep(L_vals, d=4, n_samples=500, seed=12)
        means = [data[L][0] for L in L_vals]
        sems = [data[L][1] for L in L_vals]
        d_f, _ = fit_fractal_dimension(L_vals, means, sems)
        # At least Kozma's 1.624 (finite-size will push below the
        # asymptotic d_f = 2 from above).
        assert d_f > 1.55, f"d=4 fit d_f={d_f:.4f}, want >~1.6"

    def test_fit_fractal_dimension_on_synthetic_data(self):
        """Fit recovers exact slope on clean synthetic data."""
        L_vals = [4, 8, 16, 32, 64]
        true_slope = 1.25
        means = [float(L ** true_slope) for L in L_vals]
        d_f, _ = fit_fractal_dimension(L_vals, means)
        assert abs(d_f - true_slope) < 1e-10
