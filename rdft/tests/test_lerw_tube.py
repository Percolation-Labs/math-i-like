"""Tube LERW: documented negative result.

Tests verify:
  1. The tube sampler returns simple paths that exit through the
     z-axis end caps (never through the periodic cross-section).
  2. The path length for any fixed cross-section size N scales
     ballistically (d_f^{(N)} ~ 1) as L grows --- the quasi-1D
     nature of the tube prevents d_f^{(3)} from being probed.

These tests document that the tube-transfer-matrix tower of
algebraic animals is the wrong tower for the Z^3 LERW fractal
dimension.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw_tube import sample_lerw_tube, tube_scaling_sweep, fit_tube_d_f


class TestTubeSampler:

    def test_sample_is_simple_path(self):
        rng = np.random.default_rng(0)
        path = sample_lerw_tube(N=3, L=8, rng=rng)
        assert len(set(path)) == len(path)
        for u, v in zip(path, path[1:]):
            # Exactly one coordinate differs, by 1 (accounting for
            # periodic wrap on the first two axes).
            diffs = []
            for a, b, i in zip(u, v, range(3)):
                if i < 2:
                    # periodic distance on NxN
                    diffs.append(min(abs(a - b), 3 - abs(a - b)))
                else:
                    diffs.append(abs(a - b))
            assert sum(diffs) == 1

    def test_exits_only_through_z_ends(self):
        rng = np.random.default_rng(1)
        N = 3
        L = 8
        for _ in range(10):
            path = sample_lerw_tube(N=N, L=L, rng=rng)
            # x and y coordinates always in [0, N-1]
            for p in path:
                assert 0 <= p[0] < N
                assert 0 <= p[1] < N
                assert 0 <= p[2] < L


class TestTubeBallisticCollapse:

    def test_d_f_is_one_for_every_cross_section(self):
        """For every N the LERW path length on the tube is
        ballistic in L (d_f^{(N)} ~ 1 within Monte Carlo noise).
        This is the tower-of-tubes negative result.
        """
        rng_seed = 0
        L_vals = [8, 12, 16, 24]
        for N in (1, 2, 3, 4):
            sweep = tube_scaling_sweep(
                N=N, L_vals=L_vals, n_samples=150, seed=rng_seed)
            d_f_tube = fit_tube_d_f(sweep)
            assert 0.8 < d_f_tube < 1.2, \
                f"N={N}: d_f_tube={d_f_tube:.3f} outside ballistic window"
