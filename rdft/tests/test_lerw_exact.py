"""Deterministic Kirchhoff-ratio LERW mean length.

Tests that the exact Prop-1 calculation on small Z^d_L boxes
reproduces the deterministic values computed once and cached in
rdft.ac.lerw_exact.DETERMINISTIC_VALUES, and that Monte Carlo
converges to them.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw_exact import (
    cubic_box_graph, exact_corner_to_corner_mean,
    DETERMINISTIC_VALUES, corner_distance,
)
from rdft.ac.lerw_hierarchical import sample_lerw_hierarchical


class TestCubicBoxGraph:

    def test_shape(self):
        n, edges = cubic_box_graph(3, 2)
        assert n == 9
        # 3x3 grid has 12 edges (2 per interior vertex, boundary contributes less)
        assert len(edges) == 12

    def test_3d_box(self):
        n, edges = cubic_box_graph(3, 3)
        assert n == 27
        # 3^3 = 27 vertices; 3 * 3^2 * 2 = 54 edges
        assert len(edges) == 54


class TestDeterministicSmallCases:

    def test_z2_L3_corner(self):
        """Z^2 3x3 corner-to-corner exact mean = 17/4."""
        m = exact_corner_to_corner_mean(3, 2)
        expected = DETERMINISTIC_VALUES[(3, 2)]
        assert abs(m - expected) < 1e-10

    def test_z2_L4_corner(self):
        m = exact_corner_to_corner_mean(4, 2)
        # Determinant ratios accumulate a few ulps; 1e-7 is still
        # many orders tighter than any Monte-Carlo sampler.
        assert abs(m - DETERMINISTIC_VALUES[(4, 2)]) < 1e-7

    def test_z3_L2_corner(self):
        """Z^3 2x2x2 corner-to-corner exact = 53/16."""
        m = exact_corner_to_corner_mean(2, 3)
        assert abs(m - DETERMINISTIC_VALUES[(2, 3)]) < 1e-10


class TestMonteCarloAgrees:

    def test_mc_agrees_z2_L3(self):
        """Monte Carlo via Wilson matches the exact Kirchhoff value
        within statistical noise. This validates both the exact
        enumeration and the Wilson sampler against each other.
        """
        L, d = 3, 2
        G = cubic_box_graph(L, d)
        edges_t = [(u, v) for (u, v, w) in G[1]]
        n = L ** d
        wrapped = (n, edges_t, 0, n - 1)
        rng = np.random.default_rng(0)
        lens = [len(sample_lerw_hierarchical(wrapped, rng)) - 1
                for _ in range(4000)]
        m_mc = float(np.mean(lens))
        sem = float(np.std(lens, ddof=1) / np.sqrt(len(lens)))
        m_exact = DETERMINISTIC_VALUES[(L, d)]
        # 4 sigma
        assert abs(m_mc - m_exact) < 4 * sem, \
            f"MC {m_mc:.4f} +/- {sem:.4f} vs exact {m_exact}"

    def test_mc_agrees_z3_L2(self):
        L, d = 2, 3
        G = cubic_box_graph(L, d)
        edges_t = [(u, v) for (u, v, w) in G[1]]
        n = L ** d
        wrapped = (n, edges_t, 0, n - 1)
        rng = np.random.default_rng(1)
        lens = [len(sample_lerw_hierarchical(wrapped, rng)) - 1
                for _ in range(4000)]
        m_mc = float(np.mean(lens))
        sem = float(np.std(lens, ddof=1) / np.sqrt(len(lens)))
        m_exact = DETERMINISTIC_VALUES[(L, d)]
        assert abs(m_mc - m_exact) < 4 * sem


class TestCornerDistanceFormula:

    def test_corner_distance(self):
        for L in (2, 3, 4, 5):
            for d in (1, 2, 3):
                assert corner_distance(L, d) == d * (L - 1)
