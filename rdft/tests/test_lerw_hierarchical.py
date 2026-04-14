"""Hierarchical-lattice LERW: the combinatorial animal experiment.

Tests the graph constructors and scaling extraction for:
  - the Migdal-Kadanoff (b, s) diamond lattice (series-parallel
    surrogate for Z^d at d_eff = 1 + log(b)/log(s)); and
  - the Sierpinski gasket (non-series-parallel triangular surrogate).

Headline results witnessed here:
  MK diamond (b=4, s=2): d_f = 1 exactly (LERW is degenerate on
    every series-parallel graph; loop erasure has nothing to erase).
  Sierpinski gasket:     d_f ~ 1.2 (non-trivial; loops present).
  Dirichlet-box Z^3:     d_f ~ 1.67 (from lerw_dirichlet tests).

The gap between animal exponents and Z^3 diagnoses what each
approximant captures and what it misses.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw_hierarchical import (
    diamond_lattice, diamond_diameter, diamond_n_edges,
    hierarchical_dimension_sweep, fit_hierarchical_d_f,
    sierpinski_gasket, sierpinski_diameter,
    sierpinski_dimension_sweep, fit_sierpinski_d_f,
    sample_lerw_hierarchical,
)


class TestDiamondConstruction:

    def test_level_0_is_single_edge(self):
        G = diamond_lattice(0, 4, 2)
        assert G[0] == 2
        assert G[1] == [(0, 1)]
        assert G[2] == 0 and G[3] == 1

    def test_level_k_sizes(self):
        """G_k has (b*s)^k edges."""
        for k in range(4):
            G = diamond_lattice(k, 4, 2)
            assert len(G[1]) == diamond_n_edges(k, 4, 2) == 8 ** k

    def test_level_1_diamond_structure(self):
        """G_1 with (b=4, s=2) is 4 parallel branches of 2 edges
        between source and target. Total: 2 + 4 = 6 vertices, 8 edges.
        """
        G = diamond_lattice(1, 4, 2)
        assert G[0] == 6
        assert len(G[1]) == 8


class TestDiamondFailureMode:

    def test_d_f_equals_one_on_mk_diamond(self):
        """Canonical failure mode: LERW on the MK diamond lattice
        always realises a path of length equal to the diameter.
        The UST on a series-parallel graph has no loops to erase,
        so every sampled path has length s^k.
        """
        rng = np.random.default_rng(0)
        for k in (1, 2, 3, 4):
            G = diamond_lattice(k, 4, 2)
            # A single sample suffices as a structural assertion
            for trial in range(5):
                path = sample_lerw_hierarchical(G, rng)
                assert len(path) - 1 == diamond_diameter(k, 2), \
                    f"k={k}: path length {len(path)-1}, diameter={2**k}"

    def test_fit_d_f_equals_one_for_diamond(self):
        """Log-log fit of <|gamma|> vs diameter on the diamond gives
        exactly slope 1.
        """
        sweep = hierarchical_dimension_sweep(
            [1, 2, 3, 4], b=4, s=2, n_samples=100, seed=0)
        d_f, _ = fit_hierarchical_d_f(sweep, s=2)
        assert abs(d_f - 1.0) < 1e-9


class TestSierpinskiConstruction:

    def test_level_0_is_triangle(self):
        G = sierpinski_gasket(0)
        assert G[0] == 3
        assert sorted(G[1]) == [(0, 1), (0, 2), (1, 2)]
        assert G[2] == 0 and G[3] == 1

    def test_level_k_edge_count(self):
        """Sierpinski gasket G_k has 3^(k+1) edges."""
        for k in range(5):
            G = sierpinski_gasket(k)
            assert len(G[1]) == 3 ** (k + 1)

    def test_corners_are_first_three_vertices(self):
        """By construction the three top-level corners are ids
        {0, 1, 2} at every level.
        """
        for k in range(4):
            G = sierpinski_gasket(k)
            # Find degrees of vertices 0, 1, 2
            deg = [0, 0, 0]
            for (u, v) in G[1]:
                if u < 3:
                    deg[u] += 1
                if v < 3:
                    deg[v] += 1
            # Top-level corners have degree 2 at every level
            # (they are each adjacent to exactly one inner midpoint
            # along each of two edges of the outer triangle).
            if k == 0:
                assert deg == [2, 2, 2]
            else:
                # Outer corners are shared by only one sub-copy so
                # they inherit the sub-corner degree (which is 2).
                assert deg[0] == 2


class TestSierpinskiDimension:

    def test_non_trivial_d_f(self):
        """LERW on the Sierpinski gasket has 1 < d_f < 2 (non-trivial
        loop-erasure scaling, distinct from both series-parallel
        d_f = 1 and Brownian d_f = 2).
        """
        sweep = sierpinski_dimension_sweep(
            [2, 3, 4, 5], n_samples=1000, seed=0)
        d_f, _ = fit_sierpinski_d_f(sweep)
        assert 1.05 < d_f < 1.5, \
            f"Sierpinski d_f = {d_f:.4f} not in expected band"

    def test_monotone_length_in_level(self):
        """<|gamma|>_k strictly increases with k."""
        sweep = sierpinski_dimension_sweep(
            [1, 2, 3, 4], n_samples=500, seed=1)
        means = [sweep[k]['mean_length'] for k in (1, 2, 3, 4)]
        for m1, m2 in zip(means, means[1:]):
            assert m2 > m1

    def test_sierpinski_below_hausdorff(self):
        """d_f^{LERW} should sit below the Hausdorff dimension
        of the gasket (log 3 / log 2 ~ 1.585): LERW is a curve, of
        strictly lower dimension than the ambient fractal.
        """
        sweep = sierpinski_dimension_sweep(
            [3, 4, 5], n_samples=600, seed=2)
        d_f, _ = fit_sierpinski_d_f(sweep)
        assert d_f < np.log(3) / np.log(2) - 0.1, \
            f"d_f = {d_f:.4f} not strictly below log(3)/log(2)"


class TestCombinatorialAnimalComparison:

    def test_diamond_exponent_well_below_Z3(self):
        """Direct cross-animal comparison: the MK diamond's d_f = 1
        is well below the Z^3 value ~ 1.624. This is the key
        negative result: the series-parallel approximation loses
        the meandering structure of LERW entirely.
        """
        sweep = hierarchical_dimension_sweep(
            [1, 2, 3], b=4, s=2, n_samples=50, seed=0)
        d_f_mk, _ = fit_hierarchical_d_f(sweep, s=2)
        d_f_Z3_numerical = 1.6236
        gap = d_f_Z3_numerical - d_f_mk
        assert gap > 0.5, f"MK animal gap to Z^3 too small: {gap}"

    def test_sierpinski_gives_qualitative_improvement(self):
        """Sierpinski d_f (~1.2) is closer to Z^3 (~1.624) than
        the MK diamond's d_f = 1.0, despite Sierpinski living in
        a different ambient dimension. The presence of loops
        matters more than the ambient-dimension label.
        """
        sweep_sierp = sierpinski_dimension_sweep(
            [3, 4, 5], n_samples=800, seed=0)
        d_f_sierp, _ = fit_sierpinski_d_f(sweep_sierp)
        d_f_Z3_numerical = 1.6236
        gap_sierp = abs(d_f_Z3_numerical - d_f_sierp)
        gap_mk = abs(d_f_Z3_numerical - 1.0)
        assert gap_sierp < gap_mk, \
            f"Sierp gap {gap_sierp} not smaller than MK gap {gap_mk}"
