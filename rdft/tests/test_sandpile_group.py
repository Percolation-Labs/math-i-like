"""Sandpile-group animal for LERW: a documented negative result.

Tests witness:

1. The sandpile-group order of Z^d_L torus matches the Matrix-Tree
   spanning-tree count (verification of Kirchhoff content).

2. Specific invariant-factor structures at small L match
   hand-verified values (regression).

3. The STRUCTURAL content of the group (elementary divisors,
   largest factor) does NOT scale as a universal power of L that
   tracks d_f^{(LERW)} --- the group algebra is the wrong observable
   for the fractal-dimension question.

This module documents the fifth failure mode in the combinatorial-
animal catalogue of paper/cfac/enumerative_boundary.tex.
"""

from __future__ import annotations

from rdft.ac.sandpile_group import (
    torus_reduced_laplacian,
    sandpile_invariant_factors,
    sandpile_group_stats,
)


class TestSandpileGroupBasics:

    def test_reduced_laplacian_is_symmetric(self):
        for L in (3, 4, 5):
            A = torus_reduced_laplacian(L, 2)
            assert (A == A.T).all()

    def test_reduced_laplacian_shape(self):
        for L in (3, 4, 5):
            A = torus_reduced_laplacian(L, 2)
            assert A.shape == (L * L - 1, L * L - 1)
        for L in (3,):
            A = torus_reduced_laplacian(L, 3)
            assert A.shape == (L ** 3 - 1, L ** 3 - 1)

    def test_row_sums_of_full_laplacian_are_zero(self):
        """Full Laplacian (before removing sink) has row sum 0.
        The reduced Laplacian's row sums are the columns that were
        removed, so each row sum equals the number of edges from
        that vertex to the sink (vertex 0).
        """
        for L in (3, 4):
            A_full_size = L * L
            full = _full_torus_laplacian(L, 2)
            assert (full.sum(axis=1) == 0).all()


def _full_torus_laplacian(L, d):
    import numpy as np
    n = L ** d
    strides = [L ** i for i in range(d)]
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        c = tuple((i // s) % L for s in strides)
        A[i, i] = 2 * d
        for axis in range(d):
            for shift in (-1, +1):
                c2 = list(c)
                c2[axis] = (c2[axis] + shift) % L
                j = sum(ci * si for ci, si in zip(c2, strides))
                A[i, j] -= 1
    return A


class TestSandpileGroupOrder:

    def test_matches_matrix_tree_small_cases(self):
        """Group order = product of non-trivial invariant factors =
        number of spanning trees. Test small-L closed forms:
        Z^2_L torus has tau = (1/L^2) * prod_{(j,k) != 0}
            [4 - 2 cos(2 pi j/L) - 2 cos(2 pi k/L)].
        """
        import numpy as np
        for L in (3, 4, 5):
            stats = sandpile_group_stats(L, 2)
            # Independent Matrix-Tree check
            evals = []
            for j in range(L):
                for k in range(L):
                    lam = 4 - 2 * np.cos(2 * np.pi * j / L) \
                        - 2 * np.cos(2 * np.pi * k / L)
                    if (j, k) != (0, 0):
                        evals.append(lam)
            tau_eigenvalue = round(float(np.prod(evals)) / (L * L))
            # Wait: the Matrix-Tree formula for a torus gives
            #   tau = (1/L^d) * prod_{k != 0} lambda_k
            # because the Laplacian has nullspace dim 1.
            assert stats['group_order'] == tau_eigenvalue, \
                f"L={L}: group_order={stats['group_order']} vs " \
                f"eigenvalue prod={tau_eigenvalue}"


class TestDocumentedStructures:

    def test_z2_L3_known_divisors(self):
        """Z^2_3 torus sandpile group has non-trivial divisors
        [6, 6, 18, 18] (hand-computed / reproducible).
        """
        stats = sandpile_group_stats(3, 2)
        assert stats['non_trivial_factors'] == [6, 6, 18, 18]
        assert stats['largest_factor'] == 18

    def test_z2_L4_known_divisors(self):
        stats = sandpile_group_stats(4, 2)
        assert stats['non_trivial_factors'] == [2, 2, 8, 24, 24, 24, 96]
        assert stats['largest_factor'] == 96

    def test_z2_L5_known_divisors(self):
        stats = sandpile_group_stats(5, 2)
        assert stats['non_trivial_factors'] == [10, 10, 50, 50, 50, 50, 50, 50]
        assert stats['largest_factor'] == 50

    def test_z3_L3_known_divisors(self):
        """Z^3_3 torus sandpile group has non-trivial divisors
        [6, 6, 6, 18, 18, 54, 54, 54, 54, 162, 162, 162].
        """
        stats = sandpile_group_stats(3, 3)
        assert stats['non_trivial_factors'] == \
            [6, 6, 6, 18, 18, 54, 54, 54, 54, 162, 162, 162]
        assert stats['largest_factor'] == 162


class TestNoCleanScalingForLERW:
    """Negative-result tests: the divisor-structure statistics do
    not scale as a clean power of L that tracks d_f.
    """

    def test_largest_divisor_does_not_scale_as_clean_power_of_L(self):
        """Scaling of largest divisor with L is erratic in 2D
        because it depends on L's prime factorisation, not on a
        universal LERW-like exponent.
        """
        import numpy as np
        largest_by_L = {}
        for L in (3, 4, 5):
            s = sandpile_group_stats(L, 2)
            largest_by_L[L] = s['largest_factor']
        # largest divisors: L=3 -> 18, L=4 -> 96, L=5 -> 50.
        # log(18)/log(3) = 2.63, log(96)/log(4) = 3.29,
        # log(50)/log(5) = 2.43. Range 2.43..3.29.
        slopes = [np.log(largest_by_L[L]) / np.log(L)
                  for L in (3, 4, 5)]
        slope_range = max(slopes) - min(slopes)
        assert slope_range > 0.5, \
            f"slope_range={slope_range:.3f} --- too clean for the " \
            f"documented erratic-scaling finding; check test."

    def test_num_factors_grows_with_L_in_d2(self):
        """Number of non-trivial invariant factors grows in L.
        (Quasi-linear in L --- the 1D boundary modes of the torus
        Laplacian's eigenvalue structure.)
        """
        counts = [sandpile_group_stats(L, 2)['num_non_trivial']
                  for L in (3, 4, 5)]
        assert counts[1] > counts[0]
        assert counts[2] > counts[1]
