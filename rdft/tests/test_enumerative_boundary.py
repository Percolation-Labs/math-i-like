"""Verification tests for paper/cfac/enumerative_boundary.tex.

Covers the three propositions:

1. LERW on a finite graph = ratio of Kirchhoff polynomials
   (rdft.ac.lerw).
2. GUE moments have the Harer-Zagier genus expansion in terms of
   rooted one-face orientable map counts (rdft.ac.matrix_model).
3. Replicated directed polymer = n-species CRN with pairwise
   delta-attraction, verified against Monte Carlo (rdft.ac.replica).
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw import (
    kirchhoff, contract_path, simple_paths, lerw_path_prob,
    lerw_length_gf_numer, lerw_mean_length,
)
from rdft.ac.matrix_model import (
    catalan, rooted_one_face_maps, gue_trace_moment, planar_moment,
)
from rdft.ac.replica import (
    replica_moment_exact, replica_moment_mc, coincidence_count,
)


# ========================================================== #
# LERW: Proposition 1 and Corollary 1
# ========================================================== #

def _k4_graph():
    """Complete graph K_4 with unit edge weights."""
    edges = []
    n = 4
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v, 1.0))
    return (n, edges)


def _cycle_graph(n: int):
    edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    # normalise to u < v
    edges = [(min(u, v), max(u, v), w) for u, v, w in edges]
    return (n, edges)


class TestLERW:

    def test_kirchhoff_k4(self):
        """Spanning trees of K_4: Cayley gives n^{n-2} = 16."""
        G = _k4_graph()
        tau = kirchhoff(G)
        assert abs(tau - 16.0) < 1e-9

    def test_kirchhoff_cycle(self):
        """Cycle C_n has exactly n spanning trees (remove any one edge)."""
        for n in (3, 4, 5, 6):
            tau = kirchhoff(_cycle_graph(n))
            assert abs(tau - n) < 1e-9, f"C_{n}: tau={tau}"

    def test_lerw_probabilities_sum_to_one(self):
        """Prop 1 must give a probability measure on simple a->b paths."""
        G = _k4_graph()
        for a, b in [(0, 1), (0, 2), (1, 3)]:
            total = 0.0
            for path in simple_paths(G, a, b):
                total += lerw_path_prob(G, path)
            assert abs(total - 1.0) < 1e-9, \
                f"K4: LERW mass from {a}->{b} = {total}"

    def test_lerw_on_cycle_graph(self):
        """On C_n, the two arcs from 0 to k have equal length
        distributions only when n is even and k = n/2.
        Otherwise the shorter arc has higher LERW weight.
        """
        G = _cycle_graph(6)  # C_6
        # Paths 0 -> 3 in C_6: two arcs of length 3 each -> equal weight 1/2.
        weights = sorted(lerw_path_prob(G, p) for p in simple_paths(G, 0, 3))
        assert abs(weights[0] - 0.5) < 1e-9
        assert abs(weights[1] - 0.5) < 1e-9
        # Paths 0 -> 1 in C_6: short arc length 1, long arc length 5.
        weights01 = sorted(lerw_path_prob(G, p)
                           for p in simple_paths(G, 0, 1))
        # Matrix-tree identity gives a concrete pair that sums to 1.
        assert abs(sum(weights01) - 1.0) < 1e-9
        # Short path must carry more mass than long path.
        assert weights01[1] > weights01[0]

    def test_contract_path_reduces_vertex_count(self):
        G = _k4_graph()
        path = [0, 1, 2]
        Gc = contract_path(G, path)
        assert Gc[0] == G[0] - len(path) + 1

    def test_lerw_length_gf_numer_on_k4(self):
        """The length GF numerator on K_4 between two vertices has
        nonzero coefficients at lengths 1, 2, 3 and equals the weighted
        Kirchhoff enumeration. The constant term (length 0) is zero
        since a != b.
        """
        G = _k4_graph()
        coeffs = lerw_length_gf_numer(G, 0, 1)
        # Path 0->1 direct (length 1): tau(K_4 / edge) = tau(K_3 with
        # one doubled edge... actually = spanning trees of K_3 with
        # parallel edges). For unit weights, kirchhoff returns 3
        # spanning trees (Cayley for n=3 = 3) times edge-doubling.
        # Independent check: the probabilities must sum to 1 after
        # normalisation by kirchhoff(G).
        tau_G = kirchhoff(G)
        total = sum(coeffs) / tau_G
        assert abs(total - 1.0) < 1e-9
        # Length 0 coefficient is zero.
        assert coeffs[0] == 0.0

    def test_lerw_mean_length_positive(self):
        G = _k4_graph()
        m = lerw_mean_length(G, 0, 1)
        # On K_4 with unit weights, a direct 1-edge path exists so
        # the mean length is bounded by 3 (longest simple path).
        assert 1.0 <= m <= 3.0


# ========================================================== #
# Matrix model / Harer-Zagier: Proposition 2
# ========================================================== #

class TestMatrixModel:

    def test_catalan_base_case(self):
        """eps_0(k) = C_k (planar = Catalan)."""
        expected = [1, 1, 2, 5, 14, 42, 132]  # C_0..C_6
        for k, c in enumerate(expected):
            assert catalan(k) == c
            assert rooted_one_face_maps(k, 0) == c

    def test_harer_zagier_small_cases(self):
        """Harer-Zagier first genus-1 values.

        Established small values (see e.g. the HZ paper and the
        encyclopaedic OEIS/Goulden-Jackson tables):
            eps_1(2) = 1, eps_1(3) = 10, eps_1(4) = 70, eps_1(5) = 420.
        """
        assert rooted_one_face_maps(2, 1) == 1
        assert rooted_one_face_maps(3, 1) == 10
        assert rooted_one_face_maps(4, 1) == 70
        assert rooted_one_face_maps(5, 1) == 420

    def test_harer_zagier_row_sum_at_N1(self):
        """Sum_g eps_g(k) * N^{k+1-2g} at N=1 should equal (2k-1)!!
        because E[tr M^{2k}] at N=1 is the 2k-th moment of a standard
        Gaussian scalar, which is (2k-1)!!.
        """
        def doublefact(m: int) -> int:
            r = 1
            while m > 1:
                r *= m
                m -= 2
            return r
        for k in range(1, 7):
            val = gue_trace_moment(1, k)
            expected = doublefact(2 * k - 1)
            assert abs(val - expected) < 1e-9, \
                f"k={k}: gue_trace_moment(1,{k})={val}, (2k-1)!!={expected}"

    def test_gue_moment_leading_large_N(self):
        """E[tr M^{2k}] / N^{k+1} -> C_k as N -> infinity (Wigner)."""
        for k in (1, 2, 3, 4):
            N = 200
            ratio = gue_trace_moment(N, k) / N ** (k + 1)
            assert abs(ratio - catalan(k)) / catalan(k) < 5e-3, \
                f"k={k}: ratio={ratio}, Catalan={catalan(k)}"

    def test_planar_moment_matches_catalan(self):
        for k in range(8):
            assert planar_moment(k) == catalan(k)


# ========================================================== #
# KPZ replica CRN: Proposition 3
# ========================================================== #

class TestReplica:

    def test_coincidence_count(self):
        assert coincidence_count([0, 0]) == 4
        assert coincidence_count([0, 1]) == 2
        assert coincidence_count([0, 0, 1]) == 5
        assert coincidence_count([0, 1, 2]) == 3

    def test_replica_n1_is_path_count(self):
        """At n = 1 and beta = 0, E[Z_N] = (#paths). With nearest-
        neighbour walks on [-W, W] starting at 0 with reflecting ends,
        the path count at length N equals the number of length-N
        walks on that interval.
        """
        N = 4
        W = 3
        # At beta = 0, every walk contributes 1. Count paths.
        val = replica_moment_exact(N=N, n=1, beta=0.0, W=W)
        # Independent count via transfer matrix:
        sites = np.arange(-W, W + 1)
        n_sites = len(sites)
        T = np.zeros((n_sites, n_sites))
        for i in range(n_sites):
            if i - 1 >= 0:
                T[i, i - 1] = 1.0
            if i + 1 < n_sites:
                T[i, i + 1] = 1.0
        v = np.zeros(n_sites)
        v[W] = 1.0  # origin
        count = (np.linalg.matrix_power(T, N) @ v).sum()
        assert abs(val - count) < 1e-9

    def test_replica_matches_mc_n1(self):
        """E[Z_N] at n=1: exact = MC within 3 sigma."""
        N = 3
        beta = 0.5
        exact = replica_moment_exact(N=N, n=1, beta=beta, W=3)
        mc_mean, mc_sem = replica_moment_mc(
            N=N, n=1, beta=beta, W=3, n_samples=4000,
            rng=np.random.default_rng(42),
        )
        # E[Z] = (#paths) * exp(beta^2/2) per time slice per walker
        # since each independent Gaussian contributes exp(beta^2/2).
        # Exact computation already folds that in via coincidence = 1.
        z = (exact - mc_mean) / mc_sem
        assert abs(z) < 3.5, \
            f"n=1: exact={exact:.4f}, MC={mc_mean:.4f}+/-{mc_sem:.4f} z={z:.2f}"

    def test_replica_matches_mc_n2(self):
        """E[Z_N^2] at n=2: exact = MC within 3 sigma. The
        coincidence-count weighting is the non-trivial piece.
        """
        N = 3
        beta = 0.5
        exact = replica_moment_exact(N=N, n=2, beta=beta, W=3)
        mc_mean, mc_sem = replica_moment_mc(
            N=N, n=2, beta=beta, W=3, n_samples=8000,
            rng=np.random.default_rng(43),
        )
        z = (exact - mc_mean) / mc_sem
        assert abs(z) < 3.5, \
            f"n=2: exact={exact:.4f}, MC={mc_mean:.4f}+/-{mc_sem:.4f} z={z:.2f}"

    def test_replica_moment_grows_with_beta(self):
        """E[Z_N^n] must be monotone increasing in beta^2 at fixed n,
        since the coincidence weight is nonneg.
        """
        N = 3
        vals = [replica_moment_exact(N=N, n=2, beta=b, W=2)
                for b in (0.0, 0.3, 0.6, 1.0)]
        for i in range(len(vals) - 1):
            assert vals[i + 1] > vals[i]
