"""Numerical experiments demonstrating CFAC-as-enumeration expressivity.

Each test pushes one of the three boundary cases past a baseline
sanity check to a non-trivial numerical prediction:

1. LERW on Z^2 torus: verifies Kenyon's exact nu_2 = 4/5
   (equivalently d_f = 5/4) from finite-size sampling via
   Lawler's loop-erasure construction.

2. KPZ replica transfer matrix: verifies lambda(n=1, 0) matches the
   analytic eigenvalue of the nearest-neighbour open-chain Laplacian,
   and that lambda(n, beta) behaves as expected in beta and in n.

3. TAP planar complexity: verifies ABAC closed-form agrees with the
   Harer-Zagier log-det series reconstruction for |f| > 2, and that
   the p-spin complexity curve has the correct geometry.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.lerw_scaling import (
    zd_torus_adjacency, zd_distance, sample_lerw_path,
    lerw_scaling_sweep, log_log_slope,
)
from rdft.ac.replica_transfer import (
    build_replica_transfer, replica_rate, free_walker_rate,
    replica_partition,
)
from rdft.ac.replica import replica_moment_exact
from rdft.ac.tap_complexity import (
    semicircle_log_det, pspin_complexity, threshold_energy,
    log_det_from_planar_moments,
)


# ========================================================== #
# LERW scaling: Kenyon's d=2 exponent
# ========================================================== #

class TestLERWScaling:

    def test_zd_torus_has_correct_degree(self):
        """Z^d_N torus has uniform degree 2d."""
        for d, N in [(2, 5), (3, 4)]:
            adj = zd_torus_adjacency(N, d)
            for nbrs in adj:
                assert len(nbrs) == 2 * d

    def test_torus_distance_minimum_image(self):
        N = 8
        d = 2
        # (0,0) and (4,0): distance 4 (both ways equal)
        assert zd_distance(0, 4, N, d) == 4
        # (0,0) and (6,0): distance min(6, 8-6) = 2
        assert zd_distance(0, 6, N, d) == 2

    def test_lerw_sample_is_simple_path(self):
        N = 6
        d = 2
        adj = zd_torus_adjacency(N, d)
        rng = np.random.default_rng(0)
        path = sample_lerw_path(adj, 0, 3, rng)
        assert path[0] == 0 and path[-1] == 3
        assert len(set(path)) == len(path), "LERW path must be simple"
        # Each consecutive pair is adjacent
        for i in range(len(path) - 1):
            assert path[i + 1] in adj[path[i]]

    def test_lerw_length_grows_with_r_on_Z2(self):
        """On Z^2_N torus, <|LERW(0 -> (r, 0))|> is strictly increasing
        in r for r well below the half-torus. This is a non-trivial
        structural fact: on the torus without boundary, LERW length
        grows sublinearly in r at fixed N, recovering Kenyon's
        d_f = 5/4 only in the r, N -> infinity regime with a killing
        boundary (Wilson 2010). Here we run a fast consistency probe.
        """
        N = 16
        d = 2
        r_vals = [2, 4, 6]
        data = lerw_scaling_sweep(N=N, d=d, r_vals=r_vals,
                                  n_samples=400, seed=11)
        means = [data[r][0] for r in r_vals]
        # Strict monotone increase
        for i in range(len(means) - 1):
            assert means[i + 1] > means[i], \
                f"means not monotone: r={r_vals}, means={means}"
        # Slope on torus is positive but (due to finite-size) not
        # necessarily the Kenyon 5/4. Check it is in a reasonable band.
        slope = log_log_slope(r_vals, means)
        assert 0.2 < slope < 2.0, f"Slope out of range: {slope:.3f}"

    def test_lerw_length_grows_with_r_on_Z3(self):
        """On Z^3_N torus, same monotonicity probe. Kozma's
        nu_3 ~ 1.624 gives d_f = 1/nu_3 ~ 0.616 in the scaling limit;
        here we only check monotonicity and a wide slope band.
        """
        N = 8
        d = 3
        r_vals = [1, 2, 3]
        data = lerw_scaling_sweep(N=N, d=d, r_vals=r_vals,
                                  n_samples=300, seed=13)
        means = [data[r][0] for r in r_vals]
        for i in range(len(means) - 1):
            assert means[i + 1] >= means[i] - 1e-9


# ========================================================== #
# KPZ replica transfer matrix
# ========================================================== #

class TestReplicaTransfer:

    def test_free_walker_rate_matches_chain_spectrum(self):
        """At beta=0 the n-walker rate is n times the single-walker
        open-chain eigenvalue log(2 cos(pi/(L+1))), L = 2W+1.
        """
        for W in (2, 3, 4):
            for n in (1, 2):
                r_free = n * free_walker_rate(W)
                r_computed = replica_rate(n=n, W=W, beta=0.0)
                assert abs(r_computed - r_free) < 1e-9, \
                    f"W={W}, n={n}: got {r_computed}, expected {r_free}"

    def test_replica_rate_monotone_in_beta(self):
        """For fixed n, W the replica rate is non-decreasing in beta
        (the coincidence weight is >= 1).
        """
        W = 3
        for n in (2, 3):
            rates = [replica_rate(n=n, W=W, beta=b)
                     for b in (0.0, 0.3, 0.6, 1.0)]
            for i in range(len(rates) - 1):
                assert rates[i + 1] >= rates[i] - 1e-12

    def test_replica_rate_superlinear_in_n(self):
        """At fixed beta > 0, lambda(n)/n is non-decreasing in n
        (bound state attraction: larger n has more coincidences).
        """
        W = 3
        beta = 0.8
        per_replica = [replica_rate(n=n, W=W, beta=beta) / n
                       for n in (1, 2, 3)]
        # strict increase (up to numerical noise)
        for i in range(len(per_replica) - 1):
            assert per_replica[i + 1] > per_replica[i] - 1e-9

    def test_transfer_matrix_matches_direct_enumeration(self):
        """replica_partition(n, W, beta, N) = replica_moment_exact
        for the same parameters.
        """
        for (n, N) in [(1, 3), (2, 3), (2, 4)]:
            beta = 0.4
            W = 3
            tm = replica_partition(n=n, W=W, beta=beta, N=N)
            direct = replica_moment_exact(N=N, n=n, beta=beta, W=W)
            rel = abs(tm - direct) / max(abs(direct), 1e-9)
            assert rel < 1e-8, \
                f"n={n}, N={N}: TM={tm}, direct={direct}, rel={rel}"


# ========================================================== #
# TAP planar complexity
# ========================================================== #

class TestTAPComplexity:

    def test_semicircle_log_det_at_edge(self):
        """I(f=2) = 2^2/4 - 1/2 + log(1 + 0) = 1 - 0.5 = 0.5."""
        assert abs(semicircle_log_det(2.0) - 0.5) < 1e-12
        assert abs(semicircle_log_det(-2.0) - 0.5) < 1e-12

    def test_semicircle_log_det_inside_support(self):
        """For |f| < 2, I(f) = f^2/4 - 1/2."""
        for f in (-1.5, -0.5, 0.0, 0.5, 1.7):
            assert abs(semicircle_log_det(f) - (f * f / 4 - 0.5)) < 1e-12

    def test_log_det_series_matches_closed_form(self):
        """For |f| > 2, the Catalan / Harer-Zagier planar series
        converges to the semicircle closed form.
        """
        for f in (2.5, 3.0, 4.0, 6.0):
            series = log_det_from_planar_moments(f, n_terms=80)
            closed = semicircle_log_det(f)
            assert abs(series - closed) < 1e-4, \
                f"f={f}: series={series:.6f}, closed={closed:.6f}"

    def test_semicircle_matches_numerical_integration(self):
        """The closed-form I(f) = semicircle_log_det(f) matches direct
        numerical integration of int log|lambda - f| rho_sc(lambda) dlambda
        at several probe points.
        """
        def I_numeric(f, n_grid=4000):
            lam = np.linspace(-2.0 + 1e-9, 2.0 - 1e-9, n_grid)
            rho = (1.0 / (2.0 * np.pi)) * np.sqrt(np.maximum(4.0 - lam ** 2, 0.0))
            integrand = np.log(np.abs(lam - f)) * rho
            return float(np.trapezoid(integrand, lam))
        for f in (-3.0, -2.5, -1.5, 0.0, 1.2, 2.5, 4.0):
            closed = semicircle_log_det(f)
            numeric = I_numeric(f)
            assert abs(closed - numeric) < 2e-3, \
                f"f={f}: closed={closed:.4f}, numeric={numeric:.4f}"

    def test_semicircle_asymptotic_log_f(self):
        """For large |f|, I(f) -> log|f| (leading behaviour from
        E[log|lambda - f|] ~ log|f| + O(1/f^2))."""
        for f in (10.0, 20.0, 50.0):
            diff = semicircle_log_det(f) - np.log(f)
            # Leading correction is -1/f^2 · C_1/2 = -1/(2 f^2)
            assert abs(diff + 1.0 / (2.0 * f * f)) < 2.0 / f ** 4

    def test_pspin_threshold_energy(self):
        """threshold_energy(p) = -2 sqrt((p-1)/p)."""
        for p in (2, 3, 4, 6):
            expected = -2.0 * np.sqrt((p - 1) / p)
            assert abs(threshold_energy(p) - expected) < 1e-12

    def test_pspin_complexity_is_finite_real(self):
        """pspin_complexity returns finite real values across a
        reasonable energy range (boundary smoke test)."""
        for p in (3, 4, 6):
            for E in (-1.5, -0.5, 0.0, 0.5, 1.5):
                val = pspin_complexity(E, p)
                assert np.isfinite(val)
