"""Cubic-in-n analysis of the KPZ replica rate (Bethe-ansatz signature).

Witnesses Prop 3 of paper/cfac/enumerative_boundary.tex at the
analytic level: extracts the binding coefficient b(beta) from
exact transfer-matrix eigenvalues at n = 1, 2, 3, and verifies
that the ratio D(3)/D(2) -> 4 (Kardar continuum value) as the
lattice width W grows.
"""

from __future__ import annotations

from rdft.ac.replica_cubic import (
    cubic_coefficient, cubic_W_sweep, kardar_ratio_distance,
    replica_rates_n123,
)


class TestReplicaCubic:

    def test_zero_beta_gives_zero_binding(self):
        """At beta = 0 the walkers are non-interacting:
        D(2) = D(3) = 0 (lambda(n) = n * lambda(1) exactly).
        """
        for W in (3, 4):
            res = cubic_coefficient(W, beta=0.0)
            assert abs(res['D2']) < 1e-10
            assert abs(res['D3']) < 1e-10

    def test_positive_beta_gives_positive_binding(self):
        """At beta > 0 the contact attraction lowers the n-walker
        ground-state energy, so lambda(n) > n*lambda(1) for n >= 2.
        """
        for W in (3, 4, 5):
            for beta in (0.3, 0.6, 1.0):
                res = cubic_coefficient(W, beta)
                assert res['D2'] > 0, \
                    f"W={W} beta={beta} D2={res['D2']} not > 0"
                assert res['D3'] > 0, \
                    f"W={W} beta={beta} D3={res['D3']} not > 0"

    def test_d3_over_d2_below_kardar_at_finite_W(self):
        """At finite W the lattice cubic ratio is below Kardar's
        continuum value of 4 (the open-chain finite size suppresses
        the bound-state binding for larger n more than for n = 2).
        """
        for W in (3, 4, 5, 6, 7):
            res = cubic_coefficient(W, beta=0.8)
            assert res['ratio'] < 4.0, \
                f"W={W}: ratio={res['ratio']} >= 4 (above Kardar)"
            # Lower bound: the cubic shape gives at least 3 once
            # the bound state forms.
            assert res['ratio'] > 3.0, \
                f"W={W}: ratio={res['ratio']} <= 3 (below cubic)"

    def test_kardar_ratio_monotone_in_W(self):
        """D(3)/D(2) is monotone increasing in W (continuum limit
        approached from below). Tested at a moderate beta where
        the bound state is well-formed.
        """
        beta = 0.8
        sweep = cubic_W_sweep([3, 4, 5, 6, 7], beta)
        ratios = [sweep[W]['ratio'] for W in (3, 4, 5, 6, 7)]
        for r1, r2 in zip(ratios, ratios[1:]):
            assert r2 > r1 - 1e-9, \
                f"ratio not monotone: {ratios}"
        # The W = 7 ratio should be visibly closer to 4 than W = 3.
        assert (4 - ratios[-1]) < (4 - ratios[0]) - 0.05

    def test_b_lsq_between_two_estimators(self):
        """The least-squares cubic coefficient sits between the
        two single-point estimators (D(2) and D(3)/4); it equals
        their (4-weighted) mean.
        """
        for W in (3, 5):
            res = cubic_coefficient(W, beta=0.7)
            lo, hi = sorted([res['b_from_D2'], res['b_from_D3']])
            assert lo - 1e-12 <= res['b_lsq'] <= hi + 1e-12

    def test_replica_rates_n123_consistency(self):
        """replica_rates_n123 returns the same numbers as direct
        replica_rate calls (smoke test on the wrapper).
        """
        from rdft.ac.replica_transfer import replica_rate
        W = 4
        beta = 0.6
        l1, l2, l3 = replica_rates_n123(W, beta)
        assert abs(l1 - replica_rate(1, W, beta)) < 1e-12
        assert abs(l2 - replica_rate(2, W, beta)) < 1e-12
        assert abs(l3 - replica_rate(3, W, beta)) < 1e-12

    def test_kardar_ratio_distance_decreases(self):
        """kardar_ratio_distance(W, beta) is the distance |ratio - 4|;
        it decreases monotonically in W (continuum limit).
        """
        beta = 0.6
        d3 = kardar_ratio_distance(3, beta)
        d5 = kardar_ratio_distance(5, beta)
        d7 = kardar_ratio_distance(7, beta)
        assert d5 < d3
        assert d7 < d5
