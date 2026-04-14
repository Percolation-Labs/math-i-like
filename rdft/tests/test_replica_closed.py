"""Closed form D(2, beta; Z) = 2 beta^2 - log(2 e^{beta^2} - 1).

Tier-A result for Prop 3 of paper/cfac/enumerative_boundary.tex:
the 2-body KPZ replica binding on the infinite line Z. Verified
by:
  1. Numerical transfer-matrix D(2, beta; W) converges to the
     closed form as W grows.
  2. Leading small-beta coefficient: D(2) ~ beta^4 (Kardar scaling).
  3. Taylor series coefficient by coefficient against an
     independent term-by-term expansion.
"""

from __future__ import annotations
import numpy as np

from rdft.ac.replica_closed import (
    D2_closed_Z, lambda_relative_Z,
    bound_state_decay_length, D2_small_beta_series,
)
from rdft.ac.replica_transfer import replica_rate


class TestD2ClosedForm:

    def test_beta_zero_gives_zero(self):
        """At beta = 0 the walkers are free: D(2, 0; Z) = 0."""
        assert abs(D2_closed_Z(0.0)) < 1e-14

    def test_positive_for_positive_beta(self):
        """Attractive contact interaction raises lambda(2) above
        2 lambda(1).
        """
        for beta in (0.1, 0.3, 0.6, 1.0, 1.5):
            assert D2_closed_Z(beta) > 0

    def test_monotone_in_beta(self):
        """Binding strengthens with beta."""
        betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
        vals = [D2_closed_Z(b) for b in betas]
        for v1, v2 in zip(vals, vals[1:]):
            assert v2 > v1 - 1e-12

    def test_leading_order_is_beta_fourth(self):
        """D(2, beta; Z) ~ beta^4 as beta -> 0 (Bethe-ansatz scaling
        for 1D attractive Bose gas ground-state energy).
        """
        for beta in (0.02, 0.04, 0.08):
            d = D2_closed_Z(beta)
            ratio = d / beta ** 4
            # ratio -> 1 with leading correction -beta^2 / 3
            assert abs(ratio - 1.0) < 5 * beta ** 2

    def test_small_beta_series_matches_direct(self):
        """Series expansion to O(beta^8) reproduces direct closed
        form within the truncation error.
        """
        for beta in (0.05, 0.1, 0.15, 0.2):
            direct = D2_closed_Z(beta)
            series = D2_small_beta_series(beta, order=8)
            err = abs(direct - series)
            # Truncation error O(beta^10); allow small safety.
            assert err < 5 * beta ** 10 + 1e-14, \
                f"beta={beta}: direct={direct}, series={series}, err={err}"

    def test_relative_eigenvalue_consistency(self):
        """log(mu_rel) = D(2) + log(4), i.e. mu_rel = 4 e^{D(2)} because
        the free 2-walker relative rate is log(4).
        """
        for beta in (0.0, 0.3, 0.6, 1.0):
            lhs = lambda_relative_Z(beta)
            rhs = D2_closed_Z(beta) + float(np.log(4.0))
            assert abs(lhs - rhs) < 1e-12

    def test_decay_length_monotone_decreasing(self):
        """Bound-state localisation tightens as beta grows: the
        decay length 1/log(2 e^{beta^2} - 1) is strictly decreasing
        in beta for beta > 0.
        """
        betas = [0.3, 0.5, 0.8, 1.2, 2.0]
        lengths = [bound_state_decay_length(b) for b in betas]
        for l1, l2 in zip(lengths, lengths[1:]):
            assert l2 < l1


class TestFiniteWConverges:

    def test_finite_W_converges_to_closed_form_from_above(self):
        """Numerical TM D(2, beta; W) decreases monotonically toward
        D(2, beta; Z) as W grows. Verifies the closed-form is the
        thermodynamic-limit replica rate, and that finite-W effects
        are pure localisation corrections (wavefunction tail cut by
        reflecting boundary).
        """
        beta = 0.8
        d_Z = D2_closed_Z(beta)
        W_vals = [3, 4, 5, 6, 7, 8, 9]
        d_W = []
        for W in W_vals:
            l1 = replica_rate(1, W, beta)
            l2 = replica_rate(2, W, beta)
            d_W.append(l2 - 2 * l1)
        # Monotone decreasing
        for v1, v2 in zip(d_W, d_W[1:]):
            assert v2 < v1 + 1e-9, \
                f"D(2; W) not monotone decreasing: {d_W}"
        # All above the infinite-line value
        for v in d_W:
            assert v > d_Z - 1e-9, \
                f"D(2; W)={v} below closed form D(2; Z)={d_Z}"
        # W=9 within ~7% of the closed form (finite-size correction
        # is ~ 1/W with small prefactor for beta = 0.8).
        assert abs(d_W[-1] - d_Z) / d_Z < 0.07

    def test_finite_W_converges_for_several_betas(self):
        """For each beta, the largest-W numerical value is within a
        finite-size correction of the closed form (upper-bounded).
        """
        for beta in (0.3, 0.5, 0.8, 1.0):
            d_Z = D2_closed_Z(beta)
            W = 7
            l1 = replica_rate(1, W, beta)
            l2 = replica_rate(2, W, beta)
            d_W = l2 - 2 * l1
            # d_W is always above d_Z
            assert d_W > d_Z - 1e-9, \
                f"beta={beta} W=7: d_W={d_W} below d_Z={d_Z}"
            # Relative gap bounded (tighter for larger beta / shorter
            # localisation length)
            rel_gap = (d_W - d_Z) / max(d_Z, 1e-9)
            assert rel_gap < 2.0, \
                f"beta={beta}: rel_gap={rel_gap}"

    def test_tighter_binding_reduces_finite_size_correction(self):
        """At larger beta the bound state is more localised, so the
        same finite W captures more of the closed-form value:
        (d_W - d_Z)/d_Z at fixed W decreases in beta.
        """
        W = 5
        rel_gaps = []
        for beta in (0.5, 0.8, 1.2, 1.8):
            d_Z = D2_closed_Z(beta)
            l1 = replica_rate(1, W, beta)
            l2 = replica_rate(2, W, beta)
            d_W = l2 - 2 * l1
            rel_gaps.append((d_W - d_Z) / d_Z)
        for r1, r2 in zip(rel_gaps, rel_gaps[1:]):
            assert r2 < r1 + 1e-9, \
                f"rel_gap not monotone in beta: {rel_gaps}"
