"""Tests for the non-local Ward-identity projector (conserved_3.py).

Documents what the v3 module does and does not establish:
- Provides the Ward-identity constraint expression
- Assembles all five pieces of the C_3 slotting evidence
- Explicitly identifies that polynomial DSEs alone (even with Ward
  constraint) do NOT admit a C_3 multicritical point — the missing
  ingredient is genuine non-locality.
"""
import numpy as np
import sympy as sp
import pytest

from rdft.ac.conserved_3 import (
    ward_identity_constraint,
    find_C3_multicritical_numerical,
    manna_C3_complete,
)


class TestWardIdentity:

    def test_ward_constraint_is_symbolic_expression(self):
        """ward_identity_constraint returns a sympy expression."""
        lam, chi, G, z = sp.symbols('lambda chi G z', positive=True)
        expr = ward_identity_constraint(lam, chi, G, z)
        assert isinstance(expr, sp.Expr)

    def test_ward_constraint_vanishes_at_specific_point(self):
        """At G=0, z=0 the second-derivative factor is 1, ward = -lambda."""
        lam, chi, G, z = sp.symbols('lambda chi G z', positive=True)
        expr = ward_identity_constraint(lam, chi, G, z)
        # At G=0, z=0: chi^2 * 0 / (D+D') - lambda * 1 = -lambda
        val = expr.subs([(G, 0), (z, 0)])
        assert sp.simplify(val + lam) == 0

    def test_ward_constraint_has_correct_chi_dependence(self):
        """The Ward constraint is QUADRATIC in chi (chi^2 term from soft mode)."""
        lam, chi, G, z = sp.symbols('lambda chi G z', positive=True)
        expr = ward_identity_constraint(lam, chi, G, z)
        chi_poly = sp.Poly(expr, chi)
        # chi^2 coefficient should be G^2 / (D_psi + D_rho) = G^2 / 2
        assert chi_poly.degree() == 2


class TestNumericalMulticritical:

    def test_no_polynomial_solution_in_simple_truncation(self):
        """The simple polynomial ansatz [1, chi G, -lambda G^2] does NOT
        admit a C_3 multicritical solution.

        This is the key NEGATIVE result that motivates the v3 conclusion:
        a polynomial DSE alone, even with the Ward constraint added, is
        OVERDETERMINED for the C_3 multicritical conditions.  The missing
        ingredient is the non-local diffusive convolution that the
        polynomial truncation cannot encode.
        """
        for chi in [0.5, 1.0, 2.0, 5.0]:
            sol = find_C3_multicritical_numerical(chi)
            assert sol is None, (
                f"chi={chi}: unexpected solution {sol}; the polynomial "
                "truncation should be overdetermined for the C_3 conditions"
            )


class TestMannaC3Complete:

    def test_assembles_all_five_pieces(self):
        """manna_C3_complete returns the assembled evidence dict."""
        out = manna_C3_complete()
        for key in ['codimension_argument', 'algebraic_accessibility',
                    'bridge_scale_prediction', 'remaining_open',
                    'tau_predicted', 'within_observed_range']:
            assert key in out

    def test_predicted_tau_within_observed_range(self):
        """tau_predicted = 1.327 is within [1.275, 1.338]."""
        out = manna_C3_complete()
        assert out['within_observed_range'] is True
        lo, hi = out['tau_observed_range']
        assert lo <= out['tau_predicted'] <= hi

    def test_predicted_tau_close_to_central_observation(self):
        """tau_predicted is within 3% of central observation 1.30."""
        out = manna_C3_complete()
        rel = abs(out['tau_predicted'] - 1.30) / 1.30
        assert rel < 0.03

    def test_skeleton_is_4_3(self):
        out = manna_C3_complete()
        assert abs(out['tau_skeleton_C3'] - 4.0 / 3.0) < 1e-12

    def test_dressing_negative(self):
        """gamma_3 dressing is negative (anomalous dimension reduces tau)."""
        out = manna_C3_complete()
        assert out['gamma_3_predicted'] < 0

    def test_eps_at_d1_is_2(self):
        """At d=1 (1+1d), eps = d_c(3) - 1 = 2."""
        out = manna_C3_complete(d=1.0)
        assert out['eps'] == 2.0

    def test_n_eff_default_is_3(self):
        """Default n_eff = 3 from symmetric quartic channels."""
        out = manna_C3_complete()
        assert out['n_eff'] == 3.0
