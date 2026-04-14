"""
tests/test_omega_integration.py
================================
Tests for OmegaIntegration.edge_basis_matrix and the reduce() method.

These verify the sign convention fix:
  Tree edges ALWAYS get -1 in the circuit constraint.
  Loop (non-tree) edge ALWAYS gets +1.

This implements the RDFT Schwinger-parameter convention:
  α_loop = Σ_{tree path} α_{e'}
which always has a solution in the positive orthant.

Tests:
  1. Sunset (parallel edges, L=2):  Ψ_r = 3α₀²
  2. Bubble (anti-parallel, L=1):   Ψ_r = 2α₀
  3. Triangle (directed cycle, L=1): Ψ_r = 2α₀+2α₁
  4. E_f matrix is consistent (loop edges = +1, tree edges = -1)
  5. Circuit constraints have positive solutions
"""

import pytest
import sympy as sp

from rdft.graphs.incidence import FeynmanGraph
from rdft.integrals.symanzik import SymanzikPolynomials
from rdft.integrals.parametric import OmegaIntegration


class TestOmegaIntegration:

    def test_sunset_Psi_r(self):
        """
        Sunset: Ψ_r = 3α₀² after ω-integration.

        The factor 3 is the number of spanning trees (= CFAC counting).
        After the two constraints α₁=α₀, α₂=α₀:
            Ψ = α₀α₁+α₀α₂+α₁α₂ → 3α₀²
        """
        G = FeynmanGraph.sunset()
        syms = SymanzikPolynomials(G)
        omega = OmegaIntegration(G)

        Psi_r, _, subs = omega.reduce(syms.Psi, sp.Integer(0))
        Psi_r = sp.expand(Psi_r)

        a0 = G._alpha_syms[0]
        expected = 3 * a0**2

        assert sp.simplify(Psi_r - expected) == 0, (
            f"Sunset Ψ_r: expected {expected}, got {Psi_r}"
        )

    def test_sunset_subs_positive(self):
        """Sunset substitutions set α₁=α₀, α₂=α₀ (positive, not negative)."""
        G = FeynmanGraph.sunset()
        omega = OmegaIntegration(G)
        a0, a1, a2 = G._alpha_syms

        _, _, subs = omega.reduce(syms := SymanzikPolynomials(G).Psi, sp.Integer(0))

        # Both α₁ and α₂ should equal α₀ (not -α₀)
        assert sp.simplify(subs[a1] - a0) == 0, f"Expected α₁=α₀, got {subs[a1]}"
        assert sp.simplify(subs[a2] - a0) == 0, f"Expected α₂=α₀, got {subs[a2]}"

    def test_bubble_Psi_r(self):
        """
        Bubble (one-loop self-energy): Ψ_r = 2α₀ after ω-integration.

        Constraint α₁=α₀ gives Ψ = α₀+α₁ → 2α₀.
        """
        G = FeynmanGraph.one_loop_self_energy()
        syms = SymanzikPolynomials(G)
        omega = OmegaIntegration(G)

        Psi_r, _, subs = omega.reduce(syms.Psi, sp.Integer(0))
        Psi_r = sp.expand(Psi_r)

        a0 = G._alpha_syms[0]
        expected = 2 * a0

        assert sp.simplify(Psi_r - expected) == 0, (
            f"Bubble Ψ_r: expected {expected}, got {Psi_r}"
        )

    def test_bubble_subs_positive(self):
        """Bubble substitution sets α₁=α₀ (positive)."""
        G = FeynmanGraph.one_loop_self_energy()
        omega = OmegaIntegration(G)
        a0, a1 = G._alpha_syms

        _, _, subs = omega.reduce(SymanzikPolynomials(G).Psi, sp.Integer(0))

        assert sp.simplify(subs[a1] - a0) == 0, f"Expected α₁=α₀, got {subs[a1]}"

    def test_triangle_Psi_r(self):
        """
        Triangle (directed 3-cycle): Ψ_r = 2α₀+2α₁ after ω-integration.

        Constraint α₂=α₀+α₁ gives Ψ = α₀+α₁+α₂ → 2α₀+2α₁.
        """
        G = FeynmanGraph.three_vertex_loop()
        syms = SymanzikPolynomials(G)
        omega = OmegaIntegration(G)

        Psi_r, _, subs = omega.reduce(syms.Psi, sp.Integer(0))
        Psi_r = sp.expand(Psi_r)

        a0, a1 = G._alpha_syms[0], G._alpha_syms[1]
        expected = 2*a0 + 2*a1

        assert sp.simplify(Psi_r - expected) == 0, (
            f"Triangle Ψ_r: expected {expected}, got {Psi_r}"
        )

    def test_triangle_subs_positive(self):
        """Triangle substitution sets α₂=α₀+α₁ (sum of positives)."""
        G = FeynmanGraph.three_vertex_loop()
        omega = OmegaIntegration(G)
        a0, a1, a2 = G._alpha_syms

        _, _, subs = omega.reduce(SymanzikPolynomials(G).Psi, sp.Integer(0))

        assert sp.simplify(subs[a2] - (a0 + a1)) == 0, (
            f"Expected α₂=α₀+α₁, got {subs[a2]}"
        )

    def test_Ef_tree_edges_always_minus_one(self):
        """
        The E_f matrix should have -1 for all tree edges in each circuit.

        Specifically: for each row l (one per loop), the entry for the
        loop (non-tree) edge is +1, and all other non-zero entries are -1.
        """
        for G in [FeynmanGraph.sunset(),
                  FeynmanGraph.one_loop_self_energy(),
                  FeynmanGraph.three_vertex_loop()]:
            omega = OmegaIntegration(G)
            Ef = omega.edge_basis_matrix
            int_edges = G.internal_edge_indices
            tree = omega._find_spanning_tree()
            non_tree = [e for e in int_edges if e not in tree]

            for l, loop_edge_idx in enumerate(non_tree):
                loop_pos = int_edges.index(loop_edge_idx)
                for e_pos in range(len(int_edges)):
                    entry = int(Ef[l, e_pos])
                    if e_pos == loop_pos:
                        assert entry == 1, (
                            f"Graph {G}: row {l}, loop edge pos {e_pos}: "
                            f"expected +1, got {entry}"
                        )
                    elif entry != 0:
                        assert entry == -1, (
                            f"Graph {G}: row {l}, tree edge pos {e_pos}: "
                            f"expected -1, got {entry}"
                        )

    def test_Psi_r_is_positive_definite(self):
        """
        After reduction, Ψ_r evaluated at any positive α values must be > 0.
        """
        test_cases = [
            (FeynmanGraph.sunset(), {'alpha0': 1.0}),
            (FeynmanGraph.one_loop_self_energy(), {'alpha0': 1.0}),
            (FeynmanGraph.three_vertex_loop(), {'alpha0': 1.0, 'alpha1': 2.0}),
        ]

        for G, eval_point in test_cases:
            omega = OmegaIntegration(G)
            syms = SymanzikPolynomials(G)
            Psi_r, _, _ = omega.reduce(syms.Psi, sp.Integer(0))

            # Evaluate Ψ_r at the given positive values
            subs_dict = {sp.Symbol(k, positive=True): v
                         for k, v in eval_point.items()}
            val = float(Psi_r.subs(subs_dict))

            assert val > 0, (
                f"Graph {G}: Ψ_r = {Psi_r} is not positive at {eval_point}, "
                f"got {val}"
            )

    def test_n_free_parameters_after_reduction(self):
        """
        After L substitutions, Ψ_r should depend on n_int - L free parameters.
        """
        for G in [FeynmanGraph.sunset(),
                  FeynmanGraph.one_loop_self_energy(),
                  FeynmanGraph.three_vertex_loop()]:
            omega = OmegaIntegration(G)
            syms = SymanzikPolynomials(G)
            Psi_r, _, subs = omega.reduce(syms.Psi, sp.Integer(0))

            # Count free symbols in Psi_r
            free_syms = Psi_r.free_symbols
            expected_free = G.n_internal_edges - G.L

            assert len(free_syms) == expected_free, (
                f"Graph {G}: expected {expected_free} free params in Ψ_r, "
                f"got {len(free_syms)} ({free_syms})"
            )

    def test_sunset_cfac_counting_factor(self):
        """
        Ψ_r = 3α₀² for sunset: the coefficient 3 equals the number of
        spanning trees (= CFAC counting factor).
        """
        G = FeynmanGraph.sunset()
        omega = OmegaIntegration(G)
        syms = SymanzikPolynomials(G)
        Psi_r, _, _ = omega.reduce(syms.Psi, sp.Integer(0))

        a0 = G._alpha_syms[0]
        poly = sp.Poly(Psi_r, a0)
        leading_coeff = poly.nth(G.L)  # coefficient of α₀^L

        n_spanning_trees = len(G.spanning_trees_from_kirchhoff())

        assert int(leading_coeff) == n_spanning_trees, (
            f"Leading coefficient of Ψ_r = {leading_coeff}, "
            f"spanning trees = {n_spanning_trees}: should be equal"
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
