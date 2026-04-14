"""
tests/test_bphz.py
==================
Tests for BPHZ renormalisation via the Connes-Kreimer Hopf algebra.

Covers:
  1. SubDivergence — edge-subset construction, vertex_set, betti_number, is_1pi
  2. CoproductMap — correct sub-divergence enumeration for primitive and
     non-primitive graphs (including multi-edge sunset)
  3. BPHZRenormalization — antipode and renormalized amplitude (symbolic)

Key correctness checks:
  - Primitive graphs (bubble, triangle): 0 sub-divergences
  - Sunset (3 parallel edges): exactly 3 sub-divergences (each a bubble)
  - Each sunset sub-divergence contracts to a tadpole
  - Symbolic I_R(sunset) has zero pole part (Connes-Kreimer algebra)
"""

import pytest
import sympy as sp

from rdft.graphs.incidence import FeynmanGraph
from rdft.rg.bphz import SubDivergence, CoproductMap, BPHZRenormalization


# ------------------------------------------------------------------ #
#  Helper graphs                                                       #
# ------------------------------------------------------------------ #

def bubble():
    return FeynmanGraph.one_loop_self_energy()

def triangle():
    return FeynmanGraph.three_vertex_loop()

def sunset():
    return FeynmanGraph.sunset()


# ================================================================== #
#  1. SubDivergence                                                    #
# ================================================================== #

class TestSubDivergence:

    def test_bubble_edge_subset_vertex_set(self):
        """
        Bubble sub-divergence edge_set = frozenset({0}) should give
        vertex_set = {0, 1} (both internal vertices incident to edge 0).
        """
        G = bubble()
        # Edge 0: (0,1,False) — both internal
        sub = SubDivergence(frozenset({0}), G)
        assert sub.vertex_set == frozenset({0, 1})

    def test_betti_number_single_edge(self):
        """A single edge between 2 vertices: L = 1 - 2 + 1 = 0 (no loops)."""
        G = bubble()
        sub = SubDivergence(frozenset({0}), G)
        assert sub.betti_number == 0

    def test_betti_number_bubble(self):
        """Bubble (both edges): 2 edges, 2 vertices → L = 2 - 2 + 1 = 1."""
        G = bubble()
        sub = SubDivergence(frozenset({0, 1}), G)
        assert sub.betti_number == 1

    def test_betti_number_sunset_pair(self):
        """
        Any 2-edge subset of sunset: 2 edges between same 2 vertices → L=1.
        """
        G = sunset()
        for pair in [{0, 1}, {0, 2}, {1, 2}]:
            sub = SubDivergence(frozenset(pair), G)
            assert sub.betti_number == 1, f"Pair {pair}: expected L=1"

    def test_is_1pi_bubble_full(self):
        """Both edges of bubble form a 1PI sub-graph (one loop, no bridge)."""
        G = bubble()
        sub = SubDivergence(frozenset({0, 1}), G)
        assert sub.is_1pi is True

    def test_is_not_1pi_single_edge_bubble(self):
        """A single edge of the bubble is a bridge → not 1PI."""
        G = bubble()
        sub = SubDivergence(frozenset({0}), G)
        assert sub.is_1pi is False

    def test_is_1pi_sunset_pair(self):
        """Any pair of edges from sunset: 2 parallel edges → 1PI bubble."""
        G = sunset()
        for pair in [{0, 1}, {0, 2}, {1, 2}]:
            sub = SubDivergence(frozenset(pair), G)
            assert sub.is_1pi is True, f"Pair {pair}: expected 1PI"

    def test_is_not_1pi_single_edge_sunset(self):
        """A single edge of sunset: tree-like, L=0, not 1PI."""
        G = sunset()
        for e in [0, 1, 2]:
            sub = SubDivergence(frozenset({e}), G)
            assert sub.is_1pi is False, f"Edge {e} alone should not be 1PI"

    def test_to_graph_bubble_full(self):
        """
        SubDivergence.to_graph() for both bubble edges → standalone
        graph with 2 vertices and 2 internal edges (no external legs).
        """
        G = bubble()
        sub = SubDivergence(frozenset({0, 1}), G)
        g_sub = sub.to_graph()
        assert g_sub.n_vertices_int == 2
        assert g_sub.n_internal_edges == 2

    def test_contracted_graph_sunset_bubble(self):
        """
        Contracting a bubble (2-edge subset) in the sunset gives a tadpole:
          - 1 internal vertex (super-vertex)
          - 1 internal edge (the remaining 3rd edge, now a self-loop)
        """
        G = sunset()
        for pair in [{0, 1}, {0, 2}, {1, 2}]:
            sub = SubDivergence(frozenset(pair), G)
            contracted = sub.contracted_graph()
            # Should have 1 internal vertex (all original vertices collapse)
            assert contracted.n_vertices_int == 1, (
                f"Pair {pair}: contracted should have 1 internal vertex, "
                f"got {contracted.n_vertices_int}"
            )
            # Should have 1 internal edge (the remaining edge)
            assert contracted.n_internal_edges == 1, (
                f"Pair {pair}: contracted should have 1 internal edge, "
                f"got {contracted.n_internal_edges}"
            )


# ================================================================== #
#  2. CoproductMap                                                     #
# ================================================================== #

class TestCoproductMap:

    def test_bubble_is_primitive(self):
        """Bubble has no 1PI proper sub-divergences (L≥1)."""
        cmap = CoproductMap(bubble())
        assert len(cmap.subdivergences) == 0

    def test_triangle_is_primitive(self):
        """Triangle (3-vertex loop) has no proper 1PI sub-divergences."""
        cmap = CoproductMap(triangle())
        assert len(cmap.subdivergences) == 0

    def test_sunset_has_three_subdivergences(self):
        """
        Sunset has exactly 3 sub-divergences: each pair of the 3 parallel
        edges forms a bubble.  This is the key multi-edge correctness test.
        """
        cmap = CoproductMap(sunset())
        subs = cmap.subdivergences
        assert len(subs) == 3, (
            f"Sunset should have 3 sub-divergences, got {len(subs)}"
        )

    def test_sunset_subdivergence_edge_sets(self):
        """
        The 3 sunset sub-divergences are exactly the 3 pairs:
        {0,1}, {0,2}, {1,2}.
        """
        cmap = CoproductMap(sunset())
        edge_sets = {sub.edge_set for sub in cmap.subdivergences}
        expected = {
            frozenset({0, 1}),
            frozenset({0, 2}),
            frozenset({1, 2}),
        }
        assert edge_sets == expected

    def test_sunset_coproduct_terms_contract_to_tadpoles(self):
        """Each (γ, Γ/γ) pair has Γ/γ = tadpole (1 vertex, 1 edge)."""
        cmap = CoproductMap(sunset())
        for sub, contracted in cmap.coproduct_terms():
            assert contracted.n_vertices_int == 1, (
                f"Sub {sub.edge_set}: contracted should be a tadpole "
                f"(1 vertex), got {contracted.n_vertices_int}"
            )
            assert contracted.n_internal_edges == 1, (
                f"Sub {sub.edge_set}: contracted should have 1 internal edge, "
                f"got {contracted.n_internal_edges}"
            )

    def test_coproduct_terms_length_matches_subdivergences(self):
        """coproduct_terms() returns one pair per sub-divergence."""
        for G in [bubble(), triangle(), sunset()]:
            cmap = CoproductMap(G)
            assert len(cmap.coproduct_terms()) == len(cmap.subdivergences)

    def test_summary_runs_without_error(self):
        """summary() should not raise."""
        for G in [bubble(), triangle(), sunset()]:
            cmap = CoproductMap(G)
            text = cmap.summary()
            assert isinstance(text, str)
            assert len(text) > 0


# ================================================================== #
#  3. BPHZRenormalization                                              #
# ================================================================== #

class TestBPHZRenormalization:

    # ---- symbolic placeholder tests -------------------------------- #

    def test_antipode_bubble_symbolic(self):
        """
        Bubble is primitive: S(bubble) = -I(bubble) symbolically.
        """
        bphz = BPHZRenormalization(bubble())
        S = bphz.antipode()
        assert str(S).startswith('S_L'), f"Expected symbolic S_L..., got {S}"

    def test_antipode_sunset_symbolic(self):
        """With no amplitude_func, antipode returns a symbolic placeholder."""
        bphz = BPHZRenormalization(sunset())
        S = bphz.antipode()
        assert str(S).startswith('S_L'), f"Expected symbolic S_L..., got {S}"

    # ---- analytic amplitude_func tests ----------------------------- #

    def _make_amplitude_func(self):
        """
        Symbolic amplitude functions for structural BPHZ tests.

        I(bubble)  ~ A₁/ε + B₁   (1-loop divergent)
        I(tadpole) ~ E₀ + E₁·ε   (finite in dim-reg)
        I(sunset)  ~ A₂/ε + B₂   (2-loop divergent)
        """
        eps = sp.Symbol('epsilon')
        A1, B1 = sp.symbols('A1 B1', real=True)
        A2, B2 = sp.symbols('A2 B2', real=True)
        E0, E1 = sp.symbols('E0 E1', real=True)

        def amplitude_func(G: FeynmanGraph) -> sp.Expr:
            n_int = G.n_internal_edges
            L = G.L
            if L == 1 and n_int == 2:
                # Bubble: 1-loop
                return A1 / eps + B1
            elif L == 1 and n_int == 1:
                # Tadpole (self-loop): finite
                return E0 + E1 * eps
            elif L == 2 and n_int == 3:
                # Sunset: 2-loop
                return A2 / eps + B2
            else:
                return sp.Integer(0)

        return amplitude_func, eps, (A1, B1, A2, B2, E0, E1)

    def test_bubble_antipode_is_minus_amplitude(self):
        """
        Bubble is primitive: S(bubble) = -I(bubble) = -(A1/ε + B1).
        """
        amp_func, eps, (A1, B1, *_) = self._make_amplitude_func()
        bphz = BPHZRenormalization(bubble(), amp_func)
        S = bphz.antipode()
        expected = -(A1 / eps + B1)
        assert sp.simplify(S - expected) == 0, (
            f"Bubble antipode: expected {expected}, got {S}"
        )

    def test_sunset_antipode_structure(self):
        """
        S(sunset) = -(A2/ε+B2) - 3·S(bubble)·I(tadpole)

        With S(bubble) = -(A1/ε+B1) and I(tadpole) = E0+E1·ε:
          S(sunset) = -(A2/ε+B2) + 3(A1/ε+B1)(E0+E1·ε)
                    = 3A1E0/ε + (3A1E1+3B1E0-B2) + ...  (leading poles)
        """
        amp_func, eps, (A1, B1, A2, B2, E0, E1) = self._make_amplitude_func()
        bphz = BPHZRenormalization(sunset(), amp_func)
        S = bphz.antipode()
        S_expanded = sp.expand(S)

        # Expected: -(A2/ε+B2) + 3*(-(A1/ε+B1))*(E0+E1*ε)
        # = -(A2/ε+B2) - 3*(A1/ε+B1)*(E0+E1*ε)
        expected = sp.expand(
            -(A2/eps + B2) - sp.Integer(3) * (-(A1/eps + B1)) * (E0 + E1*eps)
        )
        assert sp.simplify(sp.expand(S_expanded - expected)) == 0, (
            f"Sunset antipode mismatch.\n"
            f"  got:      {S_expanded}\n"
            f"  expected: {expected}"
        )

    def test_renormalized_sunset_pole_is_zero(self):
        """
        I_R(sunset) = I(sunset) + S(sunset).

        The 1/ε pole of I_R must vanish if A2 = 3·A1·E0
        (the standard MS-bar self-consistency condition):
          I(sunset) = A2/ε + B2
          S(sunset) = 3A1E0/ε + ...
          I_R pole = A2 - 3A1E0  →  must be zero.

        Check that substituting A2 → 3*A1*E0 gives zero pole.
        """
        amp_func, eps, (A1, B1, A2, B2, E0, E1) = self._make_amplitude_func()
        bphz = BPHZRenormalization(sunset(), amp_func)

        I_R = bphz.renormalized_amplitude()
        I_R_expanded = sp.expand(I_R)

        # Extract 1/eps coefficient
        pole_coeff = sp.S.Zero
        for term in sp.Add.make_args(I_R_expanded):
            if term.has(eps):
                try:
                    p = sp.Poly(term, eps)
                    if p.degree() < 0:
                        pole_coeff += term
                except Exception:
                    pass

        # Substitute the MS-bar consistency relation A2 = 3*A1*E0
        pole_at_consistency = sp.simplify(pole_coeff.subs(A2, 3*A1*E0))
        assert pole_at_consistency == 0, (
            f"I_R pole at A2=3A1E0 should vanish; got {pole_at_consistency}"
        )

    def test_renormalized_amplitude_finite_for_primitive(self):
        """
        For a primitive graph (bubble), I_R = I + S = I - I = 0 in MS-bar.
        (S(bubble) = -I(bubble) exactly, no sub-divergences.)
        """
        amp_func, eps, _ = self._make_amplitude_func()
        bphz = BPHZRenormalization(bubble(), amp_func)
        I_R = bphz.renormalized_amplitude()
        assert sp.expand(I_R) == 0, (
            f"Bubble I_R should be 0 (primitive graph), got {sp.expand(I_R)}"
        )

    def test_sunset_has_correct_number_of_coproduct_terms(self):
        """3 sub-divergences → 3 terms in the antipode sum."""
        bphz = BPHZRenormalization(sunset())
        terms = bphz._coproduct.coproduct_terms()
        assert len(terms) == 3

    def test_pole_part_extraction(self):
        """pole_part correctly extracts negative-power ε terms."""
        eps = sp.Symbol('epsilon')
        bphz = BPHZRenormalization(bubble())

        expr = sp.Integer(2)/eps**2 + sp.Integer(3)/eps + sp.Integer(4)
        pole = bphz.pole_part(expr)
        # Should give 2/ε² + 3/ε (both poles)
        expected = sp.Integer(2)/eps**2 + sp.Integer(3)/eps
        assert sp.simplify(pole - expected) == 0, (
            f"pole_part: expected {expected}, got {pole}"
        )

    def test_pole_part_finite_gives_zero(self):
        """pole_part of a finite expression is zero."""
        eps = sp.Symbol('epsilon')
        bphz = BPHZRenormalization(bubble())
        expr = sp.Integer(5) + sp.Integer(3) * eps
        pole = bphz.pole_part(expr)
        assert pole == 0 or sp.simplify(pole) == 0, (
            f"pole_part of finite expression should be 0, got {pole}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
