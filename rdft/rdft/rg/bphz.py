"""
rdft.rg.bphz
============
BPHZ renormalisation via the Connes-Kreimer Hopf algebra.

The BPHZ renormalisation of a Feynman graph Γ is:

    R(Γ) = Γ + Σ_{∅ ≠ γ ⊂ Γ, 1PI} (-C(γ)) · Γ/γ

where C(γ) is the counterterm (Taylor-subtracted amplitude) for the
sub-divergence γ, and Γ/γ is the contracted graph.

This is the antipode S of the Connes-Kreimer Hopf algebra H_R:

    S(Γ) = -Γ - Σ_{γ ⊊ Γ} S(γ) · (Γ/γ)

The coproduct decomposes Γ into sub-divergences:

    Δ(Γ) = Γ ⊗ 1 + 1 ⊗ Γ + Σ_{∅ ≠ γ ⊊ Γ, 1PI} γ ⊗ Γ/γ

KEY DESIGN CHOICE — edge-subset enumeration.

  A sub-divergence is defined by a proper non-empty SUBSET OF INTERNAL EDGES
  that induces a 1PI subgraph with L ≥ 1 loops.  Vertex-subset enumeration
  (the previous implementation) misses sub-divergences in multi-edge graphs:
  for the sunset (3 parallel edges between 2 vertices), the bubble
  sub-divergence uses any 2 of the 3 edges — its vertex set equals the full
  graph vertex set, so vertex enumeration finds nothing.

  Edge-subset enumeration handles this correctly:
    sunset:   3 sub-divergences (each pair of 2 edges forms a bubble)
    bubble:   0 sub-divergences (primitive)
    triangle: 0 sub-divergences (primitive)

  For a graph with n internal edges the enumeration is O(2^n); practical
  for all graphs with n ≤ ~20 (sufficient for 3-loop and beyond).

In the minimal subtraction (MS-bar) scheme, the counterterm C(γ) is the
pole part of the amplitude:

    C(γ) = -Res_{ε=0}[I(γ; d_c - ε)]

This gives the renormalisation Z-factors.

Mathematical reference:
    Connes & Kreimer (1998) Commun. Math. Phys. 199:203-242
    Amarteifio (2019) §2.6
    Yeats (2017) Chapter 4
    Bogner & Weinzierl (2010) arXiv:1003.1154
"""

from __future__ import annotations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import sympy as sp

from ..graphs.incidence import FeynmanGraph


class SubDivergence:
    """
    A 1PI sub-divergence γ of a Feynman graph Γ.

    Defined by a **subset of internal edges** of Γ whose induced subgraph
    (on the vertices incident to those edges) is 1PI with L ≥ 1 loops.

    Parameters
    ----------
    edge_set : frozenset of internal edge indices (into parent.edges)
    parent   : the parent FeynmanGraph Γ
    """

    def __init__(self, edge_set: FrozenSet[int], parent: FeynmanGraph):
        self.edge_set = edge_set
        self.parent = parent

    # ------------------------------------------------------------------ #
    #  Structural properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def induced_edges(self) -> List[int]:
        """Sorted list of edge indices in this sub-divergence."""
        return sorted(self.edge_set)

    @property
    def vertex_set(self) -> FrozenSet[int]:
        """
        Internal vertices incident to at least one edge in edge_set.
        (External vertex v_∞ is excluded.)
        """
        verts: Set[int] = set()
        n_int_v = self.parent.n_vertices_int
        for e_idx in self.edge_set:
            src, tgt, _ = self.parent.edges[e_idx]
            if src < n_int_v:
                verts.add(src)
            if tgt < n_int_v:
                verts.add(tgt)
        return frozenset(verts)

    @property
    def betti_number(self) -> int:
        """
        Loop number L = |edges| − |vertices| + 1  (for a connected subgraph).
        """
        return max(0, len(self.edge_set) - len(self.vertex_set) + 1)

    @property
    def is_1pi(self) -> bool:
        """
        True iff the induced edge-subgraph is 1PI:
          (a) connected on its vertex set, and
          (b) has no bridge (removing any single edge keeps it connected).

        Uses NetworkX on the internal-vertex subgraph only.
        """
        import networkx as nx

        v_set = self.vertex_set
        if not v_set:
            return False

        vmap = {v: i for i, v in enumerate(sorted(v_set))}

        G_sub = nx.MultiGraph()
        G_sub.add_nodes_from(range(len(v_set)))

        for i, e_idx in enumerate(self.induced_edges):
            src, tgt, _ = self.parent.edges[e_idx]
            if src in vmap and tgt in vmap:
                G_sub.add_edge(vmap[src], vmap[tgt], key=i)

        if G_sub.number_of_edges() == 0:
            return False

        # (a) connected on its own vertex set
        if not nx.is_connected(G_sub):
            return False

        # (b) no bridge: every edge must lie on a cycle
        for s, t, key in list(G_sub.edges(keys=True)):
            H = G_sub.copy()
            H.remove_edge(s, t, key=key)
            if not nx.is_connected(H):
                return False  # found a bridge

        return True

    # ------------------------------------------------------------------ #
    #  Conversion and contraction                                          #
    # ------------------------------------------------------------------ #

    def to_graph(self) -> FeynmanGraph:
        """
        Build a standalone FeynmanGraph for this sub-divergence.

        Vertices are renumbered 0...|V_γ|-1; all edges become internal
        (no external legs — used for amplitude computation of γ alone).
        """
        v_set = self.vertex_set
        vmap = {v: i for i, v in enumerate(sorted(v_set))}

        edges = []
        for e_idx in self.induced_edges:
            src, tgt, _ = self.parent.edges[e_idx]
            if src in vmap and tgt in vmap:
                edges.append((vmap[src], vmap[tgt], False))

        return FeynmanGraph(len(v_set), edges)

    def contracted_graph(self) -> FeynmanGraph:
        """
        Γ/γ: contract γ to a single super-vertex in Γ.

        Algorithm:
          1. All internal vertices of γ collapse to super-vertex 0.
          2. Remaining internal vertices of Γ become 1, 2, ...
          3. v_∞ becomes n_new_int.
          4. Edges in edge_set are deleted.
          5. Edges NOT in edge_set are remapped through the new vertex indices.

        Result is a FeynmanGraph with one fewer free parameter per loop of γ.
        For the sunset contracted by a bubble: the contracted graph is a tadpole
        (single vertex with a self-loop + two external legs).
        """
        V_gamma = self.vertex_set

        # Vertices of Γ not in γ (sorted for canonical ordering)
        other_int_verts = sorted(
            v for v in range(self.parent.n_vertices_int)
            if v not in V_gamma
        )

        # New vertex map:  V_gamma → 0,  others → 1, 2, ...,  v_∞ → n_new_int
        vmap: Dict[int, int] = {}
        for v in V_gamma:
            vmap[v] = 0
        for new_idx, v in enumerate(other_int_verts, start=1):
            vmap[v] = new_idx
        n_new_int = 1 + len(other_int_verts)
        vmap[self.parent.v_inf] = n_new_int  # v_∞

        # Build contracted edge list (skip edges that were in γ)
        new_edges = []
        for e_idx, (src, tgt, is_ext) in enumerate(self.parent.edges):
            if e_idx in self.edge_set:
                continue  # contracted away
            new_src = vmap.get(src, src)
            new_tgt = vmap.get(tgt, tgt)
            new_edges.append((new_src, new_tgt, is_ext))

        return FeynmanGraph(n_new_int, new_edges)


class CoproductMap:
    """
    The Connes-Kreimer coproduct for a Feynman graph.

    Δ(Γ) = Γ ⊗ 1 + 1 ⊗ Γ + Σ_{γ} γ ⊗ Γ/γ

    where the sum is over all proper non-empty 1PI edge-subsets γ with
    L(γ) ≥ 1.

    Parameters
    ----------
    graph : FeynmanGraph
    """

    def __init__(self, graph: FeynmanGraph):
        self.graph = graph
        self._subdivergences: Optional[List[SubDivergence]] = None

    @property
    def subdivergences(self) -> List[SubDivergence]:
        """
        All proper 1PI edge-subset sub-divergences with L ≥ 1.

        Enumerates all 2^{n_int} − 2 non-empty proper subsets of internal
        edges.  For each subset, checks:
          (a) betti_number ≥ 1  (at least one loop)
          (b) is_1pi            (connected, no bridge)

        Complexity: O(2^{n_int} × n_int²) — feasible for n_int ≤ ~20.
        """
        if self._subdivergences is not None:
            return self._subdivergences

        int_edges = self.graph.internal_edge_indices
        n_int = len(int_edges)
        result: List[SubDivergence] = []

        full_mask = (1 << n_int) - 1  # all-ones: the full edge set

        for mask in range(1, full_mask):  # 1 .. 2^n−2  (exclude empty and full)
            edge_subset = frozenset(
                int_edges[i] for i in range(n_int) if mask & (1 << i)
            )
            sub = SubDivergence(edge_subset, self.graph)
            if sub.betti_number >= 1 and sub.is_1pi:
                result.append(sub)

        self._subdivergences = result
        return self._subdivergences

    def coproduct_terms(self) -> List[Tuple[SubDivergence, FeynmanGraph]]:
        """
        Return the non-primitive coproduct terms: [(γ, Γ/γ), ...]

        Each pair is a (sub-divergence as SubDivergence, contracted graph as
        FeynmanGraph).
        """
        return [(sub, sub.contracted_graph()) for sub in self.subdivergences]

    def summary(self) -> str:
        """Human-readable summary of all sub-divergences."""
        lines = [f'CoproductMap for {self.graph}']
        subs = self.subdivergences
        lines.append(f'  {len(subs)} sub-divergence(s):')
        for s in subs:
            contracted = s.contracted_graph()
            lines.append(
                f'    edges {s.induced_edges}  '
                f'(V={s.vertex_set}, L={s.betti_number})  →  '
                f'contracted: {contracted}'
            )
        return '\n'.join(lines)


class BPHZRenormalization:
    """
    BPHZ renormalization of a Feynman graph via the Connes-Kreimer antipode.

    The renormalized amplitude is:
        I_R(Γ) = I(Γ) + Σ_γ C(γ) · I(Γ/γ)

    where C(γ) = -Pole[I(γ)] in MS-bar scheme.

    The antipode S satisfies:
        S(Γ) = -I(Γ) - Σ_{γ ⊊ Γ} S(γ)·I(Γ/γ)

    (recursive on the Hopf grading = loop order)

    Parameters
    ----------
    graph          : FeynmanGraph
    amplitude_func : callable(graph) → sympy Laurent series in ε, or None.
                     If None, symbolic placeholders are used.
    """

    def __init__(self, graph: FeynmanGraph, amplitude_func=None):
        self.graph = graph
        self.amplitude_func = amplitude_func
        self._coproduct = CoproductMap(graph)

    def pole_part(self, expr: sp.Expr) -> sp.Expr:
        """
        Extract the pole part of a Laurent series in ε (MS-bar counterterm).

        Returns the sum of all terms with negative powers of ε.

        Finds the epsilon symbol by name from the expression's free symbols
        so that assumption mismatches (positive=True vs bare) don't hide poles.
        """
        # Find epsilon in the expression (may carry positive=True assumption)
        eps = None
        for sym in expr.free_symbols:
            if sym.name == 'epsilon':
                eps = sym
                break
        if eps is None:
            return sp.S.Zero  # no epsilon → no poles

        pole = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(expr)):
            if isinstance(term, sp.Order):
                continue  # skip O(...) truncation terms
            power = term.as_powers_dict().get(eps, 0)
            if power < 0:
                pole += term
        return pole

    def counterterm(self, subgraph: FeynmanGraph) -> sp.Expr:
        """C(γ) = -Pole[I(γ; ε)]"""
        if self.amplitude_func is None:
            return sp.Symbol(f'C_L{subgraph.L}_E{subgraph.n_internal_edges}')
        amplitude = self.amplitude_func(subgraph)
        return -self.pole_part(amplitude)

    def antipode(self, graph: FeynmanGraph = None) -> sp.Expr:
        """
        Compute S(Γ) recursively (Connes-Kreimer antipode).

        S(Γ) = -I(Γ) - Σ_{γ ⊊ Γ, 1PI} S(γ) · I(Γ/γ)

        For primitive graphs (no sub-divergences): S(Γ) = -I(Γ).
        """
        G = graph if graph is not None else self.graph
        cmap = CoproductMap(G)

        if self.amplitude_func is None:
            return sp.Symbol(f'S_L{G.L}_E{G.n_internal_edges}')

        I_G = self.amplitude_func(G)

        if not cmap.subdivergences:
            return -I_G

        antipode_sum = sp.S.Zero
        for sub, contracted in cmap.coproduct_terms():
            S_gamma = self.antipode(sub.to_graph())
            I_contracted = self.amplitude_func(contracted)
            antipode_sum += S_gamma * I_contracted

        return sp.expand(-I_G - antipode_sum)

    def renormalized_amplitude(self) -> sp.Expr:
        """
        I_R(Γ) = I(Γ) + S(Γ)

        Under MS-bar this equals I(Γ) with all 1/ε poles from
        subdivergences removed.
        """
        if self.amplitude_func is None:
            return sp.Symbol('I_R')

        I_G = self.amplitude_func(self.graph)
        S_G = self.antipode()
        return sp.expand(I_G + S_G)

    def z_factor(self, coupling: sp.Symbol) -> sp.Expr:
        """
        Z-factor for renormalization: λ_0 = μ^ε · Z_λ · λ.

        Z_λ = 1 + Σ_n (z_n/ε^n) λ^n

        The pole coefficients z_n are extracted from the renormalized
        amplitude by requiring I_R to be finite.
        """
        eps = sp.Symbol('epsilon')
        I_R = self.renormalized_amplitude()
        I_R_series = sp.series(I_R, eps, 0, 1)

        pole_coeff = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(I_R_series)):
            if term.has(eps):
                power = sp.Poly(term, eps).degree()
                if power == -1:
                    pole_coeff += term * eps

        z1 = sp.simplify(pole_coeff / coupling)
        return sp.Integer(1) + z1 * coupling / eps
