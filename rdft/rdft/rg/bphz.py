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

In the minimal subtraction (MS-bar) scheme, the counterterm C(γ) is the
pole part of the amplitude:

    C(γ) = -Res_{ε=0}[I(γ; d_c - ε)]

This gives the renormalisation Z-factors.

Implementation:
  - A 1PI sub-divergence γ of Γ is identified by its vertex subset
  - Graph contraction Γ/γ: collapse all vertices of γ to a single vertex
    in the incidence matrix representation
  - The antipode is computed recursively on the Hopf algebra grading
    (graded by |V| or by loop order)

Mathematical reference:
    Connes & Kreimer (1998) Commun. Math. Phys. 199:203-242
    Amarteifio (2019) §2.6
    Yeats (2017) Chapter 4
"""

from __future__ import annotations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import sympy as sp

from ..graphs.incidence import FeynmanGraph


class SubDivergence:
    """
    A 1PI sub-divergence γ of a Feynman graph Γ.

    Parameters
    ----------
    vertex_set : frozenset of vertex indices in Γ that form γ
    parent : the parent FeynmanGraph Γ
    """

    def __init__(self, vertex_set: FrozenSet[int], parent: FeynmanGraph):
        self.vertex_set = vertex_set
        self.parent = parent

    @property
    def induced_edges(self) -> List[int]:
        """
        Internal edges of Γ that have both endpoints in vertex_set.
        These are the internal edges of γ.
        """
        result = []
        for i, (src, tgt, is_ext) in enumerate(self.parent.edges):
            if not is_ext and src in self.vertex_set and tgt in self.vertex_set:
                result.append(i)
        return result

    @property
    def is_1pi(self) -> bool:
        """Check if the induced subgraph is 1PI (no bridges)."""
        if len(self.induced_edges) == 0:
            return False
        # Build a sub-FeynmanGraph and check
        n_verts = len(self.vertex_set)
        vmap = {v: i for i, v in enumerate(sorted(self.vertex_set))}
        edges = []
        for e_idx in self.induced_edges:
            src, tgt, _ = self.parent.edges[e_idx]
            edges.append((vmap[src], vmap[tgt], False))
        if not edges:
            return False
        sub = FeynmanGraph(n_verts - 1, edges)
        return sub.is_1pi()

    @property
    def betti_number(self) -> int:
        """Loop number L = |E_int| - |V_int| + 1."""
        n_e = len(self.induced_edges)
        n_v = len(self.vertex_set)
        return max(0, n_e - n_v + 1)

    def contracted_graph(self) -> FeynmanGraph:
        """
        Contract γ to a single vertex in Γ.

        Γ/γ: replace all vertices in vertex_set with a single new vertex.
        Edges internal to γ are removed; edges between γ and Γ\γ
        become new edges connected to the contracted vertex.

        (Connes-Kreimer: quotient graph)
        """
        contracted_v = min(self.vertex_set)  # representative vertex
        other_vertices = sorted(
            v for v in range(self.parent.n_vertices_int)
            if v not in self.vertex_set
        )

        # Build new vertex set: contracted_v + other_vertices + v_∞
        new_v_map = {}
        new_idx = 0
        for v in sorted(self.vertex_set):
            new_v_map[v] = 0  # all collapse to vertex 0
        for v in other_vertices:
            new_v_map[v] = new_idx + 1
            new_idx += 1
        new_v_map[self.parent.v_inf] = new_idx + 1  # v_∞ remains

        n_new_int = new_idx + 1  # excluding v_∞

        # Build new edge list: skip internal edges of γ
        new_edges = []
        internal_gamma = set(self.induced_edges)

        for e_idx, (src, tgt, is_ext) in enumerate(self.parent.edges):
            if e_idx in internal_gamma:
                continue  # removed
            new_src = new_v_map.get(src, src)
            new_tgt = new_v_map.get(tgt, tgt)
            new_edges.append((new_src, new_tgt, is_ext))

        return FeynmanGraph(n_new_int, new_edges)


class CoproductMap:
    """
    The Connes-Kreimer coproduct for a Feynman graph.

    Δ(Γ) = Γ ⊗ 1 + 1 ⊗ Γ + Σ_{γ} γ ⊗ Γ/γ

    where the sum is over all non-empty, proper, 1PI sub-divergences γ.

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
        All non-trivial 1PI sub-divergences of self.graph.

        A sub-divergence is a proper non-empty subset of internal vertices
        that induces a 1PI subgraph with L ≥ 1 loops.
        """
        if self._subdivergences is not None:
            return self._subdivergences

        n = self.graph.n_vertices_int
        result = []

        # Enumerate all non-empty proper subsets of internal vertices
        # (For large graphs, this is exponential — practical limit ~10 vertices)
        for mask in range(1, 2**n - 1):  # exclude empty and full sets
            vset = frozenset(i for i in range(n) if mask & (1 << i))
            sub = SubDivergence(vset, self.graph)
            if sub.betti_number >= 1 and sub.is_1pi:
                result.append(sub)

        self._subdivergences = result
        return self._subdivergences

    def coproduct_terms(self) -> List[Tuple[SubDivergence, FeynmanGraph]]:
        """
        Return the non-primitive coproduct terms: [(γ, Γ/γ), ...]

        Each pair is a (sub-divergence, contracted graph).
        """
        return [(sub, sub.contracted_graph()) for sub in self.subdivergences]


class BPHZRenormalization:
    """
    BPHZ renormalization of a Feynman graph via the Connes-Kreimer antipode.

    The renormalized amplitude is:
        I_R(Γ) = I(Γ) + Σ_γ C(γ) · I(Γ/γ)

    where C(γ) = -Pole[I(γ)] in MS-bar scheme.

    The antipode S satisfies:
        S(Γ) = -Γ - Σ_{γ ⊊ Γ} S(γ) · (Γ/γ)

    (recursive on the Hopf grading = loop order)

    Parameters
    ----------
    graph : FeynmanGraph
    amplitude_func : callable(graph) → sympy expr
        Function that computes I(G; ε) for a given graph as a Laurent
        series in ε.
    """

    def __init__(self, graph: FeynmanGraph, amplitude_func=None):
        self.graph = graph
        self.amplitude_func = amplitude_func
        self._coproduct = CoproductMap(graph)

    def pole_part(self, expr: sp.Expr) -> sp.Expr:
        """
        Extract the pole part of a Laurent series in ε.
        In MS-bar: counterterm = -Res_{ε=0}[I(γ)]
        """
        eps = sp.Symbol('epsilon')
        series = sp.series(expr, eps, 0, 1)
        # Extract terms with negative powers of ε
        pole = sp.S.Zero
        for term in sp.Add.make_args(series):
            if term.has(eps) and sp.Poly(term, eps).degree() < 0:
                pole += term
        return pole

    def counterterm(self, subgraph: FeynmanGraph) -> sp.Expr:
        """C(γ) = -Pole[I(γ; ε)]"""
        if self.amplitude_func is None:
            eps = sp.Symbol('epsilon')
            return sp.Symbol(f'C_gamma_{subgraph.n_vertices_int}L{subgraph.L}')
        amplitude = self.amplitude_func(subgraph)
        return -self.pole_part(amplitude)

    def antipode(self, graph: FeynmanGraph = None) -> sp.Expr:
        """
        Compute S(Γ) recursively.

        S(Γ) = -I(Γ) - Σ_{γ ⊊ Γ, 1PI} S(γ)·I(Γ/γ)

        For tree-level (L=0): S(Γ) = -I(Γ) (no subdivergences)
        """
        G = graph or self.graph
        cmap = CoproductMap(G)

        if self.amplitude_func is None:
            return sp.Symbol(f'S_Gamma_L{G.L}')

        I_G = self.amplitude_func(G)

        if not cmap.subdivergences:
            # No non-trivial subdivergences: antipode = -amplitude
            return -I_G

        antipode_sum = sp.S.Zero
        for sub, contracted in cmap.coproduct_terms():
            # Recursively compute S(γ)
            sub_graph = FeynmanGraph(
                len(sub.vertex_set),
                [(self.graph.edges[e][0], self.graph.edges[e][1], False)
                 for e in sub.induced_edges]
            )
            S_gamma = self.antipode(sub_graph)
            I_contracted = self.amplitude_func(contracted)
            antipode_sum += S_gamma * I_contracted

        return sp.expand(-I_G - antipode_sum)

    def renormalized_amplitude(self) -> sp.Expr:
        """
        I_R(Γ) = I(Γ) + S(Γ) in the forest formula sense.

        In practice under MS-bar:
            I_R(Γ) = I(Γ) - Pole[I(Γ)] + ...
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

        # The coefficient of 1/ε gives Z_λ at one loop
        pole_coeff = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(I_R_series)):
            if term.has(eps):
                power = sp.Poly(term, eps).degree()
                if power == -1:
                    pole_coeff += term * eps

        z1 = sp.simplify(pole_coeff / coupling)
        return sp.Integer(1) + z1 * coupling / eps
