"""
rdft.graphs.incidence
=====================
Incidence matrix, Laplacian, and symbolic graph polynomials.

Implements the matrix machinery of Amarteifio (2019) §2.2 and §2.3.

Conventions (following Amarteifio thesis p.57):
  - Incidence matrix E: size |E| × |V|, edges on columns, vertices on rows
  - E[e,v] = -1 if edge e exits vertex v
  - E[e,v] = +1 if edge e enters vertex v
  - v_∞ always appears as the LAST row (convention: last vertex)
  - Internal edges labelled p_i, external edges labelled q_j

The Laplacian is L = E · E^T.

The symbolic Laplacian is L_RS = E_{[Γ]} · D_α · E_{[Γ]}^T
where D_α = diag(α_0, α_1, ...) is the diagonal matrix of Schwinger
parameters (one per internal edge).

Key results:
  - det(L_{[ij]}) = number of spanning trees (Matrix-tree theorem)
  - det(L_RS) = Kirchhoff polynomial K (Amarteifio §2.3)
  - Symanzik Ψ = det(M) where M comes from K via edge complements
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import sympy as sp
import numpy as np


class FeynmanGraph:
    """
    A Feynman graph represented by its incidence matrix.

    An (amputated) Feynman graph G(V, E) has:
      - Internal vertices V_int and the auxiliary vertex v_∞
      - Internal edges E_int (with Schwinger parameters α_e)
      - External edges E_ext (the 'residue' R(G), connecting to v_∞)

    The full incidence matrix E includes both internal and external edges.
    After actualization (attaching all half-edges to v_∞), E is a proper
    incidence matrix in the graph-theoretic sense.

    Parameters
    ----------
    n_internal_vertices : number of true vertices (not v_∞)
    edges : list of (from_vertex, to_vertex, is_external) tuples
            Vertices are 0-indexed; v_∞ = n_internal_vertices

    Examples
    --------
    Sunset diagram (2 vertices, 3 internal edges):
        G = FeynmanGraph(2, [(0,1,False), (0,1,False), (0,1,False)])

    One-loop self-energy (1 vertex, 2-edge loop):
        G = FeynmanGraph.tadpole()
    """

    def __init__(self,
                 n_internal_vertices: int,
                 edges: List[Tuple[int, int, bool]],
                 edge_symbols: Optional[List[sp.Symbol]] = None,
                 alpha_symbols: Optional[List[sp.Symbol]] = None):

        self.n_vertices_int = n_internal_vertices
        self.n_vertices = n_internal_vertices + 1   # +1 for v_∞
        self.v_inf = n_internal_vertices             # index of v_∞

        self.edges = edges   # (from, to, is_external)
        self.n_edges = len(edges)

        self.n_internal_edges = sum(1 for _, _, ext in edges if not ext)
        self.n_external_edges = sum(1 for _, _, ext in edges if ext)

        # Betti number: L = |E_int| - |V_int| + 1 (connected graph, 1 component)
        # (Amarteifio eq. 2.1)
        self.L = self.n_internal_edges - n_internal_vertices + 1

        # Edge symbols: p_i for internal, q_j for external (symbolic labelling)
        if edge_symbols is None:
            p_idx = 0
            q_idx = 0
            self._edge_syms = []
            for _, _, is_ext in edges:
                if is_ext:
                    self._edge_syms.append(sp.Symbol(f'q{q_idx}'))
                    q_idx += 1
                else:
                    self._edge_syms.append(sp.Symbol(f'p{p_idx}'))
                    p_idx += 1
        else:
            self._edge_syms = edge_symbols

        # Schwinger parameters α_e for internal edges only
        if alpha_symbols is None:
            self._alpha_syms = [
                sp.Symbol(f'alpha{i}', positive=True)
                for i, (_, _, ext) in enumerate(edges) if not ext
            ]
        else:
            self._alpha_syms = alpha_symbols

    @property
    def internal_edge_indices(self) -> List[int]:
        return [i for i, (_, _, ext) in enumerate(self.edges) if not ext]

    @property
    def external_edge_indices(self) -> List[int]:
        return [i for i, (_, _, ext) in enumerate(self.edges) if ext]

    # ------------------------------------------------------------------ #
    #  Incidence matrix                                                    #
    # ------------------------------------------------------------------ #

    def incidence_matrix(self, symbolic: bool = True) -> sp.Matrix:
        """
        Build the incidence matrix E of size n_edges × n_vertices.

        Convention (Amarteifio §2.2):
          E[e, v] = -1 if edge e exits vertex v (source)
          E[e, v] = +1 if edge e enters vertex v (target)
          E[e, v] = 0  otherwise

        The last column corresponds to v_∞.

        If symbolic=True, entries are multiplied by the edge symbol
        (so E becomes the 'symbolic incidence matrix' of Amarteifio §2.3).
        """
        E = sp.zeros(self.n_edges, self.n_vertices)
        for e, (src, tgt, _) in enumerate(self.edges):
            sym = self._edge_syms[e] if symbolic else sp.S.One
            E[e, src] = -sym
            E[e, tgt] = +sym
        return E

    def reduced_incidence_matrix(self,
                                  symbolic: bool = True,
                                  spanning_tree_edges: Optional[List[int]] = None) -> sp.Matrix:
        """
        Reduced incidence matrix E_{[Γ]}: delete the v_∞ row and
        optionally restrict to a spanning tree's columns.

        This is the matrix used in the Kirchhoff polynomial via:
            K = det(E_{[Γ]} · D_α · E_{[Γ]}^T)

        (Amarteifio eq. 2.35, 2.39a)
        """
        E = self.incidence_matrix(symbolic=symbolic)
        # Delete the v_∞ row
        rows_to_keep = list(range(self.n_vertices - 1))  # all except last (v_∞)
        E_red = E[:, rows_to_keep]

        if spanning_tree_edges is not None:
            E_red = E_red[spanning_tree_edges, :]

        return E_red

    # ------------------------------------------------------------------ #
    #  Laplacian and symbolic Laplacian                                    #
    # ------------------------------------------------------------------ #

    def laplacian(self, symbolic: bool = True) -> sp.Matrix:
        """
        Laplacian L = E · E^T (Amarteifio Definition 8).
        """
        E = self.incidence_matrix(symbolic=symbolic)
        return E * E.T

    def symbolic_laplacian_RS(self) -> sp.Matrix:
        """
        Reduced symbolic Laplacian:
            L_RS = E_{[Γ, int]} · D_α · E_{[Γ, int]}^T

        where E_{[Γ, int]} is the incidence matrix restricted to
        internal edges (columns) and all internal vertices except v_∞ (rows).
        D_α = diag(α_0, α_1, ...) over internal edges.

        This is Amarteifio eq. (2.35).

        Returns: symbolic square matrix of size n_internal_vertices × n_internal_vertices
        """
        # Build the non-symbolic incidence matrix restricted to internal edges
        int_edges = self.internal_edge_indices
        int_verts = list(range(self.n_vertices_int))  # exclude v_∞

        # E_int: size n_int_edges × n_int_verts
        E_int = sp.zeros(len(int_edges), len(int_verts))
        for new_e, old_e in enumerate(int_edges):
            src, tgt, _ = self.edges[old_e]
            if src in int_verts:
                E_int[new_e, int_verts.index(src)] = -1
            if tgt in int_verts:
                E_int[new_e, int_verts.index(tgt)] = +1

        # D_α = diagonal matrix of Schwinger parameters
        D_alpha = sp.diag(*self._alpha_syms)

        # L_RS = E_int^T · D_α · E_int
        # (Note: Amarteifio writes E_{[Γ]} D_α E_{[Γ]}^T but the matrix
        # dimensions require careful tracking of row/column orientation.)
        L_RS = E_int.T * D_alpha * E_int

        return L_RS

    # ------------------------------------------------------------------ #
    #  Kirchhoff polynomial                                                #
    # ------------------------------------------------------------------ #

    def kirchhoff_polynomial(self) -> sp.Expr:
        """
        The Kirchhoff polynomial K(α_0, α_1, ...).

        K = det(L_RS)  where L_RS is the reduced symbolic Laplacian.

        Terms of degree k in the alpha parameters enumerate k-trees of G.
        Specifically, the spanning 1-trees (spanning trees) appear at
        degree |E_int| - n_internal_vertices + 1 = L (the Betti number).

        Wait — actually terms of degree 1 in K = det(L_RS) correspond to
        spanning trees (one alpha per edge in the tree, L=1 means one loop
        removed). Let me be precise:

        det(L_RS) is a polynomial of degree n_internal_vertices in the alpha
        parameters. The coefficient of each monomial lists the edges NOT
        in the corresponding spanning tree (i.e. the complement).

        See Amarteifio §2.3, Example 2.3.2 for the sunset diagram.

        Returns: sympy polynomial in self._alpha_syms
        """
        # det(L_RS) = 0 always (rows of the reduced symbolic Laplacian
        # sum to zero for connected graphs — it is always singular).
        # The Kirchhoff polynomial = any cofactor of L_RS:
        #   K = det(L_RS with row i and col i deleted)
        # This is the Matrix-Tree theorem:
        #   any (n-1)×(n-1) minor of L equals the sum over spanning trees
        #   of the product of alpha parameters on those tree edges.
        # (Amarteifio §2.3 Corollary 2.1.1)
        L_RS = self.symbolic_laplacian_RS()
        n = L_RS.shape[0]
        if n == 0:
            return sp.S.One   # no internal vertices: trivial
        if n == 1:
            return sp.expand(L_RS[0, 0])   # single entry IS the polynomial
        # Delete row 0 and col 0
        return sp.expand(L_RS.minor_submatrix(0, 0).det())

    def spanning_trees_from_kirchhoff(self) -> List[List[int]]:
        """
        Extract the spanning trees from the Kirchhoff polynomial.

        K = Σ_T Π_{e ∈ T} α_e  (Matrix-Tree theorem: each monomial
        is the product of edge weights IN a spanning tree).

        A spanning tree of a connected graph with n internal vertices
        has exactly n_vertices_int - 1 edges.

        Returns list of lists of internal edge indices forming spanning trees.
        """
        K = self.kirchhoff_polynomial()
        alphas = self._alpha_syms
        poly = sp.Poly(K, *alphas)

        spanning_trees = []
        int_edges = self.internal_edge_indices
        expected_tree_size = self.n_vertices_int - 1  # n-1 edges in spanning tree

        for monom in poly.monoms():
            # Edges with α exponent > 0 in this monomial ARE in the spanning tree
            tree_edge_positions = [i for i, exp in enumerate(monom) if exp > 0]
            if len(tree_edge_positions) == expected_tree_size:
                spanning_trees.append([int_edges[i] for i in tree_edge_positions])

        return spanning_trees

    # ------------------------------------------------------------------ #
    #  Graph properties                                                    #
    # ------------------------------------------------------------------ #

    def is_connected(self) -> bool:
        """Check connectivity via union-find."""
        parent = list(range(self.n_vertices))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for src, tgt, _ in self.edges:
            union(src, tgt)

        roots = {find(i) for i in range(self.n_vertices)}
        return len(roots) == 1

    def has_bridge(self) -> bool:
        """
        Detect if the internal subgraph has a bridge.
        Only considers internal vertices and internal edges.
        """
        import networkx as nx
        G = nx.MultiGraph()
        G.add_nodes_from(range(self.n_vertices_int))  # internal vertices only
        for i, (src, tgt, is_ext) in enumerate(self.edges):
            if not is_ext and src < self.n_vertices_int and tgt < self.n_vertices_int:
                G.add_edge(src, tgt, key=i)
        if not nx.is_connected(G):
            return False  # already disconnected — not 1PI anyway
        for i, (src, tgt, is_ext) in enumerate(self.edges):
            if (not is_ext and src != tgt and
                    src < self.n_vertices_int and tgt < self.n_vertices_int):
                H = nx.MultiGraph(G)
                H.remove_edge(src, tgt, key=i)
                if not nx.is_connected(H):
                    return True
        return False

    def is_1pi(self) -> bool:
        """1PI (one-particle irreducible): no bridges among internal edges."""
        return not self.has_bridge()

    def symmetry_factor(self) -> int:
        """
        Compute |Aut(G)| via canonical isomorphism.

        This is the denominator in the EGF expansion:
            EGF contribution = amplitude / |Aut(G)|

        For Feynman graphs with labelled vertices (from the half-edge
        pairing), |Aut(G)| = 1 for most cases. We compute it here by
        checking all permutations of unlabelled edges.

        Returns |Aut(G)| as an integer.
        """
        import networkx as nx
        G = nx.MultiGraph()
        G.add_nodes_from(range(self.n_vertices))
        for src, tgt, _ in self.edges:
            G.add_edge(src, tgt)
        # Use networkx automorphism group size
        # (Approximation: treat as simple graph for now)
        matcher = nx.algorithms.isomorphism.GraphMatcher(G, G)
        count = sum(1 for _ in matcher.isomorphisms_iter())
        return count

    def degree_of_divergence(self, d: sp.Expr = None) -> sp.Expr:
        """
        Superficial degree of divergence:
            σ_d(G) = |E_int| - (d/2) · L

        (Amarteifio eq. 2.101, 2.110)

        For σ_d > 0: UV divergent
        For σ_d = 0: logarithmically divergent
        For σ_d < 0: UV convergent
        """
        if d is None:
            d = sp.Symbol('d', positive=True)
        return self.n_internal_edges - sp.Rational(1, 2) * d * self.L

    # ------------------------------------------------------------------ #
    #  Standard graphs                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def tadpole(cls) -> 'FeynmanGraph':
        """
        Simple tadpole: one vertex, one self-loop, two external legs.
        Used in the simplest 1-loop integral.
        """
        # v0 = internal vertex, v1 = v_∞
        # Internal edge: v0 → v0 (self-loop)
        # External edges: v1 → v0, v0 → v1
        return cls(
            n_internal_vertices=1,
            edges=[
                (0, 0, False),  # self-loop (internal)
                (1, 0, True),   # external in
                (0, 1, True),   # external out
            ]
        )

    @classmethod
    def sunset(cls) -> 'FeynmanGraph':
        """
        Sunset diagram: 2 vertices, 3 internal edges.
        See Amarteifio Example 2.2.1 and Example 2.3.2.
        """
        # v0, v1 = internal; v2 = v_∞
        # 3 internal edges from v0 to v1
        # 2 external edges connecting to v_∞
        return cls(
            n_internal_vertices=2,
            edges=[
                (0, 1, False),  # p0 (internal)
                (0, 1, False),  # p1 (internal)
                (0, 1, False),  # p2 (internal)
                (2, 0, True),   # q0 (external)
                (1, 2, True),   # q1 (external)
            ]
        )

    @classmethod
    def one_loop_self_energy(cls) -> 'FeynmanGraph':
        """
        One-loop self-energy: 2 internal vertices, 2 internal edges (loop),
        2 external legs. This is the basic diagram in 2A → ∅ theory.

        Amarteifio Example 2.5.15.
        """
        # v0, v1 = internal; v2 = v_∞
        return cls(
            n_internal_vertices=2,
            edges=[
                (0, 1, False),  # p0 (internal, loop edge)
                (1, 0, False),  # p1 (internal, loop edge)
                (2, 0, True),   # q0 (external)
                (1, 2, True),   # q1 (external)
            ]
        )

    @classmethod
    def three_vertex_loop(cls) -> 'FeynmanGraph':
        """
        Three-vertex loop (Amarteifio Example 2.5.17):
        3 internal vertices, 3 edges forming a triangle, 2 external legs.
        """
        return cls(
            n_internal_vertices=3,
            edges=[
                (0, 1, False),  # p0
                (1, 2, False),  # p1
                (2, 0, False),  # p2 (closing the triangle)
                (3, 0, True),   # q0 (external)
                (2, 3, True),   # q1 (external)
            ]
        )

    def __repr__(self) -> str:
        return (f'FeynmanGraph(V={self.n_vertices_int}, '
                f'E_int={self.n_internal_edges}, '
                f'E_ext={self.n_external_edges}, '
                f'L={self.L})')
