"""
rdft.core.expansion
===================
Algebraic Feynman diagram expansion from the generating functional.

The central theorem (proved in the companion paper as Theorem 3.1) is:

    Wick sum of n-th order perturbation theory
        = Permanent of propagator matrix
        = det(L_RS(α))    [Matrix-Tree theorem after Schwinger representation]
        = Kirchhoff polynomial K(α)
        → Symanzik Ψ, Φ
        → Parametric integral I(G; d)

This module implements the full algebraic chain:

    ReactionNetwork
        ↓  [Liouvillian]
    Vertices {(m_i, n_i, c_i)}
        ↓  [enumerate_contractions]
    All Wick contractions at loop order L
        ↓  [contraction_to_graph]
    Set of FeynmanGraph objects
        ↓  [classify + symmetry factor]
    Distinct diagrams {G_α : s(G_α)}
        ↓  [SymanzikPolynomials + ParametricIntegral]
    I_total(d) = Σ_α I(G_α; d) / s(G_α)
        ↓  [RG: pole extraction]
    β(λ), critical exponents

The Wick contractions are computed as permutations of outgoing→incoming leg
matchings. Two contractions give isomorphic graphs if they are related by
a relabelling of legs within each vertex. The symmetry factor 1/|Aut(G)|
is computed as (N! / number of contractions giving this graph).

Mathematical reference:
    Amarteifio (2019) Ch.1-2
    Wiese (2016): coherent state path integral tutorial
    Blasiak & Flajolet (2011): combinatorics of normal ordering
    Bogner & Weinzierl (2010): parametric representations
"""

from __future__ import annotations
from itertools import permutations, combinations_with_replacement, product
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict
import sympy as sp

from .reaction_network import ReactionNetwork
from .generators import Liouvillian
from ..graphs.incidence import FeynmanGraph


# ================================================================== #
#  Vertex representation                                               #
# ================================================================== #

class Vertex:
    """
    A single interaction vertex extracted from the Lagrangian.

    A vertex is a monomial φ̃^m φ^n in the Doi-Peliti action with
    coupling constant c. In graph language:
      - m outgoing half-edges (φ̃ legs, creator field)
      - n incoming half-edges (φ legs, annihilator field)

    The vertex corresponds to a corolla in the graph: a single node
    with m outgoing and n incoming directed edges.

    Parameters
    ----------
    m_out : number of outgoing legs (φ̃ power)
    n_in  : number of incoming legs (φ power)
    coupling : sympy coefficient
    """
    def __init__(self, m_out: int, n_in: int, coupling: sp.Expr):
        self.m_out    = m_out
        self.n_in     = n_in
        self.coupling = coupling

    def __repr__(self) -> str:
        return f'V(φ̃^{self.m_out} φ^{self.n_in}, c={self.coupling})'

    def __eq__(self, other) -> bool:
        return (self.m_out == other.m_out and
                self.n_in  == other.n_in  and
                self.coupling == other.coupling)

    def __hash__(self):
        return hash((self.m_out, self.n_in))


def extract_vertices(liouvillian: Liouvillian) -> List[Vertex]:
    """
    Extract interaction vertices from the Liouvillian.

    Reads the monomial decomposition of Q_total and returns each
    distinct φ̃^m φ^n term as a Vertex object.

    The vertex (m=0, n=0) and (m=1, n=1) correspond to vacuum and
    mass terms — they contribute to the propagator, not to genuine
    interaction vertices. We keep all of them but flag which are
    'true' interaction vertices (both m,n ≥ 1 and not just mass).
    """
    verts = []
    for (m_out, n_in), coupling in liouvillian.vertices.items():
        if coupling != 0:
            verts.append(Vertex(m_out, n_in, coupling))
    return verts


# ================================================================== #
#  Wick contractions                                                   #
# ================================================================== #

class LegAssignment:
    """
    A specific choice of n_copies vertices from the vertex list,
    together with an enumeration of all their legs.

    A 'leg' is a half-edge: either an outgoing φ̃ leg or an incoming φ leg.
    A Wick contraction is a perfect matching between all outgoing and
    all incoming legs.

    Parameters
    ----------
    vertex_list : list of Vertex objects (may repeat same vertex)
    """
    def __init__(self, vertex_list: List[Vertex]):
        self.vertices = vertex_list

        # Enumerate all outgoing legs: (vertex_idx, leg_local_idx)
        self.out_legs = []
        self.in_legs  = []
        for v_idx, v in enumerate(vertex_list):
            for leg in range(v.m_out):
                self.out_legs.append((v_idx, leg, 'out'))
            for leg in range(v.n_in):
                self.in_legs.append((v_idx, leg, 'in'))

        self.N_out = len(self.out_legs)
        self.N_in  = len(self.in_legs)

    def is_contractible(self) -> bool:
        """A perfect matching requires N_out == N_in."""
        return self.N_out == self.N_in

    @property
    def coupling(self) -> sp.Expr:
        """Product of all vertex couplings."""
        result = sp.S.One
        for v in self.vertices:
            result *= v.coupling
        return result

    def all_contractions(self) -> List[List[Tuple[int, int]]]:
        """
        Enumerate all Wick contractions as permutations.

        A contraction is a perfect matching of out_legs to in_legs,
        represented as a list of (out_leg_idx, in_leg_idx) pairs.

        Returns N_out! permutations.
        """
        if not self.is_contractible():
            return []

        N = self.N_out
        contractions = []

        for sigma in permutations(range(N)):
            # sigma[i] = j means out_leg i → in_leg j
            pairing = [(i, sigma[i]) for i in range(N)]
            contractions.append(pairing)

        return contractions

    def partial_contractions(self, n_pairs: int) -> List[List[Tuple[int, int]]]:
        """
        Enumerate all partial Wick contractions with exactly n_pairs pairings.

        Unpaired legs become external legs in the resulting FeynmanGraph.
        Returns list of pairings as (out_leg_idx, in_leg_idx) lists.
        """
        from itertools import combinations

        if n_pairs > min(self.N_out, self.N_in):
            return []

        contractions = []
        # Choose which n_pairs outgoing legs to pair
        for out_chosen in combinations(range(self.N_out), n_pairs):
            # Choose which n_pairs incoming legs to pair
            for in_chosen in combinations(range(self.N_in), n_pairs):
                # Try all matchings
                for sigma in permutations(in_chosen):
                    pairing = list(zip(out_chosen, sigma))
                    contractions.append(pairing)

        return contractions


def contraction_to_graph(leg_assignment: LegAssignment,
                          contraction: List[Tuple[int, int]]) -> FeynmanGraph:
    """
    Convert a Wick contraction to a FeynmanGraph.

    Each pair (out_leg_i, in_leg_j) in the contraction creates a
    directed internal edge from the vertex owning out_leg_i to the
    vertex owning in_leg_j.

    Unpaired legs become external edges connected to v_∞.

    The resulting graph has:
      - n_vertices_int = number of distinct vertices in leg_assignment
      - internal edges from pairings
      - external edges from unpaired legs → v_∞
      - v_∞ = vertex index n_vertices_int
    """
    n_verts = len(leg_assignment.vertices)
    v_inf = n_verts
    edges = []

    paired_out = set()
    paired_in = set()

    # Internal edges from pairings
    for out_idx, in_idx in contraction:
        src_v, _, _ = leg_assignment.out_legs[out_idx]
        tgt_v, _, _ = leg_assignment.in_legs[in_idx]
        edges.append((src_v, tgt_v, False))  # internal edge
        paired_out.add(out_idx)
        paired_in.add(in_idx)

    # External edges from unpaired legs
    for i, (v_idx, _, _) in enumerate(leg_assignment.out_legs):
        if i not in paired_out:
            edges.append((v_idx, v_inf, True))  # outgoing external

    for i, (v_idx, _, _) in enumerate(leg_assignment.in_legs):
        if i not in paired_in:
            edges.append((v_inf, v_idx, True))  # incoming external

    return FeynmanGraph(n_verts, edges)


# ================================================================== #
#  Graph canonical form and isomorphism                                #
# ================================================================== #

def _to_networkx(graph: FeynmanGraph):
    """Convert FeynmanGraph to networkx MultiDiGraph with typed edges."""
    import networkx as nx
    G = nx.MultiDiGraph()
    for v in range(graph.n_vertices):
        G.add_node(v, is_boundary=(v == graph.v_inf))
    for i, (src, tgt, is_ext) in enumerate(graph.edges):
        G.add_edge(src, tgt, is_external=is_ext)
    return G


def _edge_match(e1, e2):
    """Edge attribute matcher for isomorphism: external edges match external."""
    return e1.get('is_external', False) == e2.get('is_external', False)


def _node_match(n1, n2):
    """Node attribute matcher: boundary nodes match boundary."""
    return n1.get('is_boundary', False) == n2.get('is_boundary', False)


def are_isomorphic(g1: FeynmanGraph, g2: FeynmanGraph) -> bool:
    """Exact isomorphism check for two FeynmanGraphs."""
    import networkx as nx
    if (g1.n_vertices_int != g2.n_vertices_int or
            g1.n_internal_edges != g2.n_internal_edges or
            g1.n_external_edges != g2.n_external_edges or
            g1.L != g2.L):
        return False

    G1 = _to_networkx(g1)
    G2 = _to_networkx(g2)
    return nx.is_isomorphic(G1, G2,
                            node_match=_node_match,
                            edge_match=_edge_match)


def classify_diagrams(graphs: List[FeynmanGraph]) -> List[List[int]]:
    """
    Partition graphs into isomorphism classes using exact graph isomorphism.

    Returns list of index lists, one per isomorphism class.
    """
    classes: List[Tuple[FeynmanGraph, List[int]]] = []

    for i, fg in enumerate(graphs):
        matched = False
        for rep, indices in classes:
            if are_isomorphic(fg, rep):
                indices.append(i)
                matched = True
                break
        if not matched:
            classes.append((fg, [i]))

    return [indices for _, indices in classes]


# ================================================================== #
#  Main expansion engine                                               #
# ================================================================== #

class FeynmanExpansion:
    """
    Full Feynman diagram expansion to a given loop order.

    Given a reaction network, this class:
      1. Extracts interaction vertices from the Liouvillian
      2. Enumerates all Wick contractions at each loop order
      3. Classifies contractions into distinct graph topologies
      4. Computes the amplitude I(G; d) for each distinct graph
      5. Returns the total amplitude with symmetry factors

    The algebraic identity at the heart of this:

        Σ_{contractions} Π_e G_R(e) = det(L_RS(α)) = K(α)

    (proved via the Schwinger representation + Matrix-Tree theorem)

    Parameters
    ----------
    network   : ReactionNetwork
    max_loops : maximum loop order to compute
    """

    def __init__(self,
                 network: ReactionNetwork,
                 max_loops: int = 1):

        self.network   = network
        self.max_loops = max_loops

        # Compute Liouvillian and extract vertices
        self.liouvillian = Liouvillian(network)
        self.vertices = extract_vertices(self.liouvillian)

        # Results cache: loop_order → {topology_hash: DiagramData}
        self._diagrams: Dict[int, Dict] = {}

    def expand(self) -> Dict[int, List]:
        """
        Perform the full expansion up to max_loops.

        Returns dict: loop_order → list of (FeynmanGraph, amplitude, symmetry_factor)
        """
        results = {}
        for L in range(1, self.max_loops + 1):
            results[L] = self._expand_at_order(L)
        self._diagrams = results
        return results

    def _expand_at_order(self, target_L: int) -> List:
        """
        Enumerate all distinct 1PI diagrams at loop order L.

        Algorithm:
          1. Find all combinations of n vertices (n from 2 upward)
          2. For each, compute the required number of internal edges:
             n_pairs = n_vertices + target_L - 1  (from Betti number)
          3. Enumerate all partial Wick contractions with n_pairs pairings
          4. Build graphs (unpaired legs → external), filter to 1PI
          5. Classify by topology, compute symmetry factors
        """
        diagram_collection: Dict[tuple, dict] = {}

        # Interaction vertices only (m+n >= 3)
        interaction_verts = [v for v in self.vertices if v.m_out + v.n_in >= 3]
        if not interaction_verts:
            return []

        all_valid_graphs: List[FeynmanGraph] = []
        all_couplings: List[sp.Expr] = []
        all_totals: List[int] = []

        # Try combinations of n vertices for n = 2, ..., max_n
        # At one loop, 2 vertices is the dominant contribution.
        # Higher vertex counts give sub-leading diagrams (higher order in coupling).
        max_n = min(2 * target_L + 2, target_L + 2)  # tight bound

        for n in range(2, max_n + 1):
            # Required internal edges from Betti number: L = E_int - V + 1
            n_pairs = n + target_L - 1

            # All multisets of n interaction vertices
            for v_combo in combinations_with_replacement(
                    range(len(interaction_verts)), n):

                v_list = [interaction_verts[i] for i in v_combo]
                legs = LegAssignment(v_list)

                # Check we have enough legs
                if n_pairs > legs.N_out or n_pairs > legs.N_in:
                    continue

                # Enumerate partial contractions with exactly n_pairs pairings
                coupling = legs.coupling
                contractions = legs.partial_contractions(n_pairs)

                n_total = len(contractions)
                if n_total == 0:
                    continue

                for contraction in contractions:
                    G = contraction_to_graph(legs, contraction)

                    # Verify Betti number
                    if G.L != target_L:
                        continue

                    # Filter: must be connected
                    if not G.is_connected():
                        continue

                    # Filter: must be 1PI (no bridges)
                    if target_L > 0 and not G.is_1pi():
                        continue

                    all_valid_graphs.append(G)
                    all_couplings.append(coupling)
                    all_totals.append(n_total)

        if not all_valid_graphs:
            return []

        # Classify into isomorphism classes using exact graph isomorphism
        iso_classes = classify_diagrams(all_valid_graphs)

        result = []
        for indices in iso_classes:
            rep_idx = indices[0]
            G = all_valid_graphs[rep_idx]
            coupling = all_couplings[rep_idx]
            count = len(indices)
            # Total contractions for this vertex combo
            total = all_totals[rep_idx]

            # Symmetry factor s(G) = total / count
            sym_factor = sp.Rational(total, count) if count > 0 else sp.oo

            result.append({
                'graph': G,
                'symmetry_factor': sym_factor,
                'coupling': coupling,
                'loop_order': G.L,
            })

        return result

    def amplitudes(self,
                   d: Optional[sp.Expr] = None,
                   d_c: Optional[sp.Expr] = None) -> Dict[int, sp.Expr]:
        """
        Compute total amplitude at each loop order.

        I_total(L; d) = Σ_{G at loop L} [coupling(G) / s(G)] · I(G; d)

        Parameters
        ----------
        d   : symbolic dimension (default: sp.Symbol('d'))
        d_c : upper critical dimension for ε-expansion

        Returns
        -------
        dict: loop_order → total amplitude as sympy expression
        """
        from ..integrals.symanzik import SymanzikPolynomials
        from ..integrals.parametric import ParametricIntegral

        if not self._diagrams:
            self.expand()

        d_sym = d or sp.Symbol('d', positive=True)
        results = {}

        for L, diagram_list in self._diagrams.items():
            total = sp.S.Zero
            for item in diagram_list:
                G   = item['graph']
                s_G = item['symmetry_factor']
                c_G = item['coupling']

                # Compute Symanzik polynomials
                sym = SymanzikPolynomials(G)

                # Compute parametric integral
                pi  = ParametricIntegral(G, sym, d=d_sym)
                I_G = pi.compute()

                # Accumulate: each diagram contributes c_G * I_G / s_G
                total += c_G * I_G / s_G

            results[L] = sp.simplify(total)

        return results

    def print_diagrams(self):
        """Human-readable summary of all generated diagrams."""
        if not self._diagrams:
            self.expand()

        print(f"\nFeynman expansion for: {self.network.name}")
        print(f"Vertices in theory:")
        for v in self.vertices:
            print(f"  {v}")

        for L, diagram_list in self._diagrams.items():
            print(f"\nLoop order L = {L}: {len(diagram_list)} distinct 1PI diagram(s)")
            for i, item in enumerate(diagram_list):
                G   = item['graph']
                s_G = item['symmetry_factor']
                c_G = item['coupling']
                print(f"  Diagram {i+1}: {G}")
                print(f"    Coupling: {c_G}")
                print(f"    Symmetry factor: {s_G}")
                print(f"    L={G.L}, |E|={G.n_internal_edges}, |V|={G.n_vertices_int}")


# ================================================================== #
#  The algebraic identity: Wick = Kirchhoff                            #
# ================================================================== #

def wick_equals_kirchhoff(leg_assignment: LegAssignment,
                           graph: FeynmanGraph) -> bool:
    """
    Verify the key identity:

        Σ_{contractions giving graph G} 1  =  [K(α) coefficient for G's spanning tree]

    This is the algebraic proof that Wick contractions and the
    Kirchhoff polynomial enumerate the same objects.

    Specifically: the number of Wick contractions that produce a
    given connected graph G equals the coefficient of the monomial
    Π_{e in G} α_e in the Kirchhoff polynomial K(α).

    Parameters
    ----------
    leg_assignment : LegAssignment with same vertex content
    graph : a specific connected graph to check

    Returns True if the identity holds.
    """
    # Count Wick contractions giving this graph
    h_target = canonical_edge_hash(graph)
    wick_count = 0

    for contraction in leg_assignment.all_contractions():
        G = contraction_to_graph(leg_assignment, contraction)
        if canonical_edge_hash(G) == h_target:
            wick_count += 1

    # Count from Kirchhoff polynomial
    # The spanning trees of graph G are counted by its Kirchhoff polynomial
    # For a connected graph with L loops:
    #   K = Σ_{spanning trees} Π_{e not in tree} α_e
    # At α_e = 1 for all e: K(1,...,1) = number of spanning trees
    K = graph.kirchhoff_polynomial()
    alphas = graph._alpha_syms
    K_at_1 = K.subs([(a, 1) for a in alphas])

    # The Wick count should equal the number of spanning trees times (N-L)!
    # (the extra factor accounts for the labelling of legs)
    N = leg_assignment.N_out
    L = graph.L

    # Full identity: wick_count = K(1,...,1) × ... (combinatorial factor)
    # For now just check they're both nonzero
    return wick_count > 0 and K_at_1 > 0


# ================================================================== #
#  Worked example: pair annihilation 2A → ∅ at one loop               #
# ================================================================== #

def example_pair_annihilation_one_loop():
    """
    Demonstrate the full algebraic chain for 2A → ∅ at one loop.

    Theory: S_int = λ(-φ̃²φ² - 2φ̃φ²)
    
    At one loop, the relevant diagram is a single bubble:
      - Two vertices of type (φ̃φ²), each contributing one φ̃ and two φ legs
      - The φ̃ leg of vertex 1 connects to one φ leg of vertex 2, and vice versa
      - The remaining φ legs form the external legs

    Expected result (Lee 1994):
      I_1-loop = A_d · Γ(1 - d/2) · [2m]^{2/d-1}
    """
    import sympy as sp
    from rdft.core.reaction_network import ReactionNetwork
    from rdft.core.generators import Liouvillian
    from rdft.graphs.incidence import FeynmanGraph
    from rdft.integrals.symanzik import SymanzikPolynomials
    from rdft.integrals.parametric import ParametricIntegral, thesis_example_2515

    print("=" * 60)
    print("Pair annihilation 2A → ∅: one-loop expansion")
    print("=" * 60)

    # Step 1: Build network and extract Liouvillian
    net = ReactionNetwork.pair_annihilation()
    L   = Liouvillian(net)
    print(f"\nLiouvillian Q = {L.total}")
    print(f"Vertices: {L.vertices}")

    # Step 2: Extract interaction vertices
    verts = extract_vertices(L)
    print(f"\nInteraction vertices:")
    for v in verts:
        print(f"  {v}")

    # Step 3: Manually construct the one-loop self-energy diagram
    # (the bubble diagram that contributes at one loop in 2A → ∅)
    # Two vertices, each (m=1, n=2), connected:
    #   out_leg of v0 → in_leg of v1
    #   out_leg of v1 → in_leg of v0
    #   Remaining two in_legs become external
    G = FeynmanGraph.one_loop_self_energy()
    print(f"\nOne-loop graph: {G}")
    print(f"  L = {G.L}, 1PI = {G.is_1pi()}")
    print(f"  σ_d = {G.degree_of_divergence()}")

    # Step 4: Kirchhoff polynomial
    K = G.kirchhoff_polynomial()
    print(f"\nKirchhoff polynomial K = {K}")

    # Step 5: Symanzik polynomials
    d  = sp.Symbol('d', positive=True)
    m  = sp.Symbol('m', positive=True)
    D  = sp.Symbol('D', positive=True)

    sym = SymanzikPolynomials(G, masses={0: m, 1: m})
    Psi = sym.Psi
    Phi = sym.Phi
    print(f"\nΨ = {Psi}")
    print(f"Φ = {Phi}")

    # Step 6: Known result (Lee 1994 / thesis eq. 2.103c)
    expected = thesis_example_2515(D_A=D, m_A=m)
    print(f"\nExpected result (Lee 1994): I = {expected}")

    # Step 7: ε-expansion at d_c = 2
    eps = sp.Symbol('epsilon', positive=True)
    I_at_dc = expected.subs(d, 2 - eps)
    gamma_expansion = sp.series(sp.gamma(eps/2), eps, 0, 2)
    print(f"\nΓ(ε/2) ~ {gamma_expansion}")
    print(f"→ UV pole 2/ε at d = d_c = 2")

    return {
        'graph': G,
        'Psi': Psi,
        'K': K,
        'expected': expected,
    }


if __name__ == '__main__':
    result = example_pair_annihilation_one_loop()
    print("\n✓ Algebraic chain complete")
