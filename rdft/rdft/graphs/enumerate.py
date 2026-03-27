"""
rdft.graphs.enumerate
=====================
Systematic enumeration of 1PI Feynman diagrams via the shuffle product.

Given a set of primitive corollas (from the Liouvillian vertices), generates
all 1PI Feynman graphs up to a specified loop order by pairing half-edges.

The shuffle product (Amarteifio Definition 12) pairs outgoing (phi-tilde)
half-edges with incoming (phi) half-edges to form internal propagators.
Unpaired half-edges become external legs connected to v_infinity.

Reference: Amarteifio (2019) section 2.5, Definition 12
"""

from typing import List, Tuple, Optional, Set
from itertools import combinations, combinations_with_replacement, permutations
from collections import Counter
import sympy as sp

from .corolla import Corolla, HalfEdge, corollas_from_liouvillian
from .incidence import FeynmanGraph


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def enumerate_diagrams(corollas: List[Corolla],
                       max_loops: int = 1,
                       min_loops: int = 1,
                       target_residue: Optional[Tuple[int, int]] = None,
                       ) -> List[dict]:
    """
    Enumerate all distinct 1PI Feynman diagrams from given corollas.

    Parameters
    ----------
    corollas : list of Corolla
        Vertex types to use (typically from ``corollas_from_liouvillian``).
    max_loops : int
        Maximum loop order *L* to enumerate.
    min_loops : int
        Minimum loop order (default 1, skipping tree level).
    target_residue : (n_out_ext, n_in_ext) or None
        If given, keep only diagrams whose external-leg signature matches.

    Returns
    -------
    list of dict
        Each dict contains:

        * ``'graph'``: a :class:`FeynmanGraph`
        * ``'vertices'``: list of :class:`Corolla` used
        * ``'pairings'``: list of ``(src_vertex, tgt_vertex)`` pairs
        * ``'symmetry_factor'``: ``|Aut(G)|``
        * ``'coupling'``: product of vertex couplings (sympy expression)
        * ``'loop_order'``: int
        * ``'ext_out'``: number of unpaired outgoing half-edges
        * ``'ext_in'``: number of unpaired incoming half-edges
    """
    results: List[dict] = []
    seen: Set[str] = set()

    for L in range(min_loops, max_loops + 1):
        for diag in _enumerate_at_loop_order(corollas, L, target_residue):
            h = _canonical_hash(diag['graph'])
            if h not in seen:
                seen.add(h)
                diag['loop_order'] = L
                results.append(diag)

    return results


def diagrams_for_process(liouvillian,
                         max_loops: int = 1,
                         target_residue: Optional[Tuple[int, int]] = None,
                         ) -> List[dict]:
    """
    Generate all 1PI Feynman diagrams for a given Liouvillian.

    Convenience wrapper: extracts corollas, then enumerates diagrams.

    Parameters
    ----------
    liouvillian : Liouvillian
        From ``rdft.core.generators``.
    max_loops : int
        Maximum loop order.
    target_residue : (n_out, n_in) or None
        Filter by external-leg signature.

    Returns
    -------
    list of dict
    """
    cors = corollas_from_liouvillian(liouvillian)
    return enumerate_diagrams(cors, max_loops=max_loops,
                              target_residue=target_residue)


def summary(diagrams: List[dict]) -> str:
    """Return a human-readable summary of enumerated diagrams."""
    lines = [f'Found {len(diagrams)} distinct 1PI diagram(s):']
    for i, d in enumerate(diagrams):
        g = d['graph']
        lines.append(
            f'  [{i + 1}] V={g.n_vertices_int}, '
            f'E_int={g.n_internal_edges}, '
            f'E_ext={g.n_external_edges}, '
            f'L={g.L}, '
            f'ext=({d.get("ext_out", "?")},{d.get("ext_in", "?")}), '
            f'coupling={d["coupling"]}, '
            f'sym={d.get("symmetry_factor", "?")}'
        )
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Core enumeration engine                                             #
# ------------------------------------------------------------------ #

def _enumerate_at_loop_order(corollas: List[Corolla],
                              L: int,
                              target_residue: Optional[Tuple[int, int]],
                              ) -> List[dict]:
    """Enumerate diagrams at exactly loop order *L*.

    The Betti-number constraint is:

        E_int = V + L - 1

    where *V* is the number of internal vertices and *E_int* the number
    of internal edges.  Each internal edge consumes one outgoing and one
    incoming half-edge.
    """
    results: List[dict] = []

    # Upper bound on number of vertices: each corolla contributes at
    # least 3 half-edges so at most 2*E_int half-edges are consumed.
    # E_int = V + L - 1.  For safety we cap at a generous limit.
    max_verts = 2 * L + 4

    for n_verts in range(2, max_verts + 1):
        E_int = n_verts + L - 1  # required internal edges

        for combo in combinations_with_replacement(range(len(corollas)), n_verts):
            vertex_list = [corollas[i] for i in combo]

            total_out = sum(c.n_out for c in vertex_list)
            total_in = sum(c.n_in for c in vertex_list)

            # Need E_int pairings; each uses one out and one in half-edge
            if E_int > total_out or E_int > total_in:
                continue

            ext_out = total_out - E_int
            ext_in = total_in - E_int

            if target_residue is not None:
                if (ext_out, ext_in) != target_residue:
                    continue

            # Generate all structurally distinct pairings
            for edge_list in _generate_pairings(vertex_list, E_int):
                graph = _build_graph(vertex_list, edge_list)
                if graph is None:
                    continue

                # Sanity: loop order must match
                if graph.L != L:
                    continue

                if not graph.is_connected():
                    continue

                if L > 0 and not graph.is_1pi():
                    continue

                coupling = sp.Mul(*[c.coupling for c in vertex_list])

                results.append({
                    'graph': graph,
                    'vertices': list(vertex_list),
                    'pairings': edge_list,
                    'coupling': coupling,
                    'symmetry_factor': graph.symmetry_factor(),
                    'ext_out': ext_out,
                    'ext_in': ext_in,
                })

    return results


# ------------------------------------------------------------------ #
#  Pairing generation                                                  #
# ------------------------------------------------------------------ #

def _generate_pairings(vertex_list: List[Corolla],
                        n_pairs: int,
                        ) -> List[List[Tuple[int, int]]]:
    """
    Yield all distinct ways to pair *n_pairs* outgoing half-edges with
    *n_pairs* incoming half-edges across the given vertex list.

    Each pairing is a list of ``(src_vertex_idx, tgt_vertex_idx)``
    tuples (one per internal edge / propagator).

    Strategy
    --------
    We represent the outgoing and incoming half-edge pools as flat
    lists of vertex indices (with repetitions), then choose *n_pairs*
    from each pool and match them via permutations.  Because many
    half-edges on the same vertex are indistinguishable, we deduplicate
    by working with *sorted edge multisets* (frozensets of pairs with
    multiplicity).

    For low loop orders the combinatorics are manageable; at one loop
    with two cubic vertices the pools are at most length 4 total.
    """
    # Build pools: each entry is the vertex index that owns that half-edge
    out_pool: List[int] = []
    in_pool: List[int] = []
    for v_idx, cor in enumerate(vertex_list):
        out_pool.extend([v_idx] * cor.n_out)
        in_pool.extend([v_idx] * cor.n_in)

    if n_pairs > len(out_pool) or n_pairs > len(in_pool):
        return []

    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    result: List[List[Tuple[int, int]]] = []

    # Choose which outgoing half-edges participate (by index in pool)
    for out_chosen in combinations(range(len(out_pool)), n_pairs):
        out_verts = tuple(out_pool[i] for i in out_chosen)

        for in_chosen in combinations(range(len(in_pool)), n_pairs):
            in_verts = [in_pool[i] for i in in_chosen]

            # Match the chosen out half-edges to in half-edges.
            # Only try distinct permutations of in_verts to avoid
            # redundant work when several in-half-edges share a vertex.
            for perm in _unique_permutations(in_verts):
                edges = tuple(sorted(zip(out_verts, perm)))
                if edges not in seen:
                    seen.add(edges)
                    result.append(list(zip(out_verts, perm)))

    return result


def _unique_permutations(seq):
    """Yield each distinct permutation of *seq* exactly once.

    Uses a simple algorithm that generates permutations of a sorted
    list and skips duplicates.  For short sequences (length <= 6 in
    practice) this is efficient enough.
    """
    seq = sorted(seq)
    seen: Set[Tuple] = set()
    for p in permutations(seq):
        if p not in seen:
            seen.add(p)
            yield p


# ------------------------------------------------------------------ #
#  Graph construction                                                  #
# ------------------------------------------------------------------ #

def _build_graph(vertex_list: List[Corolla],
                  pairings: List[Tuple[int, int]],
                  ) -> Optional[FeynmanGraph]:
    """
    Assemble a :class:`FeynmanGraph` from vertices and pairings.

    Parameters
    ----------
    vertex_list : list of Corolla
        The vertex instances (order determines vertex indices 0 .. V-1).
    pairings : list of (src_vertex, tgt_vertex)
        Internal edges — each pair is a propagator from an outgoing
        half-edge on *src* to an incoming half-edge on *tgt*.

    Returns
    -------
    FeynmanGraph or None
        ``None`` if construction fails for any reason.

    Notes
    -----
    Unpaired half-edges are connected to ``v_inf`` as external edges.
    Outgoing unpaired half-edges produce edges ``(v, v_inf)``;
    incoming unpaired half-edges produce edges ``(v_inf, v)``.
    """
    n_verts = len(vertex_list)
    v_inf = n_verts

    edges: List[Tuple[int, int, bool]] = []

    # Internal edges from pairings
    for src, tgt in pairings:
        edges.append((src, tgt, False))

    # Count how many half-edges of each vertex are already paired
    out_used: Counter = Counter()
    in_used: Counter = Counter()
    for src, tgt in pairings:
        out_used[src] += 1
        in_used[tgt] += 1

    # External edges from unpaired half-edges
    for v_idx, cor in enumerate(vertex_list):
        for _ in range(cor.n_out - out_used.get(v_idx, 0)):
            edges.append((v_idx, v_inf, True))   # outgoing external
        for _ in range(cor.n_in - in_used.get(v_idx, 0)):
            edges.append((v_inf, v_idx, True))    # incoming external

    try:
        return FeynmanGraph(n_verts, edges)
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Isomorphism / deduplication                                         #
# ------------------------------------------------------------------ #

def _canonical_hash(graph: FeynmanGraph) -> str:
    """
    Compute a canonical hash for graph isomorphism detection.

    Uses networkx to produce a canonical form based on the
    Weisfeiler-Leman algorithm (via ``weisfeiler_lehman_graph_hash``).
    Falls back to a degree-sequence fingerprint if the WL hash is
    unavailable.

    This is an *approximation*: two graphs with the same hash are
    very likely isomorphic, but a collision is theoretically possible
    for large graphs.  For the small diagrams at one- and two-loop
    order this is more than adequate.
    """
    import networkx as nx

    G = nx.MultiGraph()
    G.add_nodes_from(range(graph.n_vertices))

    for i, (src, tgt, is_ext) in enumerate(graph.edges):
        G.add_edge(src, tgt, ext=is_ext)

    # Annotate nodes so v_inf is distinguished
    for n in G.nodes():
        G.nodes[n]['kind'] = 'inf' if n == graph.v_inf else 'int'

    try:
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr='kind',
                                             edge_attr='ext')
    except (AttributeError, TypeError):
        # Fallback: sorted degree sequence + edge counts
        degrees = tuple(sorted(dict(G.degree()).values()))
        n_int = graph.n_internal_edges
        n_ext = graph.n_external_edges
        h = f'{graph.n_vertices_int}_{n_int}_{n_ext}_{degrees}'

    return h
