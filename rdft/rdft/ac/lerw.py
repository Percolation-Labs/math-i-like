"""
rdft.ac.lerw
============
LERW as Kirchhoff enumeration.

Implements Proposition 1 of paper/cfac/enumerative_boundary.tex:
for any finite weighted graph G with distinguished vertices a, b,
the loop-erased random walk measure on simple paths from a to b is

    P(LERW = gamma) = w(gamma) * tau(G / gamma) / tau(G),

where tau(.) is the weighted Kirchhoff polynomial (matrix-tree
determinant) and G / gamma is the graph with path gamma contracted.

This identity rewrites LERW as a ratio of first Symanzik polynomials
- the same objects the CFAC pipeline computes for Feynman diagrams.
It places LERW inside the enumeration tier of CFAC independently of
whether the thermodynamic-limit generating function is D-finite.

Graph convention: G is represented as (n, edges), where n is the
number of vertices (labelled 0..n-1) and edges is a list of
(u, v, w) tuples with u < v. The graph is undirected and loopless.
"""

from __future__ import annotations
from typing import Iterable, Iterator, Sequence
import numpy as np


Edge = tuple[int, int, float]
Graph = tuple[int, list[Edge]]


# ------------------------------------------------------------------ #
#  Laplacian / Kirchhoff polynomial
# ------------------------------------------------------------------ #

def weighted_laplacian(G: Graph) -> np.ndarray:
    n, edges = G
    L = np.zeros((n, n))
    for u, v, w in edges:
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def kirchhoff(G: Graph, root: int = 0) -> float:
    """tau(G) = det of L with row/col `root` removed (matrix-tree theorem)."""
    L = weighted_laplacian(G)
    keep = [i for i in range(G[0]) if i != root]
    return float(np.linalg.det(L[np.ix_(keep, keep)]))


# ------------------------------------------------------------------ #
#  Path contraction: G / gamma
# ------------------------------------------------------------------ #

def contract_path(G: Graph, path: Sequence[int]) -> Graph:
    """Contract the simple path `path` to a single vertex.

    Multi-edges are kept (their weights add into a_uu cancelling out,
    so the Laplacian stays correct). Self-loops on the contracted
    supervertex are dropped (they don't contribute to tau).
    """
    n, edges = G
    # Map every vertex on the path to the supervertex (= path[0]).
    super_v = path[0]
    relabel = {v: super_v for v in path}
    # New vertex labels: compress the non-path vertices.
    old_to_new = {}
    next_id = 0
    for v in range(n):
        if v in relabel:
            if super_v not in old_to_new:
                old_to_new[super_v] = next_id
                next_id += 1
        else:
            old_to_new[v] = next_id
            next_id += 1
    new_edges: list[Edge] = []
    for u, v, w in edges:
        uu = old_to_new[relabel.get(u, u)]
        vv = old_to_new[relabel.get(v, v)]
        if uu == vv:
            continue  # self-loop, drop
        a, b = (uu, vv) if uu < vv else (vv, uu)
        new_edges.append((a, b, w))
    return (next_id, new_edges)


def path_weight(G: Graph, path: Sequence[int]) -> float:
    """Product of edge weights along `path`."""
    _, edges = G
    ew = {(min(u, v), max(u, v)): w for u, v, w in edges}
    prod = 1.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        key = (min(u, v), max(u, v))
        if key not in ew:
            raise ValueError(f"edge {u}-{v} not in graph")
        prod *= ew[key]
    return prod


# ------------------------------------------------------------------ #
#  Simple-path enumeration
# ------------------------------------------------------------------ #

def simple_paths(G: Graph, a: int, b: int,
                 max_length: int | None = None) -> Iterator[list[int]]:
    n, edges = G
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)

    def dfs(current: int, visited: set[int], path: list[int]):
        if current == b:
            yield list(path)
            return
        if max_length is not None and len(path) - 1 >= max_length:
            return
        for nxt in adj[current]:
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            yield from dfs(nxt, visited, path)
            path.pop()
            visited.remove(nxt)

    yield from dfs(a, {a}, [a])


# ------------------------------------------------------------------ #
#  LERW measure (Proposition 1)
# ------------------------------------------------------------------ #

def lerw_path_prob(G: Graph, path: Sequence[int]) -> float:
    """P(LERW = path) = w(path) * tau(G / path) / tau(G)."""
    w = path_weight(G, path)
    Gc = contract_path(G, path)
    # Single-vertex contracted graph: tau = 1 by convention
    tau_contracted = 1.0 if Gc[0] == 1 else kirchhoff(Gc)
    return w * tau_contracted / kirchhoff(G)


def lerw_length_gf_numer(G: Graph, a: int, b: int,
                         max_length: int | None = None,
                         ) -> list[float]:
    """Numerator of F_{ab}(z) = sum over simple paths of z^|gamma|
    * w(gamma) * tau(G/gamma), as a list of coefficients starting
    with z^0. Divide by tau(G) to get the normalised length GF.
    """
    coeffs: dict[int, float] = {}
    for path in simple_paths(G, a, b, max_length=max_length):
        w = path_weight(G, path)
        Gc = contract_path(G, path)
        tau_c = 1.0 if Gc[0] == 1 else kirchhoff(Gc)
        k = len(path) - 1  # number of edges
        coeffs[k] = coeffs.get(k, 0.0) + w * tau_c
    if not coeffs:
        return []
    d = max(coeffs)
    return [coeffs.get(i, 0.0) for i in range(d + 1)]


def lerw_mean_length(G: Graph, a: int, b: int) -> float:
    """Expected length of the LERW path from a to b on finite G.

    E[|gamma|] = sum_gamma |gamma| * P(LERW = gamma). Uses the
    Kirchhoff-ratio representation directly.
    """
    total = 0.0
    Z = 0.0
    tau_G = kirchhoff(G)
    for path in simple_paths(G, a, b):
        w = path_weight(G, path)
        Gc = contract_path(G, path)
        tau_c = 1.0 if Gc[0] == 1 else kirchhoff(Gc)
        p = w * tau_c / tau_G
        Z += p
        total += (len(path) - 1) * p
    # Z should be 1 up to numerical error; divide defensively
    return total / Z if Z > 0 else 0.0
