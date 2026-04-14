"""
rdft.ac.lerw_exact
==================
Deterministic LERW mean length from the Kirchhoff ratio (Prop 1).

This module is the honest answer to the question:
  **What is the combinatorial object that captures dynamical path
  length?**

It is already Prop 1 of paper/cfac/enumerative_boundary.tex --- the
length generating function
    F_{ab}(z; G) = sum over simple paths gamma from a to b of
                   z^{|gamma|} * w(gamma) * tau(G / gamma) / tau(G).
This is a rational function of z on every finite graph G, with
  <|gamma|> = F'_{ab}(1),
and it is deterministic --- no Monte Carlo. The values below for
small Z^d_L boxes are computed by direct enumeration of simple
paths together with Kirchhoff determinants, and they agree with
Monte Carlo to MC precision.

What this module witnesses:
  1. The length GF is the right combinatorial object to
     differentiate for d_f. It is NOT the sandpile group, NOT the
     edge-current sum, NOT any fixed-point-of-a-finite-rewrite
     exponent: it is the Kirchhoff ratio summed against z^{|gamma|}.
  2. On small Z^d_L boxes it is exactly computable (rational in
     the edge weights) --- a deterministic witness against which
     the Monte-Carlo sampler can be validated.
  3. The cost of computing it at scale is exponential in graph
     size --- there are ~ (coordination)^{diameter} simple paths,
     and the determinant ratio inside the sum must be taken for
     each. No known analytic shortcut exists in d=3 because the
     limit of F_{ab} is the non-D-finite lattice Green's function.

Values computed here (Z^d_L cubic box, diagonal corners):
  d = 2, L = 3: exact mean = 17/4 = 4.250.
  d = 2, L = 4: exact mean = 6.692283... (rational, hand-verifiable)
  d = 2, L = 5: exact mean = 9.295257...
  d = 3, L = 2: exact mean = 53/16 = 3.3125.
  d = 3, L = 3: exact mean = 7.841463... (124 s enumeration)
"""

from __future__ import annotations
import numpy as np
from rdft.ac.lerw import lerw_mean_length


def cubic_box_graph(L: int, d: int) -> tuple[int, list[tuple[int, int, float]]]:
    """Return the standard d-dimensional cubic box graph on L^d
    vertices with unit edge weights. Vertices indexed as
    sum_k c_k * L^k with c_k in {0, ..., L-1}.
    """
    strides = [L ** i for i in range(d)]
    n = L ** d
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        c = [(i // s) % L for s in strides]
        for axis in range(d):
            if c[axis] + 1 < L:
                c2 = c[:]
                c2[axis] += 1
                j = sum(ci * si for ci, si in zip(c2, strides))
                edges.append((min(i, j), max(i, j), 1.0))
    return (n, edges)


def exact_corner_to_corner_mean(L: int, d: int) -> float:
    """Deterministic mean LERW length from (0,...,0) to (L-1,...,L-1)
    on the d-dim cubic box, computed via Prop 1.

    Warning: enumeration is exponential in L^d. Tractable up to
    about (L=5, d=2) or (L=3, d=3).
    """
    G = cubic_box_graph(L, d)
    return lerw_mean_length(G, 0, L ** d - 1)


# Pre-computed deterministic values (to avoid re-running the
# expensive enumeration in tests).
DETERMINISTIC_VALUES: dict[tuple[int, int], float] = {
    (3, 2): 4.25,
    (4, 2): 6.692283168775648,
    (5, 2): 9.295256835232322,
    (2, 3): 3.3125,
    (3, 3): 7.841463414634146,
}


def corner_distance(L: int, d: int) -> int:
    """Graph (taxicab) distance from origin to opposite corner:
    d * (L - 1).
    """
    return d * (L - 1)
