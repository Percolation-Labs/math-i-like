"""
rdft.ac.lerw_scaling
====================
Finite-size scaling of LERW on Z^d tori, driven by the enumeration
layer.

Wilson's algorithm and its simple LERW specialisation (Lawler-Sznitman)
realise the measure on simple a->b paths of Prop 1 of
paper/cfac/enumerative_boundary.tex exactly as samples. Running it
on Z^d_N finite-size grids and extracting the scaling of
<|gamma|> ~ |b|^{1/nu_d} recovers known exponents:

    d = 2:   nu_2 = 4/5 exactly (Kenyon 2000, Lawler-Schramm-Werner)
    d = 3:   nu_3 ~ 1.624 (Kozma 2007, numerical: Wilson 2010)

This module is the numerical face of the Tier-B CFAC reading: the
finite-size generating function is a ratio of Kirchhoff polynomials
(Prop 1), and its scaling can be read off by sampling the measure.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


# ------------------------------------------------------------------ #
#  Z^d torus construction
# ------------------------------------------------------------------ #

def zd_torus_adjacency(N: int, d: int) -> list[list[int]]:
    """Adjacency list of the Z^d torus with side N."""
    n = N ** d
    strides = [N ** i for i in range(d)]

    def idx(coord: tuple[int, ...]) -> int:
        return sum(c * s for c, s in zip(coord, strides))

    def coord(i: int) -> tuple[int, ...]:
        out = []
        for s in strides:
            out.append((i // s) % N)
        return tuple(out)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        c = coord(i)
        for axis in range(d):
            for shift in (-1, +1):
                c2 = list(c)
                c2[axis] = (c2[axis] + shift) % N
                j = idx(tuple(c2))
                adj[i].append(j)
    return adj


def zd_distance(i: int, j: int, N: int, d: int) -> int:
    """Graph distance on Z^d torus (minimum-image)."""
    strides = [N ** k for k in range(d)]
    total = 0
    for s in strides:
        a = (i // s) % N
        b = (j // s) % N
        diff = abs(a - b)
        total += min(diff, N - diff)
    return total


# ------------------------------------------------------------------ #
#  LERW sampling (Lawler's construction)
# ------------------------------------------------------------------ #

def sample_lerw_path(adj: list[list[int]],
                     a: int, b: int,
                     rng: np.random.Generator,
                     max_steps: int | None = None) -> list[int]:
    """Sample a single LERW from a to b.

    Run a simple random walk from a until it first visits b, then
    loop-erase. Loop-erasure is chronological: at each revisit of
    vertex v, truncate the path back to its first visit of v.
    """
    if max_steps is None:
        max_steps = 100 * len(adj)
    path: list[int] = [a]
    first_visit: dict[int, int] = {a: 0}
    current = a
    steps = 0
    while current != b:
        if steps >= max_steps:
            raise RuntimeError("LERW walker did not reach target in time")
        nbrs = adj[current]
        current = nbrs[rng.integers(len(nbrs))]
        if current in first_visit:
            # truncate path back to first visit
            cut = first_visit[current]
            for drop in path[cut + 1:]:
                del first_visit[drop]
            path = path[:cut + 1]
        else:
            first_visit[current] = len(path)
            path.append(current)
        steps += 1
    return path


def mean_lerw_length(adj: list[list[int]],
                     a: int, b: int,
                     n_samples: int,
                     rng: np.random.Generator,
                     max_steps: int | None = None,
                     ) -> tuple[float, float]:
    """Monte Carlo estimate of <|LERW_{a->b}|>. Returns (mean, sem)."""
    lengths = np.empty(n_samples, dtype=float)
    for s in range(n_samples):
        p = sample_lerw_path(adj, a, b, rng, max_steps=max_steps)
        lengths[s] = len(p) - 1
    return float(lengths.mean()), \
        float(lengths.std(ddof=1) / np.sqrt(n_samples))


# ------------------------------------------------------------------ #
#  Scaling extraction
# ------------------------------------------------------------------ #

def lerw_scaling_sweep(N: int, d: int,
                       r_vals: Sequence[int],
                       n_samples: int = 2000,
                       seed: int = 0,
                       ) -> dict[int, tuple[float, float]]:
    """For each r in r_vals, compute <|LERW_{0, (r,0,...,0)}|> on Z^d_N.

    Returns {r: (mean, sem)} suitable for log-log regression.
    """
    adj = zd_torus_adjacency(N, d)
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[float, float]] = {}
    for r in r_vals:
        # target: (r, 0, ..., 0)
        b_idx = r  # since strides = [1, N, N^2, ...] and coord[0]=r
        m, e = mean_lerw_length(adj, 0, b_idx, n_samples, rng)
        out[r] = (m, e)
    return out


def log_log_slope(r_vals: Sequence[int],
                  means: Sequence[float],
                  ) -> float:
    """Least-squares slope of log(mean) vs log(r)."""
    x = np.log(np.asarray(r_vals, dtype=float))
    y = np.log(np.asarray(means, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)
