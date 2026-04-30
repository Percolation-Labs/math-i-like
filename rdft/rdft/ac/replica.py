"""
rdft.ac.replica
===============
Tier: 3 (research)

Replicated directed polymer as an n-species CRN.

Implements Proposition 3 of paper/cfac/enumerative_boundary.tex.
For the lattice directed polymer with i.i.d. Gaussian noise of
variance 1, the n-th replica moment is

    E[ Z_N^n ] = sum_{(w^1, ..., w^n)} prod_t
                    exp( (beta^2 / 2) * k_t(w) ),

where k_t(w) = sum_x (#{i : w^i_t = x})^2 is the coincidence
count at time t. The sum ranges over n-tuples of nearest-neighbour
paths on a 1D lattice of width W starting at the origin.

In CFAC language, this is the partition function of a CRN with n
distinguishable walker species and a single symmetric binary
vertex i + j -> i + j of rate beta^2 attaching at coincident sites.

The module implements the enumeration directly (exact for moderate
N, n, W) and a Monte Carlo cross-check over the disorder for
verification.
"""

from __future__ import annotations
from itertools import product
import numpy as np


def _step_positions(x: int, W: int) -> tuple[int, ...]:
    """Nearest-neighbour steps on a 1D lattice of width W,
    interval [-W, W] with reflecting ends.
    """
    out = []
    if x - 1 >= -W:
        out.append(x - 1)
    if x + 1 <= W:
        out.append(x + 1)
    return tuple(out)


def replica_moment_exact(N: int, n: int, beta: float,
                         W: int = 4) -> float:
    """E[Z_N^n] by direct enumeration of n-tuples of walks of length
    N starting at origin on [-W, W].

    Note: this grows as (2^n)^N * (steps per walker), so keep N and n
    small (N <= 6, n <= 3 is comfortable).
    """
    # Enumerate single-walker trajectories of length N from x=0
    trajs: list[tuple[int, ...]] = []
    def dfs(pos: int, depth: int, hist: list[int]):
        if depth == N:
            trajs.append(tuple(hist))
            return
        for nxt in _step_positions(pos, W):
            hist.append(nxt)
            dfs(nxt, depth + 1, hist)
            hist.pop()
    dfs(0, 0, [0])

    total = 0.0
    for tup in product(trajs, repeat=n):
        # coincidence count at each time
        log_w = 0.0
        for t in range(N + 1):
            # Count multiplicities of positions at time t across replicas
            counts: dict[int, int] = {}
            for i in range(n):
                x = tup[i][t]
                counts[x] = counts.get(x, 0) + 1
            k_t = sum(c * c for c in counts.values())
            log_w += 0.5 * beta * beta * k_t
        total += np.exp(log_w)
    return total


def replica_moment_mc(N: int, n: int, beta: float,
                      W: int = 4,
                      n_samples: int = 20000,
                      rng: np.random.Generator | None = None,
                      ) -> tuple[float, float]:
    """Monte Carlo estimate of E[Z_N^n] by direct disorder sampling.

    For each disorder realisation, compute Z_N by transfer matrix,
    then average Z_N^n over samples. Used to cross-check
    replica_moment_exact.

    Returns (mean, standard error).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sites = np.arange(-W, W + 1)
    n_sites = len(sites)
    origin_idx = W
    # Transfer matrix: nearest neighbour on [-W, W]
    T = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        if i - 1 >= 0:
            T[i, i - 1] = 1.0
        if i + 1 < n_sites:
            T[i, i + 1] = 1.0

    samples_Zn = np.empty(n_samples)
    for s in range(n_samples):
        # eta[t, x] for t = 0, 1, ..., N (length N+1 time-slices,
        # since we include the starting site at t=0 by convention).
        eta = rng.standard_normal((N + 1, n_sites))
        # psi[t] = weight distribution over end positions after t steps
        psi = np.zeros(n_sites)
        psi[origin_idx] = np.exp(beta * eta[0, origin_idx])
        for t in range(1, N + 1):
            psi = T @ psi
            psi = psi * np.exp(beta * eta[t, :])
        Z = psi.sum()
        samples_Zn[s] = Z ** n

    mean = samples_Zn.mean()
    sem = samples_Zn.std(ddof=1) / np.sqrt(n_samples)
    return mean, sem


def coincidence_count(positions: list[int]) -> int:
    """k(x) = sum_x (#{i : positions_i = x})^2 for a single time slice."""
    counts: dict[int, int] = {}
    for p in positions:
        counts[p] = counts.get(p, 0) + 1
    return sum(c * c for c in counts.values())
