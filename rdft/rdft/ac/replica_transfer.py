"""
rdft.ac.replica_transfer
========================
Tier: 3 (research)

Transfer-matrix evaluation of the KPZ n-walker replica CRN.

For the lattice directed polymer with Gaussian disorder (variance 1),
the replica moment E[Z_N^n] is the partition function of n
distinguishable walkers with pairwise contact attraction of strength
beta^2 (Prop 3 of paper/cfac/enumerative_boundary.tex). This module
builds the transfer matrix on configuration space {-W,...,W}^n and
extracts

    lambda(n, beta; W) = lim_{N -> oo} (1/N) log E[Z_N^n]

from the Perron eigenvalue. That limit is the "replica free energy"
whose analytic continuation at n -> 0 gives the KPZ free-energy
cumulants; see e.g. Krajenbrink (2020) for the continuum version.

At beta = 0 the walkers are free: lambda(n, 0; W) = n * log(lambda_1)
where lambda_1 is the Perron eigenvalue of the single-walker
transfer matrix on [-W, W] (nearest-neighbour random walk, i.e.,
path discrete Laplacian with reflecting boundaries).

The module is small by design: it uses dense matrices and is
intended for n, W small enough that (2W+1)^n fits in memory.
"""

from __future__ import annotations
from itertools import product
import numpy as np


def _neighbours(x: int, W: int) -> list[int]:
    out = []
    if x - 1 >= -W:
        out.append(x - 1)
    if x + 1 <= W:
        out.append(x + 1)
    return out


def _coincidence_weight(config: tuple[int, ...], beta: float) -> float:
    counts: dict[int, int] = {}
    for p in config:
        counts[p] = counts.get(p, 0) + 1
    k = sum(c * c for c in counts.values())
    return np.exp(0.5 * beta * beta * k)


def build_replica_transfer(n: int, W: int, beta: float) -> np.ndarray:
    """Return T of shape (S, S) where S = (2W+1)^n.

    T[c', c] is the unnormalised weight of transitioning from
    configuration c (an n-tuple of lattice sites) to c' in one time
    step: the product of nearest-neighbour step indicators times the
    coincidence weight at the new configuration.

    The partition function satisfies
        E[Z_N^n] = u^T T^N v
    for suitable boundary vectors u, v (all-ones for free boundary,
    delta at origin for point-to-line).
    """
    sites = list(range(-W, W + 1))
    configs = list(product(sites, repeat=n))
    cfg_idx = {c: i for i, c in enumerate(configs)}
    S = len(configs)
    T = np.zeros((S, S))
    for c in configs:
        # for each walker, pick a neighbour (independent)
        neighbour_lists = [tuple(_neighbours(x, W)) for x in c]
        for cp in product(*neighbour_lists):
            w = _coincidence_weight(cp, beta)
            T[cfg_idx[cp], cfg_idx[c]] += w
    return T


def replica_rate(n: int, W: int, beta: float) -> float:
    """lambda(n, beta) = log( largest real eigenvalue of T ).

    For the attractive coincidence weight T is a nonneg matrix with
    a real dominant eigenvalue (Perron-Frobenius on each component).
    """
    T = build_replica_transfer(n, W, beta)
    eigs = np.linalg.eigvals(T)
    dominant = np.max(np.abs(eigs))
    return float(np.log(dominant))


def free_walker_rate(W: int) -> float:
    """lambda for a single walker at beta = 0 on [-W, W].

    Nearest-neighbour walk with reflecting boundary; dominant
    eigenvalue is 2 cos(pi / (2W + 2))  --- classical tight-binding
    on an open chain of length 2W+1 with unit hoppings.
    """
    L = 2 * W + 1
    return float(np.log(2 * np.cos(np.pi / (L + 1))))


def replica_partition(n: int, W: int, beta: float, N: int,
                      origin_weight: bool = True,
                      ) -> float:
    """E[Z_N^n] by power iteration on the transfer matrix.

    boundary choice: both walker tuples start at the origin
    (origin_weight=True), open ends at time N.
    """
    T = build_replica_transfer(n, W, beta)
    sites = list(range(-W, W + 1))
    configs = list(product(sites, repeat=n))
    S = len(configs)
    v = np.zeros(S)
    origin = tuple(0 for _ in range(n))
    i0 = configs.index(origin)
    v[i0] = _coincidence_weight(origin, beta)  # t=0 weight at origin
    # NumPy's matmul on very sparse dense matrices can emit spurious
    # divide-by-zero / overflow warnings from the underlying BLAS
    # while still returning correct results; silence them.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(N):
            v = T @ v
    return float(v.sum())
