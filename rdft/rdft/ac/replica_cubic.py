"""
rdft.ac.replica_cubic
=====================
Cubic-in-n analysis of the KPZ replica rate.

For the replica CRN of Prop 3 of paper/cfac/enumerative_boundary.tex,
the n-th replica rate has the small-n form

    lambda(n, beta) = a(beta) * n + b(beta) * n(n^2 - 1) / 6 + O(n^5),

a Bethe-ansatz / Lieb-Liniger signature of pairwise contact
attraction (Kardar 1987, Calabrese-Le Doussal-Rosso 2010). The
coefficient b(beta) is the binding energy of the n-body bound state,
and its analytic continuation n -> 0 gives the KPZ free-energy
cumulants.

This module extracts b(beta, W) from the exact transfer-matrix
spectrum at n = 1, 2, 3 and tracks its convergence as the lattice
width W -> infinity.

Two consistency checks witness the cubic ansatz:
  D(n) := lambda(n) - n * lambda(1)
  Pure cubic ansatz: D(n) = b * n(n^2 - 1) / 6, so D(2) = b and
                                                   D(3) = 4 * b.
  Hence D(3) / D(2) -> 4 as the model approaches the continuum
  (Kardar) limit.
"""

from __future__ import annotations
import numpy as np
from rdft.ac.replica_transfer import replica_rate


def replica_rates_n123(W: int, beta: float) -> tuple[float, float, float]:
    """Return (lambda(1), lambda(2), lambda(3)) at given (W, beta)."""
    return (replica_rate(1, W, beta),
            replica_rate(2, W, beta),
            replica_rate(3, W, beta))


def cubic_coefficient(W: int, beta: float) -> dict:
    """Extract the cubic-in-n coefficient b from lambda(n) at n=1,2,3.

    Returns a dict with:
      'lambda':   (lambda(1), lambda(2), lambda(3))
      'D2', 'D3': lambda(n) - n * lambda(1) for n = 2, 3
      'b_from_D2': D(2)              (= b under pure cubic ansatz)
      'b_from_D3': D(3) / 4          (= b under pure cubic ansatz)
      'b_lsq':    least-squares fit of D(n) = b * n(n^2-1)/6 across n=2,3
      'ratio':    D(3) / D(2)        (= 4 under pure cubic ansatz)
    """
    l1, l2, l3 = replica_rates_n123(W, beta)
    D2 = l2 - 2 * l1
    D3 = l3 - 3 * l1
    # Least-squares fit D(n) = b * c_n where c_2 = 1, c_3 = 4.
    c = np.array([1.0, 4.0])
    D = np.array([D2, D3])
    b_lsq = float((c @ D) / (c @ c))
    return {
        'lambda': (l1, l2, l3),
        'D2': D2,
        'D3': D3,
        'b_from_D2': D2,
        'b_from_D3': D3 / 4.0,
        'b_lsq': b_lsq,
        'ratio': D3 / D2 if D2 != 0 else float('inf'),
    }


def cubic_W_sweep(W_vals, beta: float) -> dict[int, dict]:
    """Sweep cubic_coefficient across a range of W at fixed beta.

    Returns {W: cubic_coefficient(W, beta)}. Useful to track the
    convergence of D(3)/D(2) -> 4 (Kardar continuum limit).
    """
    return {W: cubic_coefficient(W, beta) for W in W_vals}


def kardar_ratio_distance(W: int, beta: float) -> float:
    """|D(3)/D(2) - 4|: distance from pure Kardar / Bethe-ansatz cubic
    at given (W, beta). Should decrease monotonically as W grows
    (continuum limit) at fixed beta, with subleading model-dependent
    corrections.
    """
    res = cubic_coefficient(W, beta)
    return abs(res['ratio'] - 4.0)
