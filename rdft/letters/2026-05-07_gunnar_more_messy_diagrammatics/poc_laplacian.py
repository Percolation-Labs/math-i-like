"""
Verification of the n-rail Laplacian formula for Gunnar's "more messy
diagrammatics" challenge (note of 7 May 2026).

The "1's path" is a walk on rails {0, 1, ..., n-1}. At each vertex it
either:
  - stays on its current rail a, with a bridge to some j != a, weight +v_{aj}
  - swaps from rail a to rail b != a, weight -v_{ab}

The walk starts and ends at rail 0. The order-m weighted sum is

    W^{(n)}_m = sum over walks of (sign)(prod v_{ij}) = (L^m)_{00}

where L is the weighted graph Laplacian L = D - A on K_n with edge weights
v_{ij}. For uniform v_{ij} = v this gives W^{(n)}_m = (n-1)/n * (n v)^m.

This script:
  1. brute-force enumerates the rail walks for small (n, m) and checks the
     uniform formula numerically;
  2. enumerates symbolically over v_{ij} and checks against (L^m)_{00};
  3. reports the visit-pattern multiplicities at m=5.
"""

from __future__ import annotations
from fractions import Fraction
from itertools import product
from collections import Counter

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Direct enumeration of signed rail walks
# ---------------------------------------------------------------------------

def enumerate_walks(n: int, m: int):
    """Yield (sign, sequence_of_v_pairs) for each length-m walk 0 -> 0.

    A walk is a sequence of m moves. Each move is either:
      ('stay', a, j)  with j != a, weight +v_{aj}
      ('swap', a, b)  with b != a, weight -v_{ab}
    Starting state is rail 0; final state must be rail 0.
    """
    if m == 0:
        yield 1, ()
        return
    others = lambda a: [r for r in range(n) if r != a]

    def recurse(state, depth, sign, weights):
        if depth == m:
            if state == 0:
                yield sign, tuple(weights)
            return
        # stay: bridge to any j != state, weight +v_{state, j}
        for j in others(state):
            yield from recurse(state, depth + 1, sign,
                               weights + [tuple(sorted((state, j)))])
        # swap: change rail to any b != state, weight -v_{state, b}
        for b in others(state):
            yield from recurse(b, depth + 1, -sign,
                               weights + [tuple(sorted((state, b)))])

    yield from recurse(0, 0, 1, [])


def enumerate_uniform(n: int, m: int) -> Fraction:
    """Sum weighted walks with v_{ij} = 1; return the integer count (signed)."""
    return sum(sign for sign, _ in enumerate_walks(n, m))


def enumerate_symbolic(n: int, m: int):
    """Sum weighted walks symbolically in v_{ij} (i<j)."""
    v = {(i, j): sp.Symbol(f"v_{i}{j}", real=True)
         for i in range(n) for j in range(i + 1, n)}
    expr = sp.Integer(0)
    for sign, weights in enumerate_walks(n, m):
        term = sp.Integer(sign)
        for pair in weights:
            term *= v[pair]
        expr += term
    return sp.expand(expr), v


# ---------------------------------------------------------------------------
# 2. Laplacian construction and matrix-power answer
# ---------------------------------------------------------------------------

def laplacian_uniform(n: int, v: float | int = 1) -> np.ndarray:
    L = -v * (np.ones((n, n)) - np.eye(n))
    np.fill_diagonal(L, (n - 1) * v)
    return L


def laplacian_symbolic(n: int, v_dict: dict) -> sp.Matrix:
    L = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i, i] = sum(v_dict[tuple(sorted((i, k)))]
                              for k in range(n) if k != i)
            else:
                L[i, j] = -v_dict[tuple(sorted((i, j)))]
    return L


# ---------------------------------------------------------------------------
# 3. Tests
# ---------------------------------------------------------------------------

def test_uniform():
    print("=" * 70)
    print("TEST 1  uniform v=1: enumeration vs (n-1)/n * n^m")
    print("=" * 70)
    print(f"{'n':>3} {'m':>3} {'enumeration':>14} {'(n-1)/n n^m':>14}  match")
    for n in (2, 3, 4, 5):
        for m in range(0, 8):
            count = enumerate_uniform(n, m)
            if m == 0:
                predicted = 1
            else:
                predicted = Fraction(n - 1, n) * n ** m
            ok = (count == predicted)
            print(f"{n:>3} {m:>3} {count:>14} {str(predicted):>14}  {ok}")
            assert ok, f"mismatch at n={n}, m={m}: {count} vs {predicted}"


def test_symbolic_n3():
    print()
    print("=" * 70)
    print("TEST 2  n=3: enumeration as polynomial in v_{ij} vs (L^m)_{00}")
    print("=" * 70)
    n = 3
    for m in range(0, 5):
        enum, v_dict = enumerate_symbolic(n, m)
        L = laplacian_symbolic(n, v_dict)
        Lm = L ** m
        predicted = sp.expand(Lm[0, 0])
        diff = sp.expand(enum - predicted)
        ok = (diff == 0)
        print(f"  m={m}:  enumeration polynomial  =  (L^m)_00  ?  {ok}")
        if m <= 2:
            print(f"    enumeration: {enum}")
            print(f"    (L^m)_00:    {predicted}")
        assert ok


def test_visit_patterns_m5():
    print()
    print("=" * 70)
    print("TEST 3  n=3, m=5: visit-pattern multiplicities (p(5)=7 partitions)")
    print("=" * 70)
    n, m = 3, 5
    # For each walk, record the multiset of "rail-visit chunks" by the 1.
    # That is: split the walk into maximal runs on the same rail; the multiset
    # of chunk lengths is the visit pattern.
    pattern_count: Counter = Counter()
    for sign, weights in enumerate_walks(n, m):
        # Reconstruct the rail sequence from the weights pattern.
        # We do not need it from `weights` directly --- redo the enumeration
        # tracking the trajectory.
        pass

    # Better: re-enumerate while tracking the sequence of rails.
    def trajectories(state, depth, traj, sign):
        if depth == m:
            if state == 0:
                yield sign, tuple(traj)
            return
        others = [r for r in range(n) if r != state]
        # stay (bridge): rail of 1 unchanged
        for _ in others:
            yield from trajectories(state, depth + 1, traj + [state], sign)
        # swap: rail of 1 changes to b
        for b in others:
            yield from trajectories(b, depth + 1, traj + [b], -sign)

    pattern_count = Counter()
    pattern_signed = Counter()
    for sign, traj in trajectories(0, 0, [0], 1):
        # traj has length m+1; chunks are runs of equal consecutive entries.
        chunks = []
        cur = traj[0]
        run = 1
        for r in traj[1:]:
            if r == cur:
                run += 1
            else:
                chunks.append(run)
                cur = r
                run = 1
        chunks.append(run)
        # The multiset of chunk lengths characterises the visit pattern.
        # (Sum of chunks = m+1.)
        key = tuple(sorted(chunks, reverse=True))
        pattern_count[key] += 1
        pattern_signed[key] += sign

    print(f"  trajectories enumerated:  {sum(pattern_count.values())}")
    print(f"  total signed:             {sum(pattern_signed.values())}")
    expected_total = Fraction(2, 3) * 3 ** 5
    print(f"  expected (2/3) * 3^5 =    {expected_total}")
    print()
    print(f"  {'visit pattern (chunk lengths)':32}  {'#walks':>8}  {'signed sum':>12}")
    for key in sorted(pattern_count, key=lambda k: (-len(k), k)):
        print(f"  {str(key):32}  {pattern_count[key]:>8}  {pattern_signed[key]:>12}")
    assert sum(pattern_signed.values()) == expected_total


if __name__ == "__main__":
    test_uniform()
    test_symbolic_n3()
    test_visit_patterns_m5()
    print()
    print("All checks passed.")
