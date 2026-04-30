"""
rdft.ac.sandpile_group
======================
Tier: 3 (research)

Sandpile-group animal for the LERW exponent: another failure mode.

Motivation. The Majumdar-Dhar bijection maps recurrent
configurations of the abelian sandpile model (ASM) on a graph G
to spanning trees of G; Wilson's theorem then maps the UST path
from any vertex to the sink to an LERW path. So ASM states,
spanning trees, and LERW all carry the same probabilistic content.

Algebraic upgrade. The recurrent ASM configurations form a finite
abelian group, the sandpile group:
    K(G) = Z^{V \\ sink} / Delta' Z^{V \\ sink},
with Delta' the Laplacian with the sink row/column removed. The
structure of K(G) --- its elementary divisors via Smith normal
form --- is a finite algebraic invariant, and the sequence
{K(Z^d_L)}_{L=1,2,...} is a concrete tower of finite algebraic
animals for the d-dimensional LERW scaling limit.

What we compute here. The reduced Laplacian for the d-dimensional
toroidal grid Z^d_L with one vertex as sink, then the Smith
normal form over Z. From the invariant factors we extract:
  - group order |K(G)| = number of spanning trees (Matrix-Tree);
  - number of non-trivial invariant factors;
  - largest invariant factor d_max;
  - multiplicity structure of the divisor sequence.

**Finding (negative).** For L up to 5 (d=2) and L up to 3 (d=3),
the group structure carries only the Matrix-Tree content --- the
log-order per volume converges to the bulk spanning-tree entropy,
and the divisor sequence has clear number-theoretic structure
(multiples of prime powers determined by L's factorisation), but
none of these scalars scales as a universal power of L that tracks
d_f^{(LERW)}.

The reason is structural. The sandpile GROUP is determined by the
Smith normal form of the reduced Laplacian --- a static algebraic
invariant. The LERW FRACTAL DIMENSION is a dynamical observable
(the expected length of the UST path between two points) that
depends on the full Laplacian spectrum and how the path
decomposition interacts with eigenvectors --- information the
group structure discards.

So: the bijection is real and deep, but the sandpile group is the
wrong algebraic object to extract d_f from. It does give the
**number** of spanning trees (Kirchhoff polynomial), and hence
the bulk entropy exponent, in closed form via Laplacian
eigenvalues --- but not the scaling of path lengths inside those
trees.

What WOULD carry d_f is the avalanche-size / wave-profile
distribution of the ASM dynamics. That is dynamical, not
algebraic --- the sandpile group doesn't see it.

This module is retained as a documented fifth failure mode in the
combinatorial-animal catalogue of paper/cfac/enumerative_boundary.tex.
"""

from __future__ import annotations
import numpy as np

# We defer the sympy import to function body to keep module import cheap.


def torus_reduced_laplacian(L: int, d: int) -> np.ndarray:
    """Reduced Laplacian of the d-dimensional torus Z^d_L with
    vertex 0 removed (treated as sink).
    """
    n = L ** d
    strides = [L ** i for i in range(d)]

    def coord(i: int) -> tuple[int, ...]:
        return tuple((i // s) % L for s in strides)

    def idx(c: tuple[int, ...]) -> int:
        return sum(ci * si for ci, si in zip(c, strides))

    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        c = coord(i)
        A[i, i] = 2 * d
        for axis in range(d):
            for shift in (-1, +1):
                c2 = list(c)
                c2[axis] = (c2[axis] + shift) % L
                j = idx(tuple(c2))
                A[i, j] -= 1
    return A[1:, 1:]


def sandpile_invariant_factors(Lred: np.ndarray) -> list[int]:
    """Smith-normal-form invariant factors of a reduced Laplacian.

    Uses sympy.matrices.normalforms.smith_normal_form; tractable
    for matrix size up to ~40-50 in reasonable time.
    """
    from sympy import Matrix
    from sympy.matrices.normalforms import smith_normal_form
    S = smith_normal_form(Matrix(Lred.tolist()))
    return [int(S[i, i]) for i in range(S.shape[0])]


def sandpile_group_stats(L: int, d: int) -> dict:
    """Compute sandpile-group statistics for Z^d_L torus.

    Returns a dict with:
      'size': rank of the reduced Laplacian (L^d - 1);
      'invariant_factors': full SNF diagonal;
      'non_trivial_factors': factors > 1;
      'num_non_trivial': count of factors > 1;
      'largest_factor': max invariant factor;
      'group_order': product of non-trivial factors (= Matrix-Tree
          spanning-tree count).
    """
    Lred = torus_reduced_laplacian(L, d)
    divs = sandpile_invariant_factors(Lred)
    non_trivial = [x for x in divs if x != 1]
    order = 1
    for x in non_trivial:
        order *= x
    return {
        'size': len(divs),
        'invariant_factors': divs,
        'non_trivial_factors': non_trivial,
        'num_non_trivial': len(non_trivial),
        'largest_factor': max(divs) if divs else 1,
        'group_order': order,
    }


def scaling_summary_2d(L_vals: list[int]) -> dict:
    """Sweep Z^2_L torus sandpile groups, collect scaling data."""
    out = {}
    for L in L_vals:
        out[L] = sandpile_group_stats(L, 2)
    return out


def scaling_summary_3d(L_vals: list[int]) -> dict:
    """Sweep Z^3_L torus sandpile groups, collect scaling data.

    Warning: SNF is expensive; L = 4 already takes minutes in
    pure-python SymPy.
    """
    out = {}
    for L in L_vals:
        out[L] = sandpile_group_stats(L, 3)
    return out
