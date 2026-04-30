"""
rdft.ac.matrix_model
====================
Tier: 3 (research)

Ribbon-graph enumeration for TAP complexity.

Implements Proposition 2 of paper/cfac/enumerative_boundary.tex.
The Harer-Zagier identity expresses GUE moments as a genus-graded
sum of rooted one-face map counts:

    E[ tr(M^{2k}) ] = sum_{g >= 0}  eps_g(k) * N^{k + 1 - 2g}

where M is an N x N Gaussian unitary ensemble with normalisation
<M_ij M_kl> = delta_{il} delta_{jk}, and eps_g(k) counts rooted
one-face orientable maps with k edges and genus g.

Harer-Zagier recursion (Harer-Zagier 1986, Itzykson-Zuber form):

    (k + 1) eps_g(k) = 2 (2k - 1) eps_g(k - 1)
                     + (k - 1)(2k - 1)(2k - 3) eps_{g-1}(k - 2)

with boundary conditions eps_0(0) = 1, eps_g(k) = 0 for k < 2g.

The planar (g = 0) specialisation gives the Catalan numbers:

    eps_0(k) = C_k = (2k)! / (k! (k+1)!).

CFAC reading: the right-hand side is a ribbon-graph generating
function - diagrams the existing pipeline handles natively once a
matrix-model action is specified. The Kac-Rice / TAP complexity
density is (via cluster expansion and a Legendre transform) a
functional of these map counts.
"""

from __future__ import annotations
from functools import lru_cache
from math import factorial


# ------------------------------------------------------------------ #
#  Catalan numbers (planar base case)
# ------------------------------------------------------------------ #

@lru_cache(maxsize=None)
def catalan(k: int) -> int:
    if k < 0:
        return 0
    return factorial(2 * k) // (factorial(k) * factorial(k + 1))


# ------------------------------------------------------------------ #
#  Harer-Zagier recursion
# ------------------------------------------------------------------ #

@lru_cache(maxsize=None)
def rooted_one_face_maps(k: int, g: int) -> int:
    """eps_g(k): number of rooted one-face orientable maps with k
    edges on a surface of genus g.
    """
    if g < 0 or k < 2 * g:
        return 0
    if g == 0:
        return catalan(k)
    rhs = 2 * (2 * k - 1) * rooted_one_face_maps(k - 1, g) \
        + (k - 1) * (2 * k - 1) * (2 * k - 3) * \
            rooted_one_face_maps(k - 2, g - 1)
    assert rhs % (k + 1) == 0, \
        f"HZ recursion non-integer at k={k}, g={g}: rhs={rhs}"
    return rhs // (k + 1)


# ------------------------------------------------------------------ #
#  GUE moment via genus expansion
# ------------------------------------------------------------------ #

def gue_trace_moment(N: int, k: int) -> float:
    """E[ tr(M^{2k}) ] for GUE(N), variance 1 per entry.

    Genus expansion: sum over g from 0 to floor(k/2) of
        eps_g(k) * N^{k + 1 - 2g}.
    """
    total = 0.0
    for g in range(0, k // 2 + 1):
        total += rooted_one_face_maps(k, g) * (N ** (k + 1 - 2 * g))
    return total


def planar_moment(k: int) -> int:
    """Leading-N moment: eps_0(k) = C_k (Catalan)."""
    return catalan(k)


# ------------------------------------------------------------------ #
#  Log-characteristic-polynomial expansion
# ------------------------------------------------------------------ #

def log_det_series_coeff(N: int, k: int) -> float:
    """Coefficient of 1/f^(2k) in the large-|f| expansion of

        E[ log |det(M - f I)| ]
          = N log|f| - sum_{k >= 1} (1/(2k)) * E[tr(M^{2k})] / f^(2k)
          + O(odd moments).

    For GUE the odd trace moments vanish. Returns the coefficient
    - E[tr(M^{2k})] / (2k) of 1/f^{2k}.
    """
    return -gue_trace_moment(N, k) / (2 * k)
