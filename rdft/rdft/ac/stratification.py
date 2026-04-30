"""
rdft.ac.stratification
======================
Tier: 1 (core)

The Puiseux stratification of polynomial DSE kernels.

Given a polynomial kernel phi(G) = 1 + a_1 G + ... + a_d G^d defining a
Lagrange equation G = z phi(G), this module identifies the Puiseux order k
of the dominant branch of F(G, z) = G - z phi(G) and locates the strata
C_k in coefficient space.

Core API
--------
- ``puiseux_order(phi_coeffs)`` -> (k, z_star) for the dominant branch.
- ``discriminant_in_z(phi_coeffs)`` -> coefficients of disc_G(F) as polynomial in z.
- ``on_stratum_C_k(phi_coeffs, k, tol)`` -> True iff phi sits algebraically on C_k.
- ``canonical_family(k, beta)`` -> phi_coeffs of (1+G)^k + beta G.

References
----------
Theorem A.2 of cfac_theorem.tex (stratification with d-dimensional dressing).
Banderier-Drmota (2015) for the dyadic ceiling on positive systems.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import sympy as sp
from math import comb


# ------------------------------------------------------------------ #
#  Discriminant utilities
# ------------------------------------------------------------------ #

def discriminant_in_z(phi_coeffs: Sequence[float]) -> list[float]:
    """Coefficients of disc_G(F) as polynomial in z, leading first.

    F(G, z) = G - z phi(G), where phi(G) = sum_i phi_coeffs[i] G^i.
    Uses sympy with rational coercion for numerical stability; the result
    is a list of floats suitable for numpy.roots.
    """
    G, z = sp.symbols('G z')
    rat = [sp.nsimplify(c, rational=True, tolerance=1e-12) for c in phi_coeffs]
    F = G - z * sum(c * G ** i for i, c in enumerate(rat))
    F_poly = sp.Poly(F, G, domain='QQ[z]')
    disc = F_poly.discriminant()
    disc_poly = sp.Poly(sp.expand(disc), z)
    return [float(c) for c in disc_poly.all_coeffs()]


def discriminant_roots(phi_coeffs: Sequence[float]) -> np.ndarray:
    """Nontrivial roots of disc_G(F) as a polynomial in z."""
    coeffs = discriminant_in_z(phi_coeffs)
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-12:
        coeffs = coeffs[1:]
    if len(coeffs) < 2:
        return np.array([], dtype=complex)
    roots = np.roots(coeffs)
    return np.array([r for r in roots if abs(r) > 1e-9])


def _root_multiplicity(roots: np.ndarray, z: complex, tol: float = 5e-3) -> int:
    return int(np.sum(np.abs(roots - z) < tol))


# ------------------------------------------------------------------ #
#  Stratification API
# ------------------------------------------------------------------ #

def puiseux_order(phi_coeffs: Sequence[float], tol: float = 5e-3) -> tuple[int, complex]:
    """Return (k_dom, z_star) for the dominant Puiseux branch.

    A 1/k-Puiseux branch corresponds to a (k-1)-fold root of the discriminant
    in z; multiplicity is detected numerically with tolerance ``tol``.

    Returns
    -------
    (k_dom, z_star) :
        k_dom = 1 + multiplicity at smallest |z*|; -1 if no nontrivial branch.
        z_star = the dominant branch point itself (smallest |z|).
    """
    roots = discriminant_roots(phi_coeffs)
    if len(roots) == 0:
        return -1, complex('nan')
    closest = min(roots, key=lambda r: abs(r))
    mult = _root_multiplicity(roots, closest, tol=tol)
    return 1 + mult, closest


def lambda_scgf(phi_coeffs: Sequence[float]) -> float:
    """SCGF lambda = -log|z_star| for the dominant branch."""
    _, z_star = puiseux_order(phi_coeffs)
    if not np.isfinite(abs(z_star)) or abs(z_star) == 0:
        return float('nan')
    return -float(np.log(abs(z_star)))


def on_stratum_C_k(phi_coeffs: Sequence[float], k: int, tol: float = 1e-6) -> bool:
    """Test whether phi satisfies the C_k algebraic conditions.

    For C_k: phi^(j)(G_star) = 0 for j = 2, ..., k-1, AND the branch condition
    G_star * phi'(G_star) = phi(G_star) holds.  Returns True iff a real
    G_star satisfies all these to tolerance ``tol``.

    Note: this only checks ALGEBRAIC presence of the C_k branch, not whether
    it is the DOMINANT branch.  Use puiseux_order() to check dominance.
    """
    G = sp.Symbol('G')
    rat = [sp.nsimplify(c, rational=True, tolerance=1e-12) for c in phi_coeffs]
    phi = sum(c * G ** i for i, c in enumerate(rat))

    # Need phi^(j)(G_star) = 0 for j = 2, ..., k-1 — this is k-2 conditions
    # If phi has degree < k, can't have a 1/k branch
    deg = sp.Poly(phi, G).degree()
    if deg < k:
        return False

    if k == 2:
        # Square-root branch is generic: any phi with non-trivial discriminant has it.
        return True

    # Solve phi^(k-1)(G) = 0 for G (the highest derivative-vanishing condition).
    # If phi^(k-1) is degree d - (k-1), there are deg - k + 1 roots.
    phi_k_minus_1 = sp.diff(phi, G, k - 1)
    candidates = sp.solve(phi_k_minus_1, G)

    for G_star in candidates:
        try:
            G_star = sp.simplify(G_star)
            # check all required derivatives vanish
            ok = all(
                abs(complex(sp.diff(phi, G, j).subs(G, G_star))) < tol
                for j in range(2, k)
            )
            # branch condition
            phi_val = complex(phi.subs(G, G_star))
            phi_p = complex(sp.diff(phi, G).subs(G, G_star))
            G_star_c = complex(G_star)
            branch_ok = abs(G_star_c * phi_p - phi_val) < tol
            if ok and branch_ok:
                return True
        except Exception:
            continue
    return False


# ------------------------------------------------------------------ #
#  Canonical family
# ------------------------------------------------------------------ #

def canonical_family(k: int, beta: float) -> list[float]:
    """Coefficients of phi_{k, beta}(G) = (1+G)^k + beta * G.

    This family sits on C_k for every beta and is the universal carrier for
    the 1/k-Puiseux stratum.  Branch point: G_star = -1, z_star = 1/beta.
    """
    coeffs = [float(comb(k, j)) for j in range(k + 1)]
    coeffs[1] += beta
    return coeffs


def canonical_z_star(k: int, beta: float) -> float:
    """z_star = 1/beta for the canonical family."""
    return 1.0 / beta if beta != 0 else float('inf')


# ------------------------------------------------------------------ #
#  Banderier-Drmota helpers
# ------------------------------------------------------------------ #

def is_dyadic(k: int) -> bool:
    """Returns True iff 1/k is dyadic (k is a power of 2).

    Banderier-Drmota: positive (N-algebraic) systems can only realise these k.
    """
    return k > 0 and (k & (k - 1)) == 0


def banderier_drmota_status(k: int) -> str:
    """Human-readable BD status for a Puiseux order."""
    if k <= 0:
        return 'no branch'
    if k == 2:
        return 'generic (DP class)'
    if is_dyadic(k):
        return f'dyadic (allowed to N-algebraic)'
    return f'non-dyadic (FORBIDDEN to positive systems)'


# ------------------------------------------------------------------ #
#  CRN -> phi_coeffs translator
# ------------------------------------------------------------------ #

def phi_from_reactions(reactions: list[tuple[int, int, float]],
                        max_degree: int = None) -> list[float]:
    """Translate a list of single-species reactions into phi(G) coefficients.

    Each reaction is (k_in, l_out, rate) representing k A -> l A.
    The Doi-Peliti generator
        Q = rate * ((z+1)^l - (z+1)^k) * D^k
    expands as
        rate * sum_{j=1}^{max(k,l)} (binom(l,j) - binom(k,j)) z^j D^k
    Each (z^j D^k) term is a vertex of leg-count m+n = j + k.  After the
    standard CFAC convention phi(G) = 1 + sum_{m+n>=3} g_{mn} G^{m+n-2}:

      - vertices with j + k = 2 (mass-type) modify the BARE PROPAGATOR G_0,
        not phi.  Skipped here.
      - vertices with j + k >= 3 contribute g_{j,k} G^{j+k-2} to phi(G).

    Parameters
    ----------
    reactions : list of (k_in, l_out, rate)
    max_degree : truncate phi at this degree (default: keep all).

    Returns
    -------
    list of floats: [phi_0, phi_1, ..., phi_d] = coefficients of phi(G) in G.
    By convention phi_0 = 1.
    """
    coefs: dict[int, float] = {0: 1.0}  # phi(0) = 1 by convention
    for k_in, l_out, rate in reactions:
        for j in range(1, max(k_in, l_out) + 1):
            # Skip mass-type vertices: m + n = j + k_in < 3 means j + k_in <= 2,
            # which contributes to G_0, not phi.
            if j + k_in < 3:
                continue
            cl = comb(l_out, j) if j <= l_out else 0
            ck = comb(k_in, j) if j <= k_in else 0
            c = rate * (cl - ck)
            if c == 0:
                continue
            power = j + k_in - 2
            if max_degree is not None and power > max_degree:
                continue
            coefs[power] = coefs.get(power, 0) + c
    if not coefs:
        return [1.0]
    d = max(coefs.keys())
    return [coefs.get(i, 0.0) for i in range(d + 1)]


def mass_shift_from_reactions(reactions: list[tuple[int, int, float]]) -> float:
    """Compute the propagator mass shift Sigma_0 from CRN reactions.

    Mass-type vertices (j + k_in <= 2) modify the bare propagator G_0:
        G_0 -> G_0 / (1 - G_0 * Sigma_0)
    where Sigma_0 = sum over (j=1, k_in=1) vertices of rate * (C(l,1) - C(k,1)).
    Non-positive Sigma_0 corresponds to net decay; positive to net growth.
    """
    shift = 0.0
    for k_in, l_out, rate in reactions:
        # mass shift from j=1, k_in=1 only (the (1,1) bilinear φ̃φ)
        if k_in == 1:
            shift += rate * (comb(l_out, 1) - comb(1, 1))
    return shift
