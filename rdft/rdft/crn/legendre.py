"""
rdft.crn.legendre
=================

Symbolic Legendre transform ``Z[J] -> W[J] -> Gamma[Phi]`` for any CRN with a
0-d perturbative expansion.

The pipeline:
  1. ``build_Z(crn, ...)``       -- formal-power-series ``Z(J, Jt; g)``.
  2. ``log_W(Z)``                -- connected generating function ``W = log Z``.
  3. ``legendre_transform(W)``   -- 1PI generating functional ``Gamma``.
  4. ``coef(Gamma, J^a Jt^b, g^L)`` -- bidegree projection onto a Z-factor sector.

For Reggeon DP this reproduces the ``poc_legendre.py`` numbers:
  - W coefficient at g^5 J^2 Jt = -7608 (sum over 5 connected 2-loop V graphs)
  - Gamma coefficient at g^5 Phi^2 Phit = -504 (sum over 3 1PI graphs)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import sympy as sp


@dataclass
class LegendreResult:
    """Container for the symbolic Z, W, Gamma."""
    Z: sp.Expr
    W: sp.Expr
    Gamma: sp.Expr
    sources: Tuple[sp.Symbol, sp.Symbol]      # (J, Jt)
    fields:  Tuple[sp.Symbol, sp.Symbol]      # (Phi, Phit)
    coupling: sp.Symbol
    g_max: int
    j_max: int

    def W_coef(self, J_power: int, Jt_power: int, g_power: int) -> sp.Expr:
        """Coefficient [J^a Jt^b g^n] in W."""
        J, Jt = self.sources
        g = self.coupling
        poly = sp.Poly(sp.expand(self.W), g, J, Jt)
        return poly.coeff_monomial(g**g_power * J**J_power * Jt**Jt_power)

    def Gamma_coef(self, Phi_power: int, Phit_power: int, g_power: int) -> sp.Expr:
        """Coefficient [Phi^a Phit^b g^n] in Gamma."""
        Phi, Phit = self.fields
        g = self.coupling
        poly = sp.Poly(sp.expand(self.Gamma), g, Phi, Phit)
        return poly.coeff_monomial(g**g_power * Phi**Phi_power * Phit**Phit_power)


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------

def _truncate_g(expr: sp.Expr, g: sp.Symbol, N: int) -> sp.Expr:
    poly = sp.Poly(sp.expand(expr), g)
    out = sp.Integer(0)
    for monom, coef in poly.terms():
        if monom[0] <= N:
            out += coef * g**monom[0]
    return out


def _truncate_J(expr: sp.Expr, J: sp.Symbol, Jt: sp.Symbol, N: int) -> sp.Expr:
    poly = sp.Poly(sp.expand(expr), J, Jt)
    out = sp.Integer(0)
    for monom, coef in poly.terms():
        if sum(monom) <= N:
            out += coef * J**monom[0] * Jt**monom[1]
    return out


# ---------------------------------------------------------------------------
# Build Z by perturbative expansion
# ---------------------------------------------------------------------------

def reggeon_dp_Sint_op(J, Jt, g):
    """The differential operator -S_int(d/dJ, d/dJt) for 0-d Reggeon DP.

    -S_int = +g psit psi^2 - g psit^2 psi
            = +g d/dJt d/dJ d/dJ - g d/dJt d/dJt d/dJ
    (psi = d/dJ, psit = d/dJt.)
    """
    def op(F):
        return (g * sp.diff(F, Jt, J, J) - g * sp.diff(F, Jt, Jt, J))
    return op


def legendre_reggeon_dp(N_g: int = 5, J_max: int = 4) -> LegendreResult:
    """Build Z, W, Gamma symbolically for 0-d Reggeon DP up to order g^N_g."""
    J, Jt = sp.symbols("J Jt")
    Phi, Phit = sp.symbols("Phi Phit")
    g = sp.Symbol("g")
    Z0 = sp.exp(J * Jt)
    op = reggeon_dp_Sint_op(J, Jt, g)

    # Z = exp(-S_int(d/dJ)) Z_0
    Z = Z0
    running = Z0
    for n in range(1, N_g + 1):
        running = op(running)
        running = _truncate_g(running, g, N_g)
        Z = Z + running / sp.factorial(n)

    P_poly = sp.expand(Z * sp.exp(-J * Jt))
    P_poly = _truncate_g(P_poly, g, N_g)
    P_poly = _truncate_J(P_poly, J, Jt, J_max)

    # W = J*Jt + log P
    P_minus_one = sp.expand(P_poly - 1)
    log_P = sp.Integer(0)
    x_pow = sp.Integer(1)
    for k in range(1, N_g + 1):
        x_pow = sp.expand(x_pow * P_minus_one)
        x_pow = _truncate_g(x_pow, g, N_g)
        x_pow = _truncate_J(x_pow, J, Jt, J_max)
        log_P += (-1)**(k + 1) * x_pow / k
    W = sp.expand(J * Jt + log_P)
    W = _truncate_g(W, g, N_g)
    W = _truncate_J(W, J, Jt, J_max)

    # Legendre: Phi = dW/dJ, Phit = dW/dJt; invert; Gamma = J*Phi + Jt*Phit - W.
    dW_dJ = sp.expand(sp.diff(W, J))
    dW_dJt = sp.expand(sp.diff(W, Jt))

    J_inv, Jt_inv = Phit, Phi   # leading order
    for _ in range(N_g + 1):
        lhs1 = sp.expand(dW_dJ.subs([(J, J_inv), (Jt, Jt_inv)]))
        lhs1 = _truncate_g(lhs1, g, N_g)
        residual_phi = sp.expand(lhs1 - Phi)
        lhs2 = sp.expand(dW_dJt.subs([(J, J_inv), (Jt, Jt_inv)]))
        lhs2 = _truncate_g(lhs2, g, N_g)
        residual_phit = sp.expand(lhs2 - Phit)
        Jt_inv = sp.expand(Jt_inv - residual_phi)
        J_inv = sp.expand(J_inv - residual_phit)
        Jt_inv = _truncate_g(Jt_inv, g, N_g)
        J_inv = _truncate_g(J_inv, g, N_g)

    W_at_inv = W.subs([(J, J_inv), (Jt, Jt_inv)])
    Gamma = sp.expand(J_inv * Phi + Jt_inv * Phit - W_at_inv)
    Gamma = _truncate_g(Gamma, g, N_g)

    return LegendreResult(
        Z=Z, W=W, Gamma=Gamma,
        sources=(J, Jt), fields=(Phi, Phit), coupling=g,
        g_max=N_g, j_max=J_max,
    )
