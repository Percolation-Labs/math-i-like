"""
rdft.ac.lagrange
================
Lagrange inversion and Dyson-Schwinger equations as combinatorial objects.

The Dyson-Schwinger equation (DSE) for the dressed propagator:
    G = G₀ · Φ(G)
is a Lagrange equation T = z · φ(T), where:
    T ↔ G (dressed propagator)
    z ↔ G₀ (bare propagator)
    φ(T) = 1 + Σ(T)·T (self-energy functional)

Lagrange inversion gives the perturbative expansion:
    [z^n] T = (1/n) [T^{n-1}] φ(T)^n

The singularity of T(z) — where the IFT fails — is the Landau pole /
non-perturbative scale. Its type (square-root branch, pole, essential)
determines the universality class.

References:
    Flajolet & Sedgewick (2009), §A.6 (Lagrange inversion)
    Yeats (2017), §2.3 (DSE as fixed-point equations)
    Amarteifio (2026), Tutorial §4 (AC recovers Doi-Peliti)
"""

import sympy as sp
from sympy import Symbol, sqrt, Rational, oo, series, solve, diff, factorial
from typing import Optional, Dict, List, Tuple, Union


class LagrangeEquation:
    """
    A Lagrange equation T = z · φ(T).

    Parameters
    ----------
    phi : sympy expression in variable T, the kernel function
    T_var : sympy Symbol for T (default: Symbol('T'))
    z_var : sympy Symbol for z (default: Symbol('z'))
    """

    def __init__(self, phi: sp.Expr, T_var: sp.Symbol = None, z_var: sp.Symbol = None):
        self.T = T_var or Symbol('T')
        self.z = z_var or Symbol('z')
        self.phi = phi  # φ(T), expression in self.T

    def coefficients(self, n_max: int = 10) -> List[sp.Expr]:
        """
        Compute [z^n] T(z) for n = 1, ..., n_max via Lagrange inversion.

        [z^n] T = (1/n) [T^{n-1}] φ(T)^n

        Returns list of coefficients [c_1, c_2, ..., c_{n_max}].
        """
        coeffs = []
        T = self.T

        for n in range(1, n_max + 1):
            # φ(T)^n expanded as a series in T
            phi_n = self.phi**n
            # We need the coefficient of T^{n-1} in the expansion of φ(T)^n
            # Use series expansion to extract it
            s = sp.series(phi_n, T, 0, n)
            coeff_T = s.coeff(T, n - 1)
            coeffs.append(sp.Rational(1, n) * coeff_T)

        return coeffs

    def branch_point(self) -> Union[Tuple[sp.Expr, sp.Expr], List[Tuple[sp.Expr, sp.Expr]]]:
        """
        Find the branch point (T*, z*) where the IFT fails.

        Conditions (Lagrange singularity):
            1 = z* · φ'(T*)     [IFT failure: ∂F/∂T = 0]
            T* = z* · φ(T*)     [the equation itself]

        Dividing: T* · φ'(T*) = φ(T*)
        Then: z* = T* / φ(T*)

        Returns (T_star, z_star) as sympy expressions.
        """
        T = self.T
        phi = self.phi
        phi_prime = sp.diff(phi, T)

        # Solve T* · φ'(T*) = φ(T*)
        eq = sp.Eq(T * phi_prime, phi)
        T_star_solutions = sp.solve(eq, T)

        results = []
        for T_star in T_star_solutions:
            # Must have φ(T*) ≠ 0 and T* ≠ 0
            phi_at_star = phi.subs(T, T_star)
            if phi_at_star != 0 and T_star != 0:
                z_star = T_star / phi_at_star
                results.append((sp.simplify(T_star), sp.simplify(z_star)))

        if len(results) == 1:
            return results[0]
        if len(results) == 0:
            raise ValueError("No finite branch point found")
        return results  # multiple branch points

    def singularity_type(self) -> Dict[str, sp.Expr]:
        """
        Determine the singularity type near the branch point.

        Near (T*, z*), expanding F(T, z) = T - z·φ(T) to second order:
            0 = F_z·(z* - z) + ½·F_TT·(δT)²

        Since F_z = -φ(T*) and F_TT = -z*·φ''(T*):
            δT ~ √(2φ(T*) / (z*·φ''(T*))) · (z* - z)^{1/2}

        If φ''(T*) ≠ 0: square-root branch point → n^{-3/2} asymptotics
        If φ''(T*) = 0: need to go to higher order
        """
        T = self.T
        phi = self.phi

        bp = self.branch_point()
        if isinstance(bp, list):
            bp = bp[0]  # take the dominant (smallest |z*|) branch point
        T_star, z_star = bp

        phi_pp = sp.diff(phi, T, 2)
        phi_pp_at_star = phi_pp.subs(T, T_star)
        phi_at_star = phi.subs(T, T_star)

        result = {
            'T_star': T_star,
            'z_star': z_star,
            'phi_at_star': phi_at_star,
            'phi_double_prime': sp.simplify(phi_pp_at_star),
        }

        if phi_pp_at_star != 0:
            result['type'] = 'square_root_branch'
            result['alpha'] = sp.Rational(1, 2)
            # Amplitude of the square root
            C_sq = 2 * phi_at_star / (z_star * phi_pp_at_star)
            result['amplitude'] = sp.simplify(sp.sqrt(C_sq))
        else:
            # Check third derivative
            phi_ppp = sp.diff(phi, T, 3)
            phi_ppp_at_star = phi_ppp.subs(T, T_star)
            if phi_ppp_at_star != 0:
                result['type'] = 'cubic_branch'
                result['alpha'] = sp.Rational(1, 3)
            else:
                result['type'] = 'higher_order'
                result['alpha'] = None

        return result

    def summary(self) -> str:
        """Human-readable summary."""
        T, z = self.T, self.z
        lines = [f'Lagrange Equation: {T} = {z} * ({self.phi})']

        coeffs = self.coefficients(5)
        lines.append('First 5 coefficients [z^n]T:')
        for i, c in enumerate(coeffs, 1):
            lines.append(f'  n={i}: {c}')

        try:
            bp = self.branch_point()
            if isinstance(bp, tuple):
                T_star, z_star = bp
                lines.append(f'Branch point: T* = {T_star}, z* = {z_star}')
            sing = self.singularity_type()
            lines.append(f'Singularity type: {sing["type"]}')
            if sing.get('amplitude'):
                lines.append(f'  T ~ {sing["T_star"]} - {sing["amplitude"]} * sqrt(z* - z)')
        except Exception as e:
            lines.append(f'Branch point analysis: {e}')

        return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Standard Lagrange equations for known processes                     #
# ------------------------------------------------------------------ #

def sir_epidemic(R0: sp.Expr = None) -> LagrangeEquation:
    """
    SIR epidemic final-size GF.

    T(z) = z * exp(R_0 * (T - 1))

    phi(T) = exp(R_0 * (T - 1))

    Branch point: T* = 1/R_0, z* = (1/R_0) * e^{1 - 1/R_0}
    Singularity: square-root branch -> n^{-3/2} * z*^{-n}
    At R_0 = 1: P(size = n) ~ n^{-3/2} (Borel distribution)
    """
    T = Symbol('T')
    if R0 is None:
        R0 = Symbol('R_0', positive=True)
    phi = sp.exp(R0 * (T - 1))
    return LagrangeEquation(phi, T)


def first_passage_1d() -> LagrangeEquation:
    """
    First-passage GF for symmetric 1D random walk.

    F(z) = z * (1 + F^2) / 2

    phi(F) = (1 + F^2) / 2

    Branch point: F* = 1, z* = 1
    Singularity: square-root -> f_n ~ n^{-3/2}
    Integrating: P_surv(t) ~ t^{-1/2} -> rho(t) ~ t^{-1/2}

    This gives the 2A->0 density decay exponent via AC.
    """
    F = Symbol('F')
    phi = (1 + F**2) / 2
    return LagrangeEquation(phi, F)


def pair_annihilation_dse(d: sp.Expr = None) -> LagrangeEquation:
    """
    DSE for pair annihilation 2A->0.

    The dressed propagator G satisfies:
        G = G_0 * (1 + lambda * G^2)

    where lambda is the effective coupling (includes loop integral factor).

    phi(G) = 1 + lambda * G^2

    Branch point: G* = 1/(2*lambda*z*), giving z* related to Landau pole.
    """
    G = Symbol('G')
    lam = Symbol('lambda', positive=True) if d is None else d
    phi = 1 + lam * G**2
    return LagrangeEquation(phi, G)


def general_reaction_dse(vertex_types: Dict[Tuple[int, int], sp.Expr]) -> LagrangeEquation:
    """
    Construct the DSE Lagrange equation from a set of vertex types.

    For a theory with vertices {(m,n): g_{mn}}, the self-energy is:
        Sigma(G) = sum_{vertices} g_{mn} * G^{m+n-2}  (schematic, one-loop)

    And the DSE: G = G_0 * (1 + Sigma(G) * G) = G_0 * phi(G)

    This is a simplified version; the full DSE requires loop integrals.
    """
    G = Symbol('G')
    sigma = sp.S.Zero
    for (m, n), g in vertex_types.items():
        if m + n >= 3:  # interaction vertices only
            # One-loop contribution: g * G^{n_legs - 2}
            sigma += g * G**(m + n - 2)

    phi = 1 + sigma * G
    return LagrangeEquation(phi, G)
