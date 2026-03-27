"""
rdft.ac.transfer
================
Flajolet-Odlyzko transfer theorem for coefficient asymptotics.

Given a generating function A(z) with dominant singularity at z* of type:
    A(z) ~ K * (1 - z/z*)^alpha    as z -> z*

the transfer theorem gives:
    [z^n] A(z) ~ K / Gamma(-alpha) * n^{-alpha-1} * z*^{-n}

Key cases:
    alpha = 1/2  -> n^{-3/2}  (square-root branch: trees, epidemics, annihilation)
    alpha = -1/2 -> n^{-1/2}  (inverse square-root: random walk return)
    alpha = -1   -> z*^{-n}   (simple pole: geometric growth)
    alpha = -k   -> n^{k-1} * z*^{-n} (k-th order pole)

Reference: Flajolet & Sedgewick (2009), Theorem VI.3
"""

import sympy as sp
from sympy import Symbol, gamma as Gamma, Rational, pi, sqrt, oo
from typing import Dict, Optional, Tuple


class Singularity:
    """
    A singularity of a generating function.

    Parameters
    ----------
    z_star : location of the singularity
    alpha : exponent in (1 - z/z*)^alpha
    amplitude : the coefficient K in K * (1 - z/z*)^alpha
    """

    def __init__(self, z_star: sp.Expr, alpha: sp.Expr,
                 amplitude: sp.Expr = sp.S.One):
        self.z_star = z_star
        self.alpha = alpha
        self.amplitude = amplitude

    @property
    def singularity_type(self) -> str:
        """Human-readable singularity type."""
        a = self.alpha
        if a == Rational(1, 2):
            return 'square-root branch point'
        elif a == Rational(-1, 2):
            return 'inverse square-root'
        elif a == -1:
            return 'simple pole'
        elif a.is_integer and a < 0:
            return f'pole of order {-a}'
        elif a == 0:
            return 'logarithmic'
        else:
            return f'branch point with alpha={a}'

    def coefficient_asymptotics(self, n: sp.Symbol = None) -> sp.Expr:
        """
        Apply the transfer theorem to get [z^n] asymptotics.

        [z^n] A(z) ~ K / Gamma(-alpha) * n^{-alpha-1} * z*^{-n}

        Returns the asymptotic expression in n.
        """
        if n is None:
            n = Symbol('n', positive=True, integer=True)

        K = self.amplitude
        alpha = self.alpha
        z_star = self.z_star

        return K / Gamma(-alpha) * n**(-alpha - 1) * z_star**(-n)

    def coefficient_asymptotics_simplified(self, n: sp.Symbol = None) -> Dict[str, sp.Expr]:
        """
        Return the asymptotics in structured form.

        Returns dict with:
            'power_law_exponent': the n^{-alpha-1} exponent
            'exponential_growth': z*^{-n} growth rate
            'prefactor': K / Gamma(-alpha) prefactor
            'full': the complete asymptotic expression
        """
        if n is None:
            n = Symbol('n', positive=True, integer=True)

        alpha = self.alpha

        return {
            'power_law_exponent': -alpha - 1,
            'exponential_growth': self.z_star**(-1),
            'prefactor': self.amplitude / Gamma(-alpha),
            'full': self.coefficient_asymptotics(n),
        }

    def density_exponent(self, d: sp.Expr = None) -> Dict[str, sp.Expr]:
        """
        For reaction-diffusion processes, convert coefficient asymptotics
        to density decay exponent.

        If [z^n] ~ n^{-3/2} (square-root branch), then:
            P_surv(t) = sum_{n>t} n^{-3/2} ~ t^{-1/2}
            rho(t) ~ P_surv(t) ~ t^{-1/2}

        More generally, if [z^n] ~ n^{-beta}, then:
            P_surv(t) ~ t^{1-beta}  (for beta > 1)

        For d-dimensional processes (Gaussian integral):
            rho(t) ~ t^{-d/2}  (from integral d^d k / (s + Dk^2) ~ s^{d/2-1})
        """
        if d is None:
            d = Symbol('d', positive=True)

        alpha = self.alpha
        beta = -alpha - 1  # power law exponent: [z^n] ~ n^{-beta}
        # Note: for alpha=1/2, beta = -3/2, so |beta| = 3/2
        # We want the absolute exponent for the n^{-|beta|} decay
        neg_alpha_minus_1 = -alpha - 1  # this is the exponent directly

        # Survival probability exponent
        # If [z^n] ~ n^{neg_alpha_minus_1}, and neg_alpha_minus_1 < -1,
        # then sum_{n>t} ~ t^{neg_alpha_minus_1 + 1} = t^{-alpha}
        surv_exponent = -alpha  # from integrating n^{-alpha-1}

        result = {
            'coefficient_exponent': neg_alpha_minus_1,  # [z^n] ~ n^{this}
            'survival_exponent': surv_exponent,  # P_surv ~ t^{this}
        }

        # For square-root branch: alpha = 1/2
        # coefficient_exponent = -3/2, survival_exponent = -1/2
        # rho ~ t^{-1/2} in d=1
        if alpha == Rational(1, 2):
            result['density_exponent_1d'] = Rational(-1, 2)
            result['density_exponent_d'] = -d / 2
            result['upper_critical_dimension'] = sp.S(2)  # for 2A->0

        return result

    def __repr__(self) -> str:
        return (f'Singularity(z*={self.z_star}, type={self.singularity_type}, '
                f'alpha={self.alpha}, K={self.amplitude})')


def from_lagrange(lagrange_eq) -> Singularity:
    """
    Extract the dominant singularity from a LagrangeEquation.

    Any Lagrange equation T = z * phi(T) with phi''(T*) != 0 at the branch point
    produces a square-root branch point with alpha = 1/2.
    """
    sing_info = lagrange_eq.singularity_type()

    z_star = sing_info['z_star']

    if sing_info['type'] == 'square_root_branch':
        alpha = Rational(1, 2)
        amplitude = sing_info.get('amplitude', sp.S.One)
        return Singularity(z_star, alpha, amplitude)
    elif sing_info['type'] == 'cubic_branch':
        alpha = Rational(1, 3)
        return Singularity(z_star, alpha)
    else:
        # Generic case
        alpha = sing_info.get('alpha', Rational(1, 2))
        return Singularity(z_star, alpha)


# ------------------------------------------------------------------ #
#  Standard singularity table                                          #
# ------------------------------------------------------------------ #

TRANSFER_TABLE = {
    'simple_pole': {
        'alpha': -1,
        'asymptotics': 'C * z*^{-n}',
        'example': 'Fibonacci, geometric sequences',
    },
    'square_root_branch': {
        'alpha': Rational(1, 2),
        'asymptotics': 'C * n^{-3/2} * z*^{-n}',
        'example': 'SIR epidemic, trees, A+A->0',
    },
    'inverse_square_root': {
        'alpha': Rational(-1, 2),
        'asymptotics': 'C * n^{-1/2} * z*^{-n}',
        'example': 'Simple random walk return',
    },
    'logarithmic': {
        'alpha': 0,
        'asymptotics': 'C * n^{-1} * z*^{-n}',
        'example': 'Cycles, permutations',
    },
}
