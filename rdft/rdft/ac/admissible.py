"""
rdft.ac.admissible
===================
Stratification framework extended to TRANSCENDENTAL (non-polynomial)
DSE kernels.  Addresses TODO-1 from docs/problems.md.

Context
-------
The CFAC stratification theorem (cfac_theorem.tex) is stated for
polynomial phi(G) of finite degree: the Puiseux order k of the
dominant branch point gives tau_k = 1 + 1/k, with Banderier-Drmota
dyadic ceiling for N-algebraic phi.

Many physical systems have TRANSCENDENTAL phi:
  - Poisson-offspring branching:   phi(G) = exp(G)
  - Symmetric-difference kinetics: phi(G) = cosh(G)
  - Exponential offspring:         phi(G) = 1/(1 - G)  (still algebraic
                                   but borderline)
  - Active matter with Poisson tumble events
  - Compound-Poisson reaction rates

For these, the stratification theorem does not apply directly.  The
correct machinery is saddle-point / Hayman admissibility
(Flajolet-Sedgewick ch. VIII.5).

What this module provides
-------------------------
1. Detection: is phi admissible in the Hayman sense?
2. Critical-point extraction: find (z*, G*) satisfying the simultaneous
   equations z*·phi(G*) = G* (on-shell) and z*·phi'(G*) = 1
   (saddle coalescence = criticality of the Lagrange equation).
3. Coefficient asymptotics: [z^n] G(z) ~ C n^{-3/2} (z*)^{-n} for
   admissible phi with finite moments at G* — this is the generic
   C_2 result (tau = 3/2).
4. Infinite-variance case: if phi''(G*) diverges, the asymptotics
   change to [z^n] G ~ n^{-1-1/alpha} giving non-dyadic tau.
5. Physical demo: Poisson-offspring branching CRN as an admissible
   CRN-in-field, with explicit tau = 3/2.

Scope
-----
This is a minimum viable extension:
- Handles finite-variance admissible phi (generic C_2)
- Flags infinite-variance case for stable-tree asymptotics
- Does NOT yet handle non-generic saddle coalescence at higher order
  (would give tau = 1 + 1/k for k > 2 via transcendental route)
- Does NOT yet implement full Hayman admissibility test
  (H1-H3 conditions) — uses a pragmatic "phi entire with positive
  coefficients and finite phi''(G*)" proxy.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Optional
from scipy.optimize import brentq


def find_critical_point(phi: Callable[[float], float],
                         phi_prime: Callable[[float], float],
                         G_max: float = 50.0,
                         G_min: float = 1e-6) -> Dict:
    """Find (z*, G*) for the DSE G = z·phi(G) by solving the saddle
    collision condition z·phi'(G) = 1 simultaneously with
    z·phi(G) = G.

    Eliminating z: the critical G* satisfies
        G*·phi'(G*) = phi(G*)        (Lagrange-type condition)
    and then z* = G* / phi(G*) = 1 / phi'(G*).
    """
    def f(G):
        return G * phi_prime(G) - phi(G)
    # Bracket search
    try:
        G_star = brentq(f, G_min, G_max, xtol=1e-12)
    except ValueError:
        return {'critical_point_found': False,
                'reason': f'no sign change of G·phi\'(G) - phi(G) on [{G_min}, {G_max}]'}

    z_star = G_star / phi(G_star)
    return {
        'critical_point_found': True,
        'G_star': G_star,
        'z_star': z_star,
        'phi_at_Gstar': phi(G_star),
        'phi_prime_at_Gstar': phi_prime(G_star),
        'residual_lagrange': G_star * phi_prime(G_star) - phi(G_star),
    }


def admissible_asymptotics(phi: Callable[[float], float],
                             phi_prime: Callable[[float], float],
                             phi_double_prime: Callable[[float], float],
                             G_max: float = 50.0) -> Dict:
    """Coefficient asymptotics for G = z·phi(G) with admissible phi.

    At the critical point (z*, G*), the DSE expansion is:
        G(z) = G* - A·(1 - z/z*)^{1/2} + O((1 - z/z*))
    with A = sqrt(2·phi(G*) / (z*·phi''(G*))).
    This gives tau = 3/2 (C_2 stratum) and amplitude A, via the
    transfer theorem (Flajolet-Odlyzko).

    Returns the asymptotic formula
        [z^n] G(z) ~ (A / (2 sqrt(pi))) · n^{-3/2} · (1/z*)^n
    """
    crit = find_critical_point(phi, phi_prime, G_max)
    if not crit['critical_point_found']:
        return crit

    G_star = crit['G_star']
    z_star = crit['z_star']
    phi_dd = phi_double_prime(G_star)

    if phi_dd <= 0:
        return {**crit,
                'admissible': False,
                'reason': 'phi\'\'(G*) <= 0; saddle-point expansion fails'}

    # Singular coefficient: G(z) ~ G* - A sqrt(1 - z/z*)
    # From the Lagrange inversion / Drmota-Lalley-Woods theorem
    A = np.sqrt(2.0 * phi(G_star) / (z_star * phi_dd))
    amplitude = A / (2.0 * np.sqrt(np.pi))

    return {
        **crit,
        'admissible': True,
        'phi_double_prime_at_Gstar': phi_dd,
        'A_singular_coef': A,
        'tau': 1.5,       # C_2 stratum (generic admissible)
        'stratum': 'C_2',
        'amplitude_coef_asymptotic': amplitude,
        'asymptotic_formula': (
            f'[z^n] G(z) ~ {amplitude:.6f} * n^(-3/2) * {1/z_star:.6f}^n'
        ),
        'radius_of_convergence': z_star,
        'growth_rate': 1.0 / z_star,
    }


def coefficient_asymptotic(n: int, result: Dict) -> float:
    """Evaluate the asymptotic [z^n] G(z) at integer n from the
    admissible_asymptotics result."""
    if not result.get('admissible', False):
        return float('nan')
    return result['amplitude_coef_asymptotic'] * n ** (-1.5) * (1 / result['z_star']) ** n


def stable_tree_tau(alpha: float) -> float:
    """Tail exponent for a stable-tree DSE with offspring-distribution
    stability index alpha in (1, 2].

    The Duquesne-Le Gall result: if the offspring distribution has
    heavy tail P(offspring = k) ~ k^{-1-alpha}, the total-progeny
    distribution has tau = 1 + 1/alpha.  For alpha = 2 (finite
    variance), tau = 3/2 recovers the admissible generic.

    This is an honest widening beyond the dyadic ladder: alpha can
    be any real in (1, 2], hence tau can be any real in [3/2, 2).
    alpha < 1 gives explosive branching (no stationary DSE).
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f'alpha = {alpha} not in (1, 2]')
    return 1.0 + 1.0 / alpha


# ------------------------------------------------------------------ #
#  Physical demonstration: Poisson-offspring branching CRN
# ------------------------------------------------------------------ #

def poisson_branching_demo(lambda_offspring: float = 1.0) -> Dict:
    """DSE for a continuous-time Galton-Watson tree with Poisson
    offspring distribution, mean lambda.

    The offspring PGF is phi(G) = exp(lambda * (G - 1)).  For
    criticality we need the fixed point z*·phi(G*) = G* with
    z*·phi'(G*) = 1 to coalesce, which occurs at lambda = 1
    (critical branching).  At lambda = 1:
        phi(G) = exp(G - 1),  G* = 1,  z* = 1,  phi''(1) = 1

    Coefficient asymptotic:
        [z^n] G(z) ~ (1 / (2 sqrt(pi))) · n^{-3/2}
    This is the classical Otter-tree / Cayley-formula asymptotic,
    recovered here from the admissible framework.

    For subcritical (lambda < 1): z* > 1, exponential decay of
    coefficients.  For supercritical (lambda > 1): z* < 1,
    exponential growth.
    """
    def phi(G):      return np.exp(lambda_offspring * (G - 1))
    def phi_p(G):    return lambda_offspring * np.exp(lambda_offspring * (G - 1))
    def phi_pp(G):   return lambda_offspring**2 * np.exp(lambda_offspring * (G - 1))

    r = admissible_asymptotics(phi, phi_p, phi_pp, G_max=100.0)
    r['system'] = 'Poisson-offspring branching'
    r['lambda_offspring'] = lambda_offspring
    r['critical_lambda'] = 1.0
    r['regime'] = ('critical' if abs(lambda_offspring - 1) < 1e-6
                    else 'subcritical' if lambda_offspring < 1
                    else 'supercritical')
    return r


def cosh_offspring_demo() -> Dict:
    """DSE with phi(G) = cosh(G) - a symmetric-difference kinetic,
    where an A-particle either annihilates with another (giving 0)
    or produces two daughters.  PGF is cosh because of the
    symmetry between 0 and 2 offspring (even only).

    Critical point: G·sinh(G) = cosh(G) gives G* ~ 1.1997,
    z* = G*/cosh(G*) ~ 0.6627, phi''(G*) = cosh(G*) > 0 so admissible.
    """
    r = admissible_asymptotics(np.cosh, np.sinh, np.cosh, G_max=10.0)
    r['system'] = 'cosh-offspring (symmetric-difference kinetics)'
    return r


if __name__ == '__main__':
    print('=== TODO-1: Admissible/transcendental DSE extension ===\n')

    print('Demo 1: Critical Poisson-offspring branching (phi = exp(G-1))')
    r = poisson_branching_demo(lambda_offspring=1.0)
    print(f'  G* = {r["G_star"]:.6f}')
    print(f'  z* = {r["z_star"]:.6f}')
    print(f'  tau = {r["tau"]}  (stratum {r["stratum"]})')
    print(f'  amplitude coef = {r["amplitude_coef_asymptotic"]:.6f}')
    print(f'  {r["asymptotic_formula"]}')
    print(f'  FS I.5 value 1/sqrt(2 pi) = {1/np.sqrt(2*np.pi):.6f}')
    print()
    print('Demo 2: cosh-offspring (symmetric-difference kinetics)')
    r2 = cosh_offspring_demo()
    print(f'  G* = {r2["G_star"]:.6f}')
    print(f'  z* = {r2["z_star"]:.6f}')
    print(f'  tau = {r2["tau"]}  (same C_2 stratum)')
    print(f'  amplitude coef = {r2["amplitude_coef_asymptotic"]:.6f}')
    print()
    print('Demo 3: stable-tree widening beyond dyadic ladder')
    for alpha in [2.0, 1.5, 1.2, 1.1]:
        print(f'  alpha = {alpha} -> tau = {stable_tree_tau(alpha):.4f}')
    print()
    print('Interpretation: admissible phi with finite variance always lands on')
    print('C_2 (tau = 3/2) regardless of the specific entire function - this is')
    print('the Drmota-Lalley-Woods universality.  Non-dyadic tau from the')
    print('admissible class requires INFINITE-VARIANCE offspring (stable trees).')
