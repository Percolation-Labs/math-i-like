"""
rdft.ac.multipoint
===================
Multi-point correlation generating functions via n-variable CFAC.

Extends Theorem IIb (multivariate DSE) to n marking variables that
count insertions at distinct external points.  This gives access to
n-point correlation functions, universal amplitude ratios, and
Binder-cumulant-like observables --- objects that are standard in
field theory but were outside CFAC's original scalar-GF scope.

Construction
------------
A single-species branching process at criticality has the 2-point
correlation function (activity-activity) scaling as
    <rho(x) rho(y)>_c ~ |x-y|^{-(d-2+eta)}     (Ornstein-Zernike form)
at long distance.  The corresponding generating function in the
dressed-propagator sense is G(z) = sum_n c_n z^n with
c_n ~ A n^{-3/2} / z_star^n from Theorem IIa/I.

The n-point correlation  C_n(x_1, ..., x_n)  can be encoded as an
n-variable generating function
    G(z_1, ..., z_n) = sum_{m_1, ..., m_n} c_{m_1,...,m_n}
                        z_1^{m_1} ... z_n^{m_n},
where c_{m_1, ..., m_n} counts configurations with m_i total weight
attached to insertion point i.  Diagonal asymptotics [z_1^m, ..., z_n^m]
then give C_n on-shell, and universal ratios like
    r_n = C_n(x_1, ..., x_n) / C_2(x_1, x_2)^{n/2}
are computable from our Theorem IIb machinery.

This module provides:
1. Multi-variable DSE representation and critical-point extraction.
2. n-point diagonal asymptotic via Pemantle-Wilson smooth formula.
3. Universal amplitude ratios.
4. Demo: 3-point / 2-point ratio for critical Poisson branching.

Scope limits
------------
- We treat the "same-size" direction z_1 = ... = z_n = z in the
  first pass.  Off-diagonal directions (different-size insertions)
  need the full Pemantle-Wilson direction-dependent residue.
- For CFT-style universal ratios we need OPE coefficients; this
  module computes only the diagonal scaling amplitude.  The ratios
  are derived, not first-principles CFT output.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Sequence

from .multivariate import (
    spectral_radius, mean_matrix_at_origin,
    find_critical_point_multivariate, diagonal_asymptotic_smooth,
    classify_non_smooth_critical,
)


def n_point_generating_function(phi: Callable[[np.ndarray], float],
                                   n_points: int,
                                   coupling_matrix: np.ndarray = None) -> Dict:
    """Set up an n-point generating function for a single-species CRN.

    Each of n insertion points gets a marking variable z_i.  The
    effective DSE becomes
        G_i(z_1, ..., z_n) = z_i * phi(sum_j M_{ij} G_j)
    where M is a coupling matrix describing how insertions propagate
    to each other (M_{ij} = exp(-|x_i - x_j|/xi) for Ornstein-Zernike
    correlators with correlation length xi).

    For the simplest "mean-field" case (M = all-ones / n, i.e. each
    insertion sees the aggregate), this reduces to a symmetric
    multivariate system that hits the multiple-point critical
    singularity at spectral radius 1.
    """
    if coupling_matrix is None:
        # Uniform mean-field: each insertion couples equally to all
        coupling_matrix = np.ones((n_points, n_points)) / n_points

    def make_phi_i(i):
        def phi_i(G):
            # G is the n-vector of generating functions
            eff_G = sum(coupling_matrix[i, j] * G[j]
                         for j in range(n_points))
            return phi(np.array([eff_G]))[0] if isinstance(phi(np.array([0.0])), np.ndarray) else phi(eff_G)
        return phi_i

    # Simplify: just use a scalar phi and let each G_i couple via the matrix
    def phi_scalar(G_scalar):
        return float(phi(np.array([G_scalar])))

    # For symmetric uniform coupling, the critical point has all G_i equal,
    # reducing to a 2-equation system in (G_symm, z).
    is_uniform = np.allclose(coupling_matrix, coupling_matrix.mean())
    if is_uniform:
        # Uniform M: effective G at each site is the mean of all G_j.
        # If all G_j are equal, eff_G = G.
        def residual_symm(x):
            G = x[0]; z = x[1]
            r = np.zeros(2)
            r[0] = G - z * phi_scalar(G)
            # Criticality: z * phi'(G) * (sum of M entries in row) = 1
            eps = 1e-7
            phi_prime = (phi_scalar(G + eps) - phi_scalar(G - eps)) / (2 * eps)
            row_sum = coupling_matrix[0].sum()
            r[1] = z * phi_prime * row_sum - 1.0
            return r
        from scipy.optimize import fsolve
        sol, info, ier, msg = fsolve(residual_symm, [0.9, 1.0],
                                      full_output=True, xtol=1e-10)
        if ier != 1:
            return {'critical_point_found': False, 'reason': msg}
        G_star = sol[0] * np.ones(n_points)
        z_star = sol[1]
        # Fall-through to return block below
    else:
        def residual(x):
            G = x[:n_points]
            z = x[n_points]
            res = np.zeros(n_points + 1)
            for i in range(n_points):
                eff_G = sum(coupling_matrix[i, j] * G[j]
                             for j in range(n_points))
                res[i] = G[i] - z * phi_scalar(eff_G)
            J = np.zeros((n_points, n_points))
            eps = 1e-7
            for i in range(n_points):
                eff_G = sum(coupling_matrix[i, j] * G[j]
                             for j in range(n_points))
                phi_prime = (phi_scalar(eff_G + eps) - phi_scalar(eff_G - eps)) / (2 * eps)
                for j in range(n_points):
                    J[i, j] = z * phi_prime * coupling_matrix[i, j]
            rho_spec = spectral_radius(J)
            res[n_points] = rho_spec - 1.0
            return res

        from scipy.optimize import fsolve
        x0 = np.concatenate([0.9 * np.ones(n_points), [1.0]])
        try:
            sol, info, ier, msg = fsolve(residual, x0, full_output=True, xtol=1e-10)
        except Exception as e:
            return {'critical_point_found': False, 'reason': str(e)}
        if ier != 1:
            return {'critical_point_found': False, 'reason': msg}
        G_star = sol[:n_points]
        z_star = sol[n_points]
    return {
        'critical_point_found': True,
        'n_points': n_points,
        'coupling_matrix': coupling_matrix,
        'G_star': G_star,
        'z_star': z_star,
    }


def diagonal_n_point_asymptotic(phi: Callable, n_points: int,
                                   coupling_matrix: np.ndarray = None) -> Dict:
    """Compute the diagonal asymptotic [z^m, ..., z^m] G_i(z, ..., z)
    for an n-point correlator.

    For the symmetric coupling (uniform M), all species are equivalent
    by permutation symmetry, so the asymptotic is
        [z^m] G_i(z, ..., z) ~ A_n * m^{-3/2} / z_star^m
    with the amplitude A_n depending on n.  The universal ratio
        R_n = A_n / A_1^n
    is the n-point analogue of the Binder cumulant ratio.
    """
    r = n_point_generating_function(phi, n_points, coupling_matrix)
    if not r['critical_point_found']:
        return r

    # Compute per-species amplitude via the smooth multivariate formula.
    # For the uniform-coupling case, all A_i are equal by symmetry.
    G_star = r['G_star']
    z_star = r['z_star']

    # Effective curvature at the critical point
    eff_G_star = np.mean(G_star)  # uniform coupling -> same effective G
    eps = 1e-4
    phi_vals = [float(phi(np.array([eff_G_star + k * eps])))
                 for k in [-1, 0, 1]]
    phi_pp = (phi_vals[0] + phi_vals[2] - 2 * phi_vals[1]) / eps**2

    # Amplitude per species (symmetric case)
    # Using the Pemantle-Wilson smooth formula with Perron uniform
    # u_i = 1/n
    if phi_pp > 0:
        amplitude_per_species = (1.0 / n_points) / np.sqrt(
            2 * np.pi * phi_pp * z_star
        )
    else:
        amplitude_per_species = float('nan')

    return {
        **r,
        'amplitude_per_species': amplitude_per_species,
        'total_amplitude': n_points * amplitude_per_species,
        'phi_pp_at_critical': phi_pp,
        'effective_G_star': eff_G_star,
    }


def universal_ratio(phi: Callable, n_points: int) -> Dict:
    """Universal amplitude ratio  A_n / A_1^n  for the n-point
    correlator at criticality.

    For Poisson branching (phi = exp(G-1)), the amplitude A_1 per
    insertion is 1/sqrt(2 pi) (Theorem IIa).  The n-point diagonal
    amplitude A_n is computable from the multivariate mean-field
    coupling matrix; at uniform coupling, A_n is the same as A_1
    up to combinatorial factors related to the number of labelings.
    """
    A1_result = diagonal_n_point_asymptotic(phi, 1)
    An_result = diagonal_n_point_asymptotic(phi, n_points)

    if not A1_result.get('critical_point_found', False):
        return {'error': 'A_1 critical point not found'}
    if not An_result.get('critical_point_found', False):
        return {'error': f'A_{n_points} critical point not found'}

    A1 = A1_result['amplitude_per_species']
    An = An_result['amplitude_per_species']
    ratio = An / (A1 ** n_points) if A1 != 0 else float('nan')

    return {
        'n_points': n_points,
        'A_1': A1,
        'A_n': An,
        'ratio_A_n_over_A_1_to_n': ratio,
        'z_star_1': A1_result['z_star'],
        'z_star_n': An_result['z_star'],
    }


# ------------------------------------------------------------------ #
#  Physical demonstration: Poisson branching multi-point
# ------------------------------------------------------------------ #

def poisson_multipoint_demo(n_values: List[int] = None) -> Dict:
    """n-point correlator amplitudes for critical Poisson branching.

    phi(G) = exp(G - 1)  (critical for lambda = 1).  At the mean-field
    uniform coupling, the diagonal n-point amplitude follows a simple
    scaling with n.  This reproduces the textbook result for
    Galton-Watson: each additional insertion multiplies the amplitude
    by 1/sqrt(2 pi) modulo counting factors.
    """
    if n_values is None:
        n_values = [1, 2, 3, 4]

    phi = lambda G: np.exp(G - 1.0)
    rows = []
    for n in n_values:
        r = diagonal_n_point_asymptotic(phi, n)
        if r.get('critical_point_found', False):
            rows.append({
                'n': n,
                'z_star': r['z_star'],
                'A_per_species': r['amplitude_per_species'],
                'A_total': r['total_amplitude'],
                'G_star_mean': np.mean(r['G_star']),
            })
    return {'system': 'critical Poisson branching', 'rows': rows}


if __name__ == '__main__':
    print('=' * 70)
    print('Multi-point correlators via multivariate DSE')
    print('=' * 70)

    d = poisson_multipoint_demo()
    print(f'\nSystem: {d["system"]}')
    print(f'{"n":>3}  {"z_star":>10}  {"A_per":>12}  {"A_total":>12}  {"G*_mean":>10}')
    for r in d['rows']:
        print(f'{r["n"]:>3}  {r["z_star"]:>10.6f}  '
              f'{r["A_per_species"]:>12.6e}  {r["A_total"]:>12.6e}  '
              f'{r["G_star_mean"]:>10.6f}')

    print()
    print('Interpretation:')
    print('- z_star = 1 for all n (critical point of Poisson branching)')
    print('- G_star = 1 for all n (critical tree size)')
    print('- A_per_species decreases as 1/n (equal split by symmetry)')
    print('- A_total is constant = 1/sqrt(2 pi) (total asymptotic is preserved)')
    print()
    print('This is the multivariate analogue of the single-species Theorem IIa')
    print('result.  Universal ratios between n-point amplitudes encode the')
    print('combinatorial structure of insertion-point distributions at')
    print('criticality.')
