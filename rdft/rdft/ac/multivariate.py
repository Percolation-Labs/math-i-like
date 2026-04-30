"""
rdft.ac.multivariate
=====================
Tier: 2 (extension)

Multivariate ACSV for coupled-DSE systems: the Pemantle-Wilson
framework applied to CFAC.  Addresses TODO-3 from docs/problems.md.

Context
-------
The CFAC stratification theorem treats a single-species DSE
G = z phi(G).  Real CFAC problems (coupled particle-field: KS
chemotaxis, ant colonies, multi-species CRNs, age-structured
epidemic models) are natively multivariate:

    G_i = z_i · phi_i(G_1, ..., G_n),   i = 1, ..., n

One can always reduce to univariate by computing the resultant and
tracking the dominant branch, but that projection LOSES structure:
the amplitudes, the mean-matrix eigenvalues at criticality, and
the direction-dependent scaling functions are all lost.

Pemantle-Wilson *Analytic Combinatorics in Several Variables* (CUP
2013) gives the full theory: diagonal / directional asymptotics of
multivariate GFs are controlled by critical points of the singular
variety V = {F = 0} when F is the denominator of the rational GF,
or more generally by contact points of the log-direction ray with V.

What this module provides
-------------------------
1. Critical-point computation for a coupled n-species DSE via the
   spectral-radius criterion for the offspring mean matrix M.
2. Singular-variety classification: smooth / multiple / cone, per
   Pemantle-Wilson (the three-way classification).
3. Diagonal-coefficient asymptotics for smooth-point cases:
     [z_1^m ... z_n^m] G_i ~ C_i · rho^{-nm} · (nm)^{-d/2}
   with d = 1 for smooth boundary of V at the critical point.
4. Physical demonstration: two-type branching process (e.g.
   S/R ants with state transitions) as a multivariate CRN.

Scope
-----
- Handles smooth-point case explicitly (the generic one).
- Flags multiple-point and cone-point for future extension.
- Does NOT yet implement the full Morse-theoretic analysis of V
  for non-smooth singularities.
- Works in symbolic form via sympy when phi is algebraic.
- Numerical fall-back when phi is transcendental.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence
from scipy.optimize import fsolve


def mean_matrix_at_origin(phi_funcs: Sequence[Callable],
                           G0: Optional[np.ndarray] = None,
                           eps: float = 1e-6) -> np.ndarray:
    """The n x n offspring mean matrix M_ij = d phi_i / d G_j at G=0.

    For branching processes this is the matrix whose spectral radius
    governs subcriticality / criticality / supercriticality.
    """
    n = len(phi_funcs)
    if G0 is None:
        G0 = np.zeros(n)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Gp = G0.copy(); Gp[j] += eps
            Gm = G0.copy(); Gm[j] -= eps
            M[i, j] = (phi_funcs[i](Gp) - phi_funcs[i](Gm)) / (2 * eps)
    return M


def spectral_radius(M: np.ndarray) -> float:
    """Perron-Frobenius / spectral radius of M."""
    eigs = np.linalg.eigvals(M)
    return max(abs(eigs))


def find_critical_point_multivariate(
        phi_funcs: Sequence[Callable],
        n: int,
        z_init: Optional[np.ndarray] = None,
        G_init: Optional[np.ndarray] = None) -> Dict:
    """Find (z*, G*) for the coupled DSE G_i = z_i phi_i(G).

    Critical condition: the Jacobian J_ij = z_i d phi_i / d G_j has
    spectral radius 1 at (z*, G*).  We restrict to the DIAGONAL
    critical point z_i = rho for all i — the natural generalisation
    of the univariate z* for size-distribution asymptotics.

    System of 2n equations:
      G_i = rho * phi_i(G)         (on-shell, n equations)
      spectral radius of rho * J(G) = 1   (criticality, 1 equation)
    Plus normalisation: sum G_i = const (or fix a reference).

    For the diagonal case we have n+1 unknowns (G_1,...,G_n,rho) and
    n+1 equations (n on-shell + 1 criticality).
    """
    if z_init is None:
        z_init = np.ones(n)
    if G_init is None:
        G_init = 0.5 * np.ones(n)

    def residual(x):
        G = x[:n]
        rho = x[n]
        # On-shell: G_i - rho phi_i(G) = 0
        res = np.zeros(n + 1)
        for i in range(n):
            res[i] = G[i] - rho * phi_funcs[i](G)
        # Criticality: spectral radius of rho J(G) = 1
        J = np.zeros((n, n))
        eps = 1e-7
        for i in range(n):
            for j in range(n):
                Gp = G.copy(); Gp[j] += eps
                Gm = G.copy(); Gm[j] -= eps
                J[i, j] = (phi_funcs[i](Gp) - phi_funcs[i](Gm)) / (2 * eps)
        rho_spec = spectral_radius(rho * J)
        res[n] = rho_spec - 1.0
        return res

    x0 = np.concatenate([G_init, [1.0]])
    try:
        sol, infodict, ier, msg = fsolve(residual, x0, full_output=True,
                                          xtol=1e-10)
    except Exception as e:
        return {'critical_point_found': False, 'reason': str(e)}

    if ier != 1:
        return {'critical_point_found': False, 'reason': msg}

    G_star = sol[:n]
    rho_star = sol[n]
    M = mean_matrix_at_origin(phi_funcs, G_star)
    J_star = rho_star * M
    eigs = np.linalg.eigvals(J_star)

    return {
        'critical_point_found': True,
        'G_star': G_star,
        'rho_star': rho_star,
        'mean_matrix_at_Gstar': M,
        'Jacobian_eigenvalues': eigs,
        'spectral_radius': spectral_radius(J_star),
    }


def classify_singular_point(phi_funcs: Sequence[Callable],
                              G_star: np.ndarray,
                              rho_star: float,
                              eps_num: float = 1e-6) -> Dict:
    """Classify the critical point as smooth / multiple / cone
    (Pemantle-Wilson).

    Smooth point: V = {F = 0} has rank-1 gradient at (z*, G*).  In
    this case the diagonal asymptotic is the generic d=1 form.

    Multiple point: two or more sheets of V cross transversally.
    Additional n^{-(d-1)/2} factor where d = number of sheets.

    Cone point: non-transverse intersection (tangential cone).
    Most exotic; power law determined by cone geometry.

    Detection: we look at the Jacobian eigenvalues of the coupled
    system at the critical point.
      - All eigenvalues real, one equals 1 exactly: smooth
      - Two eigenvalues at 1 (degenerate): multiple
      - Complex-conjugate pair at magnitude 1: cone candidate
    """
    n = len(phi_funcs)
    M = mean_matrix_at_origin(phi_funcs, G_star)
    J_star = rho_star * M
    eigs = np.linalg.eigvals(J_star)

    # Distance of each eigenvalue to 1
    eig_near_1 = [e for e in eigs if abs(e - 1) < 1e-5]
    n_near_1 = len(eig_near_1)

    if n_near_1 == 1:
        classification = 'smooth'
        explanation = 'single dominant eigenvalue = 1 at critical point'
    elif n_near_1 == 2:
        # Check if they are real (distinct -> multiple point) or complex conjugate
        if all(abs(e.imag) < 1e-6 for e in eig_near_1):
            classification = 'multiple'
            explanation = 'two real eigenvalues at 1: transverse crossing'
        else:
            classification = 'cone_candidate'
            explanation = 'complex pair at magnitude 1: possible cone geometry'
    else:
        classification = f'degenerate_order_{n_near_1}'
        explanation = f'{n_near_1} eigenvalues at magnitude 1'

    return {
        'eigenvalues_at_critical': eigs,
        'num_eigenvalues_at_one': n_near_1,
        'classification': classification,
        'explanation': explanation,
    }


def classify_non_smooth_critical(phi_funcs: Sequence[Callable],
                                    G_star: np.ndarray,
                                    rho_star: float,
                                    tol: float = 1e-4,
                                    verbose: bool = False) -> Dict:
    """Classify a critical point as smooth / multiple / cone via
    the Jacobian spectrum and eigenvector geometry.

    Pemantle-Wilson terminology (ACSV, Pemantle-Wilson 2013 Ch. 9):
      - SMOOTH point: single dominant eigenvalue of rho_star·J(G_star)
        equal to 1 (multiplicity 1).  Standard tau = 3/2 with the
        Perron-eigenvector-weighted amplitude of
        diagonal_asymptotic_smooth().
      - MULTIPLE point: r >= 2 sheets of the singular variety V cross
        TRANSVERSALLY at (z_star, G_star).  The Jacobian has r
        linearly-independent eigenvectors at eigenvalue 1.
        Pemantle-Wilson give an asymptotic as a sum over sheets, with
        exponent structure depending on the (r, n) geometry.
      - CONE point: r >= 2 sheets meet NON-transversally (their
        tangent planes are coplanar or close).  The asymptotic is
        non-universal and depends on the cone geometry.

    This function DETECTS the case and returns a structured
    classification; it does NOT always produce an asymptotic, because
    the Pemantle-Wilson multi-sheet residue integral is problem-specific.
    When the smooth-amplitude formula is applicable per sheet, the sum
    is returned; otherwise the user is directed to carry out the
    residue computation by hand.
    """
    n = len(phi_funcs)
    M = mean_matrix_at_origin(phi_funcs, G_star)
    J_star = rho_star * M
    eigs, V = np.linalg.eig(J_star)

    # Indices of eigenvalues near 1
    tol = 1e-4
    unit_idx = [i for i, e in enumerate(eigs) if abs(e - 1.0) < tol]
    r = len(unit_idx)

    if r == 0:
        return {'error': 'no eigenvalue near 1 — not at a critical point'}

    # Collect right eigenvectors for the unit eigenspace
    u_vectors = []
    for i in unit_idx:
        u = np.real(V[:, i])
        # Normalise in L^1
        u = u / (np.sum(np.abs(u)) + 1e-20)
        u_vectors.append(u)

    # Left eigenvectors (transpose)
    eigs_L, W = np.linalg.eig(J_star.T)
    unit_idx_L = [i for i, e in enumerate(eigs_L) if abs(e - 1.0) < tol]
    v_vectors = []
    for i in unit_idx_L:
        v = np.real(W[:, i])
        v = v / (np.sum(np.abs(v)) + 1e-20)
        v_vectors.append(v)

    if verbose:
        print(f'Multiplicity r = {r}; u vectors = {u_vectors}; '
              f'v vectors = {v_vectors}')

    # Check transversality: the right eigenvectors should be linearly
    # independent.  If they are (nearly) parallel, we are at a CONE
    # point rather than a multiple point.
    if r >= 2:
        # Gram matrix
        U = np.array(u_vectors)  # r x n
        G_gram = U @ U.T
        det_gram = np.linalg.det(G_gram)
        is_transverse = abs(det_gram) > 1e-6
    else:
        is_transverse = True

    # Curvature: sum over each sheet's contribution
    eps = 1e-4
    phi_pp_per_sheet = []
    for u_right, v_left in zip(u_vectors, v_vectors):
        Gp = G_star + eps * u_right
        Gm = G_star - eps * u_right
        phi_pp_vec = np.array([
            (phi_funcs[i](Gp) + phi_funcs[i](Gm) - 2 * phi_funcs[i](G_star)) / eps**2
            for i in range(n)
        ])
        curvature = np.dot(v_left, phi_pp_vec)
        phi_pp_per_sheet.append(curvature)

    if r == 1:
        classification = 'smooth'
        advice = 'use diagonal_asymptotic_smooth()'
    elif r >= 2 and is_transverse:
        classification = 'multiple_point'
        advice = (
            f'{r} sheets cross transversally; Pemantle-Wilson '
            'residue integral per sheet needed for full asymptotic. '
            'Per-sheet smooth amplitude contributions listed in '
            'amplitude_per_sheet.'
        )
    else:
        classification = 'cone_point'
        advice = (
            f'{r} sheets meet non-transversally (det Gram of '
            f'right-eigenvectors = {det_gram:.3e}); asymptotic is '
            'non-universal and depends on cone geometry.  Case-by-case '
            'contour integration required.'
        )

    # Per-sheet smooth amplitude (first approximation for multiple-point)
    amplitudes_per_sheet = []
    for u_r, curv in zip(u_vectors, phi_pp_per_sheet):
        if curv > 0:
            amp = np.abs(u_r) / np.sqrt(2 * np.pi * curv * rho_star)
            amplitudes_per_sheet.append(amp)
        else:
            amplitudes_per_sheet.append(None)

    return {
        'classification': classification,
        'multiplicity_r': r,
        'is_transverse': is_transverse,
        'rho_star': rho_star,
        'G_star': G_star,
        'right_eigenvectors': u_vectors,
        'left_eigenvectors': v_vectors,
        'curvatures_per_sheet': phi_pp_per_sheet,
        'amplitude_per_sheet': amplitudes_per_sheet,
        'advice': advice,
    }


def diagonal_asymptotic_smooth(phi_funcs: Sequence[Callable],
                                 G_star: np.ndarray,
                                 rho_star: float,
                                 species_index: int = 0) -> Dict:
    """Asymptotic coefficient [z^m] G_i(z, z, ..., z) for smooth
    critical point.

    For smooth V, the Pemantle-Wilson result is
      [z^m] G_i(z, ..., z) ~ C_i · rho^{-m} · m^{-3/2}
    where rho = rho_star and C_i depends on the left/right
    eigenvectors of the Jacobian and phi_i''(G*).

    The exponent 3/2 is the C_2 stratum for any smooth multivariate
    branching.  The AMPLITUDE C_i is what encodes the multi-species
    structure.
    """
    n = len(phi_funcs)
    M = mean_matrix_at_origin(phi_funcs, G_star)
    J_star = rho_star * M

    # Left/right eigenvectors for the unit eigenvalue
    eigs, V = np.linalg.eig(J_star)
    idx = np.argmin(np.abs(eigs - 1.0))
    u_right = np.real(V[:, idx])   # right eigenvector
    u_right = u_right / np.sum(np.abs(u_right))  # normalise

    # Left eigenvector via transpose
    eigs_L, W = np.linalg.eig(J_star.T)
    idx_L = np.argmin(np.abs(eigs_L - 1.0))
    v_left = np.real(W[:, idx_L])
    v_left = v_left / np.sum(np.abs(v_left))

    # Curvature of phi at critical point (2nd derivatives in the
    # right-eigenvector direction)
    eps = 1e-4
    Gp = G_star + eps * u_right
    Gm = G_star - eps * u_right
    phi_star = np.array([phi_funcs[i](G_star) for i in range(n)])
    phi_pp = np.array([
        (phi_funcs[i](Gp) + phi_funcs[i](Gm) - 2 * phi_funcs[i](G_star)) / eps**2
        for i in range(n)
    ])
    curvature = np.dot(v_left, phi_pp)

    # Amplitude (simplified formula for smooth critical point;
    # full PW formula includes the Hessian determinant on V)
    amplitude = 1.0 / np.sqrt(2 * np.pi * abs(curvature) * rho_star) \
                 * abs(u_right[species_index]) * abs(v_left[species_index])

    return {
        'tau': 1.5,
        'stratum': 'C_2',
        'rho_star': rho_star,
        'G_star': G_star,
        'right_eigenvector': u_right,
        'left_eigenvector': v_left,
        'curvature_at_critical': curvature,
        'amplitude_species_{}'.format(species_index): amplitude,
        'asymptotic_formula': (
            f'[z^m] G_{species_index}(z,...,z) ~ {amplitude:.6f} '
            f'* m^(-3/2) * {1/rho_star:.6f}^m'
        ),
    }


# ------------------------------------------------------------------ #
#  Physical demonstration: independent two-species (multiple-point)
# ------------------------------------------------------------------ #

def cone_point_demo(epsilon: float = 0.01) -> Dict:
    """Construct a CONE-POINT critical singularity.

    We take a 2x2 Jacobian rho*J(G_star) = [[1, 0], [eps, 1]] —
    a Jordan block in the limit eps -> 0, or equivalently, two
    sheets whose tangent directions both align with (1, 0) to
    leading order when eps is small.

    The offspring functions phi_1, phi_2 are constructed to give
    this Jacobian at the chosen G_star.  For small eps the two
    right eigenvectors are (1, 0) and (1, -eps) — nearly parallel.
    The Gram determinant ~ eps^2 is below our transversality
    threshold, so the classification flags 'cone_point'.

    This is a contrived example — physical CRNs rarely sit exactly
    at a cone point — but it illustrates the detection logic.
    """
    def phi1(G):
        return G[0] + 0.5 * G[0] ** 2  # gives dphi_1/dG_1 = 1 + G_0 at G_star
    def phi2(G):
        return epsilon * G[0] + G[1] + 0.5 * G[1] ** 2

    # Pick G_star such that phi_i(G_star) = G_star (on-shell for rho=1)
    # and Jacobian has eigenvalue 1 with the Jordan-like structure.
    # Use G_star = (0, 0) as a trivial basepoint; the linear kernel gives
    # J = [[1, 0], [eps, 1]].
    G_star = np.array([0.0, 0.0])
    rho_star = 1.0

    cls = classify_non_smooth_critical([phi1, phi2], G_star, rho_star)
    return {
        'system': f'cone-point demo with eps = {epsilon}',
        'G_star': G_star.tolist(),
        'rho_star': rho_star,
        **cls,
    }


def independent_two_species_demo() -> Dict:
    """Two species branching INDEPENDENTLY, each critical Poisson.

    phi_1(G) = exp(G_1 - 1),  phi_2(G) = exp(G_2 - 1).
    At G_star = (1, 1), the Jacobian rho_star * J = I has eigenvalue
    1 with algebraic and geometric multiplicity 2 — a multiple point
    where two independent sheets of V cross transversally.

    The right eigenvectors (1, 0) and (0, 1) are orthogonal, so the
    crossing is maximally transverse — NOT a cone point.

    Physical interpretation: two non-interacting branching processes
    at simultaneous criticality.  Each sheet of V corresponds to one
    of the species being critical.  The diagonal coefficient
    [z^m] G_i(z, z) factors as a convolution of the two independent
    species' generating functions, giving the same m^(-3/2) tail as
    the single-species case (no log factor in this direction because
    the non-interaction makes the two sheets independent after
    diagonal projection).

    This demonstrates that the multiple-point CLASSIFICATION is
    correctly identified even when the physical consequences are
    simple.
    """
    def phi1(G):  return np.exp(G[0] - 1.0)
    def phi2(G):  return np.exp(G[1] - 1.0)

    # Manually set the critical point (since fsolve might not find a
    # unique one when there's a continuous family of critical points
    # for independent species)
    G_star = np.array([1.0, 1.0])
    rho_star = 1.0

    cls = classify_non_smooth_critical([phi1, phi2], G_star, rho_star)
    return {
        'system': 'two independent critical Poisson branching',
        'G_star': G_star.tolist(),
        'rho_star': rho_star,
        **cls,
    }


# ------------------------------------------------------------------ #
#  Physical demonstration: 2-type branching (S/R ant states)
# ------------------------------------------------------------------ #

def two_type_branching_demo(m11: float = 0.5, m12: float = 0.5,
                              m21: float = 0.5, m22: float = 0.5) -> Dict:
    """Two-type branching with mean matrix M = [[m11, m12], [m21, m22]].

    Offspring PGFs: phi_i(G) = exp(sum_j m_ij (G_j - 1)) — Poisson in
    each outgoing channel with the given mean.

    Critical when spectral radius of M equals 1.  For the symmetric
    case m_ij = 1/2 (total mean 1, half to each type), we're at
    criticality with diagonal asymptotic.
    """
    M = np.array([[m11, m12], [m21, m22]])

    def phi1(G):  return np.exp(m11 * (G[0] - 1) + m12 * (G[1] - 1))
    def phi2(G):  return np.exp(m21 * (G[0] - 1) + m22 * (G[1] - 1))

    crit = find_critical_point_multivariate([phi1, phi2], n=2,
                                              z_init=np.ones(2),
                                              G_init=np.array([0.9, 0.9]))
    if not crit['critical_point_found']:
        return crit

    cls = classify_singular_point([phi1, phi2], crit['G_star'],
                                    crit['rho_star'])

    if cls['classification'] == 'smooth':
        asymp = diagonal_asymptotic_smooth([phi1, phi2], crit['G_star'],
                                             crit['rho_star'], species_index=0)
    else:
        asymp = {'note': f'non-smooth case ({cls["classification"]}) '
                          'not yet implemented'}

    return {
        'mean_matrix': M,
        'spectral_radius_M': spectral_radius(M),
        **crit,
        **cls,
        **asymp,
    }


if __name__ == '__main__':
    print('=== TODO-3: Multivariate ACSV for coupled DSE ===\n')

    print('Demo: 2-type branching with symmetric mean matrix')
    print('(e.g., ant in state S/R with 50/50 transition, total mean 1)\n')
    r = two_type_branching_demo(m11=0.5, m12=0.5, m21=0.5, m22=0.5)
    print(f'  Mean matrix M = {r["mean_matrix"].tolist()}')
    print(f'  Spectral radius of M = {r["spectral_radius_M"]:.6f}')
    print(f'  Critical point G* = {r["G_star"]}')
    print(f'  rho* = {r["rho_star"]:.6f}')
    print(f'  Singular-point classification: {r["classification"]}')
    print(f'  tau = {r.get("tau", "N/A")} (stratum {r.get("stratum", "N/A")})')
    print(f'  {r.get("asymptotic_formula", "N/A")}')
    print()
    print('Interpretation: two-type branching with smooth critical point')
    print('gives the same C_2 stratum (tau = 3/2) as single-species, but the')
    print('AMPLITUDE is now a function of the mean-matrix eigenvectors - new')
    print('structure that the univariate resultant projection would lose.')
    print()

    print('Demo 2: Asymmetric mean matrix (subcritical total, species')
    print('asymmetry, should still give smooth diagonal if critical)')
    r2 = two_type_branching_demo(m11=0.3, m12=0.5, m21=0.6, m22=0.2)
    print(f'  Mean matrix M = {r2["mean_matrix"].tolist()}')
    print(f'  Spectral radius of M = {r2["spectral_radius_M"]:.6f}')
    print(f'  rho* = {r2["rho_star"]:.6f}')
    print(f'  Classification: {r2["classification"]}')
