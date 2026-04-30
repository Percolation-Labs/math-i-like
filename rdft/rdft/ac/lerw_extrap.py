"""
rdft.ac.lerw_extrap
===================
Tier: 3 (research)

Algebraic extrapolation schemes for d_f from finite-size LERW data.

After the tube-transfer-matrix animal failed (rdft/ac/lerw_tube.py:
all d_f^{(N)} collapse to 1 because narrow tubes are quasi-1D),
the correct "algebraic animal" for d_f^{(3)} is the sequence of
extrapolation schemes acting on isotropic-box Monte-Carlo data.
Each scheme is a rational transformation of a finite sequence
<|gamma|>(L_1), <|gamma|>(L_2), ..., <|gamma|>(L_m) to a single
estimate of d_f^{(3)}.

Four schemes:

1. **Naive log-log fit.** <|gamma|>(L) = A * L^{d_f}.
   (Baseline used in rdft/ac/lerw_dirichlet.py.)

2. **Leading 1/L correction.** <|gamma|>(L) = A * L^{d_f} (1 + B/L).
   Fits d_f, A, B from at least three L values.

3. **Effective-exponent Richardson extrapolation.** Define
       d_f_eff(L) := log(<|gamma|>(L+dL) / <|gamma|>(L-dL))
                       / log((L+dL) / (L-dL)).
   Each d_f_eff(L) is a discrete approximation to d_f with
   leading-order finite-size correction. Richardson extrapolation
   eliminates the 1/L correction, giving a sequence that
   converges faster.

4. **Neville-Richardson tableau.** Assumes corrections at known
   powers (1/L, 1/L^2, ...) and triangulates them out.

These are classical numerical-analysis moves but they are also
the AC extraction toolkit: each scheme is a polynomial / rational
operation on the input sample, hence its output is algebraic in
the inputs. Running several and seeing agreement (or not) is the
convergence witness.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


# ------------------------------------------------------------------ #
#  1. Naive log-log fit
# ------------------------------------------------------------------ #

def naive_fit(L_vals: Sequence[int], means: Sequence[float]
              ) -> tuple[float, float]:
    """Least-squares slope of log(mean) vs log(L)."""
    x = np.log(np.asarray(L_vals, dtype=float))
    y = np.log(np.asarray(means, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


# ------------------------------------------------------------------ #
#  2. Nonlinear fit with 1/L correction
# ------------------------------------------------------------------ #

def fit_with_correction(L_vals: Sequence[int], means: Sequence[float],
                        omega: float = 1.0,
                        n_iter: int = 100) -> dict:
    """Fit <|gamma|>(L) = A L^{d_f} (1 + B / L^omega) by alternating
    minimisation. omega is a fixed correction-to-scaling exponent
    (typically 1 for analytic corrections, smaller for universal
    corrections).

    Returns {'d_f': ..., 'A': ..., 'B': ..., 'residual': ...}.
    """
    L = np.asarray(L_vals, dtype=float)
    y = np.asarray(means, dtype=float)
    d_f, log_A = naive_fit(L_vals, means)
    B = 0.0
    A = float(np.exp(log_A))
    for _ in range(n_iter):
        # Fix (A, d_f), update B in least-squares sense from
        #   y = A L^{d_f} (1 + B L^{-omega})
        #   => (y / (A L^{d_f}) - 1) = B L^{-omega}
        u = L ** (-omega)
        r = y / (A * L ** d_f) - 1.0
        denom = float(np.sum(u * u))
        if denom == 0:
            break
        B = float(np.sum(u * r) / denom)
        # Guard: prevent (1 + B/L^omega) from going non-positive
        corr = 1.0 + B * u
        if np.any(corr <= 0):
            # Shrink B toward zero
            B *= 0.5
            corr = 1.0 + B * u
        y_corr = y / corr
        d_f_new, log_A_new = naive_fit(L_vals, y_corr)
        if abs(d_f_new - d_f) < 1e-10:
            d_f, log_A = d_f_new, log_A_new
            A = float(np.exp(log_A))
            break
        d_f, log_A = d_f_new, log_A_new
        A = float(np.exp(log_A))
    residual = float(np.sum(
        (y - A * L ** d_f * (1.0 + B * L ** (-omega))) ** 2))
    return {'d_f': float(d_f), 'A': A, 'B': B,
            'omega': float(omega), 'residual': residual}


def scan_omega(L_vals: Sequence[int], means: Sequence[float],
               omega_grid: Sequence[float] | None = None) -> dict:
    """Grid-search for the correction-to-scaling exponent omega.

    For each omega in omega_grid, fit d_f and report the one with
    smallest residual. This lets omega be determined from the data
    instead of imposed by hand.
    """
    if omega_grid is None:
        omega_grid = np.linspace(0.3, 2.0, 35)
    best = None
    for omega in omega_grid:
        res = fit_with_correction(L_vals, means, omega=float(omega))
        if best is None or res['residual'] < best['residual']:
            best = res
    return best


# ------------------------------------------------------------------ #
#  3. Effective-exponent Richardson extrapolation
# ------------------------------------------------------------------ #

def effective_exponents(L_vals: Sequence[int], means: Sequence[float]
                        ) -> list[tuple[float, float]]:
    """Compute d_f_eff at consecutive-pair midpoints.

    d_f_eff_i := log(mean[i+1]/mean[i]) / log(L[i+1]/L[i]).
    Returns list of (L_mid, d_f_eff).
    """
    out = []
    for i in range(len(L_vals) - 1):
        L1, L2 = float(L_vals[i]), float(L_vals[i + 1])
        m1, m2 = float(means[i]), float(means[i + 1])
        d = np.log(m2 / m1) / np.log(L2 / L1)
        out.append((0.5 * (L1 + L2), float(d)))
    return out


def richardson_extrapolate(L_mids: Sequence[float],
                           d_effs: Sequence[float],
                           ) -> float:
    """Richardson extrapolate d_f_eff(L) = d_f + C / L + O(1/L^2)
    using two consecutive points:
      d_f ~ (L2 * d_eff2 - L1 * d_eff1) / (L2 - L1).

    Returns the extrapolated estimate from the largest two L_mids.
    """
    L1, L2 = float(L_mids[-2]), float(L_mids[-1])
    d1, d2 = float(d_effs[-2]), float(d_effs[-1])
    return float((L2 * d2 - L1 * d1) / (L2 - L1))


# ------------------------------------------------------------------ #
#  4. Neville-Richardson tableau (assumed 1/L, 1/L^2, ... tower)
# ------------------------------------------------------------------ #

def neville_richardson(L_mids: Sequence[float],
                       d_effs: Sequence[float],
                       max_depth: int | None = None,
                       ) -> list[list[float]]:
    """Build the Neville-Richardson tableau assuming corrections
    at integer powers 1/L, 1/L^2, ....

    Returns the triangular array T[depth][index] where T[0] = d_effs
    and T[depth+1][i] = (L[i+depth+1] * T[depth][i+1] - L[i] * T[depth][i])
                        / (L[i+depth+1] - L[i]).

    The diagonal T[depth][-1] at max depth gives the best estimate.
    """
    n = len(d_effs)
    if max_depth is None:
        max_depth = n - 1
    max_depth = min(max_depth, n - 1)
    T: list[list[float]] = [list(map(float, d_effs))]
    for depth in range(max_depth):
        prev = T[-1]
        new_row: list[float] = []
        for i in range(len(prev) - 1):
            L1 = float(L_mids[i])
            L2 = float(L_mids[i + depth + 1])
            t = (L2 * prev[i + 1] - L1 * prev[i]) / (L2 - L1)
            new_row.append(float(t))
        T.append(new_row)
    return T


def best_neville(L_mids: Sequence[float], d_effs: Sequence[float]) -> float:
    """Top-right entry of the Neville-Richardson tableau: the
    deepest extrapolant using all available data.
    """
    T = neville_richardson(L_mids, d_effs)
    return T[-1][-1]


# ------------------------------------------------------------------ #
#  High-level API: apply all schemes to one dataset
# ------------------------------------------------------------------ #

def all_extrapolations(L_vals: Sequence[int], means: Sequence[float],
                       sems: Sequence[float] | None = None,
                       ) -> dict:
    """Run all four schemes on the given (L, <|gamma|>) data.

    Returns a dict with each scheme's estimate, suitable for
    scheme-to-scheme agreement checks.
    """
    naive, _ = naive_fit(L_vals, means)
    with_corr = fit_with_correction(L_vals, means, omega=1.0)
    best_omega = scan_omega(L_vals, means)
    eff = effective_exponents(L_vals, means)
    L_mids = [lm for lm, _ in eff]
    d_effs = [de for _, de in eff]
    rich = richardson_extrapolate(L_mids, d_effs) \
        if len(L_mids) >= 2 else float('nan')
    nev = best_neville(L_mids, d_effs) if len(L_mids) >= 2 else float('nan')
    return {
        'naive': naive,
        'with_correction_om1': with_corr['d_f'],
        'best_omega_fit': {
            'd_f': best_omega['d_f'],
            'omega': best_omega['omega'],
            'B': best_omega['B'],
        },
        'richardson': rich,
        'neville': nev,
        'effective_exponents': eff,
    }
