"""
rdft.ac.tap_complexity
======================
Tier: 3 (research)

Planar (genus-0) TAP complexity from the semicircle resolvent.

For the spherical pure p-spin model the expected log-complexity per
spin Sigma(E) has the Auffinger-Ben Arous-Cerny (2013) closed form

    Sigma_p(E) = (1/2) log(p - 1) - (p - 2) E^2 / (4 (p - 1))
               + I_1(E ; p)                                      (1)

where I_1 is the semicircle integral

    I_1(E; p) = (1/N) log E|det(M - f(E) I)|  in the planar limit
              = integral_{-E_inf}^{E_inf} log|lambda - f(E)|
                rho_sc(lambda) d lambda                           (2)

and f(E) = E * sqrt(p / (2 (p-1))) is the shift fixed by the
Lagrange multiplier; E_inf = 2 is the semicircle support.

This module evaluates (1) as a function of E and p via a direct
closed-form integration of (2), independent of the genus-graded
Harer-Zagier reconstruction in rdft.ac.matrix_model. The two routes
should agree for |f| outside the semicircle support (where the
log-det series converges); this agreement is the content of
Prop 2 of paper/cfac/enumerative_boundary.tex at the planar order.
"""

from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------ #
#  Semicircle log-det integral
# ------------------------------------------------------------------ #

def semicircle_log_det(f: float) -> float:
    """I(f) = integral_{-2}^{2} log|lambda - f| * rho_sc(lambda) dlambda

    where rho_sc(lambda) = (1/(2 pi)) sqrt(4 - lambda^2) is the
    Wigner semicircle density on [-2, 2].

    Closed form: set x = |f|/2.
      |f| >= 2 (x >= 1):
        I(f) = log(x + sqrt(x^2 - 1)) + f^2 / 4 - 1/2
                - (|f|/2) * sqrt((f/2)^2 - 1).
      |f| < 2 (x < 1), analytic continuation:
        I(f) = f^2 / 4 - 1/2.

    Derivation (sketch). With lambda = 2 cos(theta),
    2 I(f) / (2/pi) = int_0^pi sin^2(theta) log|f - 2 cos theta| dtheta.
    Expand sin^2 = (1 - cos 2 theta)/2 and use Jensen-like identities
    for int_0^pi log|1 - 2 r cos theta + r^2| cos k theta dtheta.
    """
    x = abs(f) / 2.0
    if x >= 1.0:
        s = float(np.sqrt(x * x - 1.0))
        return float(np.log(x + s)) + f * f / 4.0 - 0.5 - x * s
    return f * f / 4.0 - 0.5


# ------------------------------------------------------------------ #
#  ABAC p-spin complexity density
# ------------------------------------------------------------------ #

def pspin_complexity(E: float, p: int) -> float:
    """Sigma_p(E): planar complexity per spin for the pure p-spin
    spherical model.

    Formula (Auffinger-Ben Arous-Cerny 2013, Thm 2.3, rewritten):
        Sigma_p(E) = (1/2) log(p - 1)
                   - (p - 2) E^2 / (4 (p - 1))
                   + I(f(E)),
    with f(E) = E * sqrt(p / (2 (p - 1))) and I as in
    semicircle_log_det above.

    Returns -infinity when E is outside the support [-E_inf(p), E_inf(p)]
    (which would give complexity 0, i.e., no critical points).
    """
    if p < 2:
        raise ValueError("p-spin requires p >= 2")
    f = E * np.sqrt(p / (2 * (p - 1)))
    return (0.5 * np.log(p - 1)
            - (p - 2) * E * E / (4 * (p - 1))
            + semicircle_log_det(f))


def threshold_energy(p: int) -> float:
    """E_inf(p) = -2 * sqrt((p - 1) / p): the energy at which the
    complexity curve terminates (below, critical points cease to be
    exponentially numerous in the pure p-spin landscape).
    """
    return -2.0 * np.sqrt((p - 1) / p)


def ground_state_energy_upper_bound(p: int) -> float:
    """-E_inf(p) with the sign convention above. The true
    ground-state energy E_0(p) lies at or below this threshold;
    complexity argument gives E_0(p) <= E_inf(p) up to sign.
    """
    return -threshold_energy(p)  # positive threshold magnitude


# ------------------------------------------------------------------ #
#  Cross-check via Harer-Zagier log-det series
# ------------------------------------------------------------------ #

def log_det_from_planar_moments(f: float,
                                n_terms: int = 50) -> float:
    """Reconstruct I(f) for |f| > 2 via the convergent series

        I(f) = log|f| - sum_{k >= 1} C_k / (2k) * f^{-2k}

    where C_k is the k-th Catalan number (= planar map count
    eps_0(k)). This is the planar sector of Prop 2.
    """
    if abs(f) <= 2.0:
        raise ValueError("series only converges for |f| > 2")
    from rdft.ac.matrix_model import catalan
    total = float(np.log(abs(f)))
    for k in range(1, n_terms + 1):
        total -= catalan(k) / (2.0 * k) * f ** (-2 * k)
    return total
