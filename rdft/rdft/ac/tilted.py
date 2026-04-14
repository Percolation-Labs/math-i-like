"""
rdft.ac.tilted
==============
Tilted Doi-Peliti DSEs and the SCGF of currents.

For a CRN with reactions {(k_i, l_i, lambda_i)}, tilting the rate of one
reaction by a factor exp(s) is equivalent to studying the moment generating
function of the net flux through that channel:
    Z(s; t) = E[ exp(s * N_channel(t)) ]
where N_channel(t) is the number of times the channel has fired by time t.
The scaled cumulant generating function is
    lambda(s) = lim_{t -> oo} (1/t) log Z(s; t)
and equals -log|z_star(s)| where z_star(s) is the dominant branch of the
tilted DSE.

Stochastic-thermodynamics significance
--------------------------------------
For thermodynamically meaningful tilts (entropy production, heat dissipation,
specific currents), this is the standard Donsker-Varadhan / Touchette /
Garrahan-Chandler framework.  CFAC adds the algebraic structure: the tilt
deforms phi to phi_s and the SCGF is the dominant-branch radius of the
deformed algebraic curve.  Stratification crossings produce dynamical phase
transitions in lambda(s).
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

from .stratification import (
    phi_from_reactions,
    discriminant_roots,
    lambda_scgf,
    puiseux_order,
)


# ------------------------------------------------------------------ #
#  Rate tilting
# ------------------------------------------------------------------ #

def tilt_reaction_rates(reactions: list[tuple[int, int, float]],
                         tilt_indices: Sequence[int],
                         s: float) -> list[tuple[int, int, float]]:
    """Multiply selected reactions' rates by exp(s).

    Parameters
    ----------
    reactions : list of (k_in, l_out, rate)
    tilt_indices : indices of reactions whose rate to tilt
    s : tilt parameter

    Returns
    -------
    new reaction list with selected rates multiplied by exp(s).
    """
    tilted = []
    for i, (k, l, rate) in enumerate(reactions):
        new_rate = rate * np.exp(s) if i in tilt_indices else rate
        tilted.append((k, l, new_rate))
    return tilted


def tilted_phi(reactions: list[tuple[int, int, float]],
                tilt_indices: Sequence[int],
                s: float) -> list[float]:
    """Compute phi_s(G) for the rate-tilted CRN."""
    tilted = tilt_reaction_rates(reactions, tilt_indices, s)
    return phi_from_reactions(tilted)


def tilted_phi_coeffs(phi_coeffs: Sequence[float],
                       coeff_index: int,
                       s: float) -> list[float]:
    """Tilt one coefficient of phi by exp(s).

    Useful when starting from a phi (e.g.\ the canonical family) rather than
    a CRN rate set.  At s=0 the result equals phi_coeffs.  This is the
    natural "tilt" in coefficient space rather than rate space.
    """
    out = list(phi_coeffs)
    out[coeff_index] = out[coeff_index] * float(np.exp(s))
    return out


def tilted_phi_canonical(k: int, beta: float, s: float) -> list[float]:
    """Tilt the canonical family phi_{k, beta}(G) = (1+G)^k + beta G by
    multiplying beta by exp(s).  At s=0 sits on C_k with branch point
    z* = 1/beta; for s != 0, beta -> beta exp(s) and the branch may move
    or change order if dominance is lost."""
    from .stratification import canonical_family
    return canonical_family(k, beta * float(np.exp(s)))


# ------------------------------------------------------------------ #
#  Entropy-production tilt for paired forward/reverse reactions
# ------------------------------------------------------------------ #

def entropy_production_tilt(reactions: list[tuple[int, int, float]],
                              pairs: list[tuple[int, int]],
                              s: float) -> list[tuple[int, int, float]]:
    """Tilt paired forward/reverse reactions by the entropy-production current.

    For each pair (i_forward, i_reverse) of indices into ``reactions``, the
    affinity A = ln(rate_forward / rate_reverse) defines the entropy
    produced per net forward transition.  Tilting trajectory weight by
    exp(s * sigma) multiplies the forward rate by exp(s * A) and the
    reverse rate by exp(-s * A).

    Gallavotti-Cohen prediction: the SCGF lambda_sigma(s) of this tilt
    satisfies lambda_sigma(s) = lambda_sigma(-1 - s) for ANY paired CRN
    with finite affinity.  Test by computing the residual.
    """
    new_reactions = list(reactions)
    for i_fwd, i_rev in pairs:
        k_fwd, l_fwd, rate_fwd = new_reactions[i_fwd]
        k_rev, l_rev, rate_rev = new_reactions[i_rev]
        if rate_rev == 0 or rate_fwd == 0:
            continue
        A = float(np.log(rate_fwd / rate_rev))  # affinity
        new_reactions[i_fwd] = (k_fwd, l_fwd, rate_fwd * np.exp(s * A))
        new_reactions[i_rev] = (k_rev, l_rev, rate_rev * np.exp(-s * A))
    return new_reactions


def entropy_production_scgf(reactions: list[tuple[int, int, float]],
                              pairs: list[tuple[int, int]],
                              s_values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """SCGF lambda_sigma(s) for the entropy-production current of paired
    reactions.  Returns (s_array, lambda_array)."""
    from .stratification import lambda_scgf
    s_arr = np.asarray(s_values)
    lams = np.array([
        lambda_scgf(phi_from_reactions(
            entropy_production_tilt(reactions, pairs, s)
        ))
        for s in s_arr
    ])
    return s_arr, lams


def gallavotti_cohen_test(reactions, pairs, s_values):
    """Compute lambda_sigma(s) - lambda_sigma(-1-s) over s_values.

    Predicts zero residual under the entropy-production tilt for any
    properly-paired DP CRN.  Nonzero residual indicates either (a) the
    tilt is not the entropy-production current (wrong implementation), or
    (b) the test is in a regime where the SCGF is non-analytic and the
    formal symmetry is broken.
    """
    s_arr, lam_pos = entropy_production_scgf(reactions, pairs, s_values)
    _, lam_mirror = entropy_production_scgf(reactions, pairs,
                                              [-1 - s for s in s_values])
    return s_arr, lam_pos, lam_mirror, lam_pos - lam_mirror


# ------------------------------------------------------------------ #
#  SCGF
# ------------------------------------------------------------------ #

def scgf(reactions: list[tuple[int, int, float]],
          tilt_indices: Sequence[int],
          s_values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute lambda(s) over a grid of s values.

    Returns (s_array, lambda_array).
    """
    lams = np.array([lambda_scgf(tilted_phi(reactions, tilt_indices, s)) for s in s_values])
    return np.asarray(s_values), lams


def scgf_with_branch(reactions, tilt_indices, s_values):
    """Compute lambda(s) AND the dominant branch z_star(s)."""
    s_arr = np.asarray(s_values)
    lams = np.zeros_like(s_arr, dtype=float)
    zs_abs = np.zeros_like(s_arr, dtype=float)
    zs_arg = np.zeros_like(s_arr, dtype=float)
    branch_orders = np.zeros_like(s_arr, dtype=int)
    for i, s in enumerate(s_arr):
        phi = tilted_phi(reactions, tilt_indices, s)
        k, z = puiseux_order(phi)
        lams[i] = -np.log(abs(z)) if abs(z) > 0 else np.nan
        zs_abs[i] = abs(z)
        zs_arg[i] = np.angle(z)
        branch_orders[i] = k
    return s_arr, lams, zs_abs, zs_arg, branch_orders


# ------------------------------------------------------------------ #
#  Diagnostics
# ------------------------------------------------------------------ #

def gallavotti_cohen_residual(reactions, tilt_indices, s_values):
    """Compute lambda(s) - lambda(-1-s) for GC symmetry test.

    For thermodynamically meaningful tilts (specifically the entropy
    production current), GC predicts lambda(s) = lambda(-1-s).  Any nonzero
    residual indicates either (a) the tilt is not the entropy-production
    current, or (b) GC symmetry is broken.
    """
    s_arr, lam_pos = scgf(reactions, tilt_indices, s_values)
    _, lam_mirror = scgf(reactions, tilt_indices, [-1 - s for s in s_values])
    return s_arr, lam_pos - lam_mirror


def detect_dynamical_phase_transition(reactions, tilt_indices, s_values,
                                        d_threshold: float = 1.5):
    """Detect kink in lambda(s) — first-order dynamical phase transition.

    Returns indices of s_values where d^2 lambda / d s^2 exceeds threshold
    (a heuristic for kink detection).
    """
    s_arr, lams, *_ = scgf_with_branch(reactions, tilt_indices, s_values)
    d_lam = np.gradient(lams, s_arr)
    d2_lam = np.gradient(d_lam, s_arr)
    kink_indices = np.where(np.abs(d2_lam) > d_threshold * np.median(np.abs(d2_lam) + 1e-12))[0]
    return s_arr, lams, d_lam, d2_lam, kink_indices
