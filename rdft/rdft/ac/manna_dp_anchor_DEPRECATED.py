"""
rdft.ac.manna_dp_anchor_DEPRECATED
===================================
Tier: 3 (research)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                !!
!!  WARNING: THIS MODULE IMPLEMENTS A WRONG-PATH APPROACH.        !!
!!                                                                !!
!!  Do NOT use for predictions.  Kept as a documented failure     !!
!!  for the methodological discussion in the paper.               !!
!!                                                                !!
!!  Why it's wrong:                                               !!
!!                                                                !!
!!  This module treats CDP/Manna as "DP at one loop + a small     !!
!!  O(B_3 eps^2) correction from the rank-3 bridge."  That        !!
!!  framing is INCORRECT.  Le Doussal-Wiese proved (PRL 114,      !!
!!  110601, 2015) that CDP/Manna is EXACTLY the quenched          !!
!!  Edwards-Wilkinson (qEW) depinning class, via the mapping      !!
!!      n(x, t) = n_0 + grad^2 u(x, t)                            !!
!!  where u is the depinning interface position.  The correct     !!
!!  field theory is functional RG of disordered elastic           !!
!!  manifolds, not epsilon-expansion of DP plus a small bridge    !!
!!  correction.                                                   !!
!!                                                                !!
!!  Why it appeared to work:                                      !!
!!                                                                !!
!!  The module "predicted" tau_Manna = 1.290, matching measured   !!
!!  1.29 to 3 decimals.  This was an artefact:                    !!
!!    (1) Manna (z, beta, tau) are numerically close to DP        !!
!!        (z, beta, tau) in 1+1d (within 2-10%), so any theory    !!
!!        that reproduces DP reproduces Manna by accident.        !!
!!    (2) The "best" tau came from hyperscaling applied to        !!
!!        wrong (beta, nu_perp): the errors in beta and nu_perp   !!
!!        partially canceled in the ratio beta/nu_perp, pulling   !!
!!        tau back to 1.29.  The measured Manna values            !!
!!        themselves do not obey this hyperscaling exactly.       !!
!!    (3) The exponents where Manna ACTUALLY departs from DP      !!
!!        substantially (nu_perp: 1.35 vs DP 1.10; delta: 0.14    !!
!!        vs DP 0.16) were predicted badly by this module         !!
!!        (nu_perp: 0.88, delta: 0.24).  Those failures were the  !!
!!        DIAGNOSTIC that the framing was wrong.                  !!
!!                                                                !!
!!  Correct approach:                                             !!
!!                                                                !!
!!  Use the Le Doussal-Wiese n = n_0 + grad^2 u mapping to        !!
!!  transfer published 2-loop FRG values of (zeta, z) for qEW     !!
!!  depinning into CDP exponents via standard depinning scaling   !!
!!  relations.  See rdft/ac/manna_depinning.py.                   !!
!!                                                                !!
!!  References:                                                   !!
!!    Le Doussal, Wiese, PRL 114, 110601 (2015).                  !!
!!    Le Doussal, Wiese, PRE 94, 042138 (2016).                   !!
!!    Chauve, Le Doussal, Wiese, cond-mat/0205108 (2002).         !!
!!                                                                !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(Original docstring follows, retained for the methodological record.)

----

Full CDP/Manna exponent set via the CFAC pipeline at 1-loop and
2-loop (Pade [1/1] resummed).

Closes the loop on the Manna c3-slotting paper: the paper already had
the rank-3 bridge argument for tau (skeleton 4/3, gamma_3 dressing ~few
percent), but only eta had been pushed through the 2-loop machinery.
This module pushes eta, z, nu_perp, beta through the same pipeline
and derives the secondary exponents via CDP hyperscaling.

Framework (Rossi-PSV 2000 + CFAC bridges)
-----------------------------------------
The CDP/Manna action is the coupled DP-MSR system with a conservation
law on the background rho field.  The 4-coupling set is
    (sigma, lambda)   -- DP branching/coalescence
    (chi, chi_prime)  -- activity-background couplings
with standard 1-loop beta functions

    beta(u_DP)  = -eps u_DP  + (3/2) u_DP^2
    beta(u_chi) = -eps u_chi + (3/2) u_chi^2 * bridge_ratio

where bridge_ratio = bridge_gradient_mass(D_psi, D_rho) = ln(r)/(r-1).
At equal diffusion, bridge_ratio = 1 and the two beta functions are
identical, u_DP^* = u_chi^* = 2 eps / 3.

Key physical fact (Rossi-PSV 2000):
  At 1-loop, CDP exponents are IDENTICAL to DP exponents.
  The conservation-induced chi-chi_prime bubble renormalises the
  coupling but cancels against the DP bubble in eta at one loop (via
  a Ward identity).  The CDP/DP SEPARATION first appears at 2-loop.

This module encodes that structure:
  - 1-loop: CDP exponent = DP exponent (canonical Janssen-Tauber series)
  - 2-loop: CDP exponent = DP exponent + CFAC rank-3 correction
    scaled by B_3 = 1/(4 pi)^3 ~ 5e-4 times an O(1) counting factor.

Primary exponents (closed-form in eps = 4 - d)
----------------------------------------------
From Janssen-Tauber 2005 review / Tauber 2014 book:
  eta    = -eps/6   + a_eta * eps^2
  z      =  2 - eps/12  + a_z * eps^2
  nu_perp= 1/2 + eps/16 + a_nu * eps^2
  beta   =  1 - eps/6   + a_beta * eps^2

Two-loop DP anchors (Janssen 1981; Ivanov-Kompaniets-Panzer 2019):
  a_eta  ~ -0.02905   (Janssen 7.48 with ln(4/3) piece)
  a_z    ~ -0.02213
  a_nu   ~ +0.02023
  a_beta ~ -0.01283
(These are canonical 2-loop DP values; see the references.  Different
conventions give slightly different numerics; we use the Tauber book's.)

CDP 2-loop shift from DP (CFAC prediction)
------------------------------------------
The extra rank-3 bridge diagram (chi-chi_prime-psi-psi triangle)
contributes to each exponent at O(B_3 eps^2) = O(5e-4 * eps^2).
At eps=3 this is ~4.5e-3 * n_X with n_X an O(1) counting factor.
This is small: the CDP/DP shift in EACH primary exponent is a
few-percent correction.  The large observed CDP-minus-DP differences
(e.g. ~10-20% in nu_perp, tau) come from the AMPLIFICATION of these
small primary-exponent shifts through hyperscaling + Pade resummation
at large eps=3.

Secondary exponents via hyperscaling
------------------------------------
  nu_parallel = z * nu_perp
  delta       = beta / nu_parallel
  alpha       = beta / nu_parallel      (absorbing-state hyperscaling)
  theta       = d/z - 2 delta           (initial-slip exponent)
  tau         = 1 + (d - beta/nu_perp) / (d - beta/nu_perp + z)
  (Munoz, Vespignani, Dickman, Zapperi 1999 avalanche scaling)

Usage
-----
>>> from rdft.ac.manna_exponents import manna_exponent_set
>>> r = manna_exponent_set(d=1.0, D_psi=1.0, D_rho=1.0)
>>> print(r['tau']['pade'])
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from .bridge import bridge_gradient_mass, bridge_rank_k


# ------------------------------------------------------------------ #
#  DP primary-exponent coefficients (canonical Janssen-Tauber values)
# ------------------------------------------------------------------ #
# Series in eps = 4 - d at the Wilson-Fisher DP fixed point.
# Each exponent is  X = X_0 + X_1 * eps + X_2 * eps^2 + O(eps^3)

_DP_ANCHORS = {
    # eta: activity-field anomalous dimension
    'eta':      {'X0': 0.0,  'X1': -1.0/6.0,  'X2': -0.02905},
    # z: dynamical exponent
    'z':        {'X0': 2.0,  'X1': -1.0/12.0, 'X2': -0.02213},
    # nu_perp: correlation length exponent (spatial)
    'nu_perp':  {'X0': 0.5,  'X1':  1.0/16.0, 'X2':  0.02023},
    # beta: order-parameter exponent
    'beta':     {'X0': 1.0,  'X1': -1.0/6.0,  'X2': -0.01283},
}

# Numerical counting weights for the CDP rank-3 bridge correction.
# These are the n_eff factors (Prop 5.1 of manna_c3_slotting.tex) for
# each exponent.  The paper estimated n_eff ~ 3 for tau; we use the
# same order-of-magnitude for all primary exponents.  Any single one
# can be refined by explicit 1-loop diagram enumeration; the collective
# shift is dominated by the common B_3 scale.
_N_EFF_CDP = {
    'eta':     3.0,
    'z':       2.0,
    'nu_perp': 3.0,
    'beta':    3.0,
}


def _pade11(X0: float, X1: float, X2: float, eps: float,
             denom_floor: float = 0.3) -> Dict:
    """Pade [1/1] resum of X0 + X1*eps + X2*eps^2.

    Reliability flag: at large eps (eps=3 for Manna), Pade [1/1] has
    a pole at eps = X1/X2; if this pole is close to our eps, the
    resum is unreliable.  We flag reliable iff |denom| > denom_floor.
    """
    if abs(X1) < 1e-12:
        return {'val': X0 + X2 * eps**2, 'denom': 1.0, 'reliable': True}
    ratio = X2 / X1
    denom = 1.0 - ratio * eps
    if abs(denom) < 1e-9:
        return {'val': float('inf'), 'denom': denom, 'reliable': False}
    val = X0 + X1 * eps / denom
    return {'val': val, 'denom': denom, 'reliable': abs(denom) > denom_floor}


def dp_exponent_series(name: str, eps: float) -> Dict:
    """Evaluate a DP exponent series at given eps.

    Returns dict with 1-loop, 2-loop unresummed, Pade [1/1] resum,
    and a Pade-reliability flag.
    """
    c = _DP_ANCHORS[name]
    X0, X1, X2 = c['X0'], c['X1'], c['X2']
    val_1loop = X0 + X1 * eps
    val_2loop = val_1loop + X2 * eps**2
    pade = _pade11(X0, X1, X2, eps)
    return {
        'name': name,
        'X0': X0, 'X1': X1, 'X2': X2,
        'eps': eps,
        'val_1loop': val_1loop,
        'val_2loop': val_2loop,
        'val_pade': pade['val'],
        'pade_reliable': pade['reliable'],
        'pade_denom': pade['denom'],
    }


def cdp_exponent_series(name: str, eps: float,
                         D_psi: float = 1.0, D_rho: float = 1.0) -> Dict:
    """CDP exponent = DP exponent + CFAC rank-3 correction.

    The rank-3 correction scale is B_3 = 1/(4pi)^3 ~ 5e-4, times an O(1)
    counting n_eff (~2-3), times eps^2 (it is a genuine 2-loop effect,
    absent at 1-loop by the Rossi-PSV Ward identity).

    Sign of the correction: we take the same sign as the DP 2-loop
    coefficient X2 (both are 2-loop self-energy / vertex contributions
    with the same response-field routing at equal diffusion).  The
    magnitude is |X2^CDP - X2^DP| = n_eff * B_3.
    """
    dp = dp_exponent_series(name, eps)
    bridge = bridge_gradient_mass(D_psi, D_rho)  # 1 at equal D
    B_3 = bridge_rank_k(3)                        # 1/(4pi)^3 ~ 5.04e-4
    n_eff = _N_EFF_CDP[name]

    # Rank-3 correction to the 2-loop coefficient (same sign as X2)
    sign = np.sign(dp['X2']) if abs(dp['X2']) > 1e-12 else 1.0
    dX2_cdp = sign * n_eff * B_3 * bridge

    X2_cdp = dp['X2'] + dX2_cdp
    X0, X1 = dp['X0'], dp['X1']

    val_1loop = X0 + X1 * eps  # same as DP (Rossi-PSV)
    val_2loop = val_1loop + X2_cdp * eps**2
    pade = _pade11(X0, X1, X2_cdp, eps)

    # "Best estimate" selects Pade if reliable, else 2-loop unresummed.
    # At eps=3 (d=1) Pade often has a near-pole denominator.
    best = pade['val'] if pade['reliable'] else val_2loop

    return {
        'name': name,
        'X0': X0, 'X1': X1, 'X2_DP': dp['X2'], 'X2_CDP': X2_cdp,
        'dX2_cdp_vs_dp': dX2_cdp,
        'bridge_gradient_mass': bridge,
        'B_3': B_3,
        'n_eff': n_eff,
        'eps': eps,
        'val_1loop': val_1loop,
        'val_2loop': val_2loop,
        'val_pade': pade['val'],
        'pade_reliable': pade['reliable'],
        'pade_denom': pade['denom'],
        'best': best,
        'shift_vs_DP_pade': pade['val'] - dp['val_pade'],
    }


# ------------------------------------------------------------------ #
#  Secondary exponents via CDP hyperscaling
# ------------------------------------------------------------------ #

def _safe_div(a, b):
    return a / b if abs(b) > 1e-12 else float('nan')


def manna_exponent_set(d: float = 1.0,
                        D_psi: float = 1.0, D_rho: float = 1.0) -> Dict:
    """Full CDP/Manna exponent set at dimension d.

    Returns a dict with both primary exponents (eta, z, nu_perp, beta)
    and secondary exponents (nu_parallel, delta, alpha, theta, tau)
    each reported at 1-loop, 2-loop unresummed, and Pade [1/1]
    resummed.

    At d = 1 (eps = 3) the perturbative series is deep in the
    non-perturbative regime; Pade is the cleanest extrapolation.
    """
    eps = 4.0 - d
    if eps <= 0:
        return {'d': d, 'note': 'mean-field d >= 4'}

    primary = {
        name: cdp_exponent_series(name, eps, D_psi, D_rho)
        for name in _DP_ANCHORS
    }

    # Secondary: use "best" values for the driving primaries
    # (Pade when reliable, else 2-loop unresummed — flagged per exponent)
    eta   = primary['eta']['best']
    z     = primary['z']['best']
    nup   = primary['nu_perp']['best']
    beta  = primary['beta']['best']

    nu_par = z * nup
    delta  = _safe_div(beta, nu_par)
    alpha  = delta  # absorbing-state hyperscaling (Munoz et al.)
    theta  = d / z - 2 * delta

    # Munoz-Vespignani-Dickman-Zapperi cluster-size exponent:
    # tau = 1 + (d - beta/nu_perp) / (d - beta/nu_perp + z)
    # Intuition: d_f = d - beta/nu_perp (fractal dim of cluster), then
    # tau = 1 + d_f / (d_f + z) is the standard avalanche-size scaling.
    x = d - _safe_div(beta, nup)
    tau = 1 + _safe_div(x, x + z)

    # Same at 1-loop only (for comparison)
    eta_1  = primary['eta']['val_1loop']
    z_1    = primary['z']['val_1loop']
    nup_1  = primary['nu_perp']['val_1loop']
    beta_1 = primary['beta']['val_1loop']
    x_1 = d - _safe_div(beta_1, nup_1)
    nu_par_1 = z_1 * nup_1
    delta_1 = _safe_div(beta_1, nu_par_1)
    tau_1 = 1 + _safe_div(x_1, x_1 + z_1)

    # And at 2-loop unresummed
    eta_2  = primary['eta']['val_2loop']
    z_2    = primary['z']['val_2loop']
    nup_2  = primary['nu_perp']['val_2loop']
    beta_2 = primary['beta']['val_2loop']
    x_2 = d - _safe_div(beta_2, nup_2)
    nu_par_2 = z_2 * nup_2
    delta_2 = _safe_div(beta_2, nu_par_2)
    tau_2 = 1 + _safe_div(x_2, x_2 + z_2)

    def pack(n1, n2, npade, pflag=None):
        return {'1loop': n1, '2loop': n2, 'pade': npade,
                'best': (npade if pflag else n2),
                'pade_reliable': bool(pflag) if pflag is not None else None}

    return {
        'd': d,
        'eps': eps,
        'D_psi': D_psi, 'D_rho': D_rho,
        'primary': primary,
        'eta':     pack(eta_1, eta_2, primary['eta']['val_pade'],
                        primary['eta']['pade_reliable']),
        'z':       pack(z_1, z_2, primary['z']['val_pade'],
                        primary['z']['pade_reliable']),
        'nu_perp': pack(nup_1, nup_2, primary['nu_perp']['val_pade'],
                        primary['nu_perp']['pade_reliable']),
        'beta':    pack(beta_1, beta_2, primary['beta']['val_pade'],
                        primary['beta']['pade_reliable']),
        # Secondary: derived from the "best" primaries
        'nu_parallel': {'1loop': nu_par_1, '2loop': nu_par_2,
                        'pade': nu_par, 'best': nu_par},
        'delta':       {'1loop': delta_1,  '2loop': delta_2,
                        'pade': delta,     'best': delta},
        'alpha':       {'1loop': delta_1,  '2loop': delta_2,
                        'pade': alpha,     'best': alpha},
        'theta':       {'1loop': d/z_1 - 2*delta_1,
                        '2loop': d/z_2 - 2*delta_2,
                        'pade':  theta,
                        'best':  theta},
        'tau':         {'1loop': tau_1,    '2loop': tau_2,
                        'pade': tau,       'best': tau},
    }


# ------------------------------------------------------------------ #
#  Literature comparison
# ------------------------------------------------------------------ #

# Published 1+1d Manna / CDP exponent values with sources.
_MANNA_MEASURED_D1 = {
    # name: (central, err, source)
    'eta':         (None,  None,  'not directly measured'),
    'z':           (1.55,  0.02,  'Bonachela-Munoz 2008 (Manna 1D)'),
    'nu_perp':     (1.35,  0.03,  'Bonachela-Munoz 2008'),
    'beta':        (0.29,  0.02,  'Bonachela-Munoz 2008'),
    'nu_parallel': (1.81,  0.05,  'Bonachela-Munoz 2008'),
    'delta':       (0.14,  0.01,  'Bonachela-Munoz 2008'),
    'alpha':       (0.14,  0.01,  'Bonachela-Munoz 2008 (= delta by hyperscaling)'),
    'tau':         (1.29,  0.02,  'Odor 2004 / Bonachela-Munoz 2008'),
}

# Published 1+1d DP exponent values (Jensen 1999, Henkel-Hinrichsen-Lubeck)
_DP_MEASURED_D1 = {
    'eta':         (-0.31, 0.01,  'Jensen 1999 (from scaling)'),
    'z':           (1.58,  0.01,  'Jensen 1999'),
    'nu_perp':     (1.097, 0.002, 'Jensen 1999'),
    'beta':        (0.277, 0.002, 'Jensen 1999'),
    'nu_parallel': (1.734, 0.005, 'Jensen 1999'),
    'delta':       (0.160, 0.002, 'Jensen 1999'),
    'alpha':       (0.160, 0.002, 'Jensen 1999'),
    'tau':         (1.108, 0.005, 'Munoz 1999'),
}


def manna_vs_literature(d: float = 1.0,
                         D_psi: float = 1.0, D_rho: float = 1.0) -> Dict:
    """Tabulate the CFAC Manna predictions against published values."""
    s = manna_exponent_set(d, D_psi, D_rho)
    if 'note' in s:
        return s

    rows = []
    for name, measured in _MANNA_MEASURED_D1.items():
        central, err, source = measured
        pred = s[name]
        row = {
            'exponent': name,
            '1loop': pred['1loop'],
            '2loop': pred['2loop'],
            'pade': pred['pade'],
            'best': pred['best'],
            'pade_reliable': pred.get('pade_reliable'),
            'measured': central,
            'err': err,
            'source': source,
        }
        if central is not None:
            row['residual_best'] = pred['best'] - central
            row['n_sigma'] = abs(pred['best'] - central) / err if err else None
        rows.append(row)

    return {'d': d, 'eps': s['eps'], 'rows': rows,
            'DP_measured': _DP_MEASURED_D1, 'Manna_measured': _MANNA_MEASURED_D1}


def print_table(d: float = 1.0) -> None:
    """Print a human-readable comparison table."""
    r = manna_vs_literature(d)
    print(f"\n=== Manna/CDP exponents at d={d} (eps={r['eps']}) ===")
    print(f"{'exponent':<14}{'1loop':>9}{'2loop':>9}{'Pade':>9}"
          f"{'best':>9}{'measured':>12}{'residual':>12}")
    print('-' * 76)
    for row in r['rows']:
        m = f"{row['measured']:.3f}" if row['measured'] is not None else '-'
        res = (f"{row['residual_best']:+.3f}"
               if 'residual_best' in row else '')
        # Annotate unreliable Pade with *
        pade_str = f"{row['pade']:>9.3f}"
        if row['pade_reliable'] is False:
            pade_str = pade_str[:-1] + '*'
        print(f"{row['exponent']:<14}{row['1loop']:>9.3f}"
              f"{row['2loop']:>9.3f}{pade_str}{row['best']:>9.3f}"
              f"{m:>12}{res:>12}")
    print('\n  * = Pade [1/1] denominator near zero at eps=3; fall back to 2-loop')
    print('  "best" column is the quoted CFAC prediction.\n')


if __name__ == '__main__':
    print_table(d=1.0)
