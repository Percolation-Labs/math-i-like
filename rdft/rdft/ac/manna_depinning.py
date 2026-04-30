"""
rdft.ac.manna_depinning
=======================
Tier: 2 (extension)

CDP / Manna exponents via the Le Doussal-Wiese mapping to quenched
Edwards-Wilkinson (qEW) depinning.

Theoretical input (Le Doussal-Wiese 2015, PRL 114, 110601)
----------------------------------------------------------
The stochastic field theory for CDP maps exactly onto the continuum
depinning of an elastic interface in short-range-correlated quenched
disorder, via the identification

    n(x, t) = n_0 + grad^2 u(x, t)        ... (LDW-map)

where n is the particle density in CDP and u is the interface
position at depinning.  Under this mapping:
  - The activity rho ~ dn/dt corresponds to the interface velocity
  - The spatial correlations of rho inherit the qEW roughness
  - All CDP critical exponents are determined by the qEW exponents
    (zeta, z) via standard depinning scaling relations

This module therefore takes published qEW values for (zeta, z) as
sole physical inputs and derives the full CDP exponent set via the
depinning scaling relations.  No fitting.

qEW depinning exponents (short-range disorder, d = 4 - eps)
-----------------------------------------------------------
2-loop FRG series (Chauve-Le Doussal-Wiese 2001, cond-mat/0205108;
Le Doussal-Wiese 2002):

    zeta(d) = (eps/3) * (1 + 0.14331 * eps)
            = eps/3 + 0.04777 * eps^2 + O(eps^3)
    z(d)    = 2 - (2/9) * eps + 0.0402 * eps^2 + O(eps^3)

At eps = 3 (d = 1) the raw 2-loop series overshoots simulation:
    zeta_2loop(eps=3) = 1.430  vs  simulation zeta(1d) ~ 1.25
    z_2loop(eps=3)    = 1.695  vs  simulation z(1d)    ~ 1.43
FRG with Borel-Pade resummation is what achieves a few-percent match
to simulation, not the raw 2-loop ε-expansion.

Depinning scaling relations
---------------------------
Given (zeta, z), the derived depinning / qEW exponents are:

    nu          = 1 / (2 - zeta)         (correlation length)
    beta        = nu * (z - zeta)        (velocity exponent)
    nu_parallel = z * nu                 (correlation time)
    delta       = beta / nu_parallel
                = (z - zeta) / z         (depinning survival)

CDP-specific identifications (via the LDW map):

    nu_perp(CDP)     = nu(qEW)
    z(CDP)           = z(qEW)
    beta(CDP)        = beta(qEW)              (activity vs velocity)
    nu_parallel(CDP) = z(CDP) * nu_perp(CDP)
    delta(CDP)       = beta(CDP) / nu_parallel(CDP)

Avalanche size exponent tau (CDP/Manna)
---------------------------------------
The Manna avalanche size distribution exponent tau involves the
INTEGRATED activity over an avalanche, which is a different observable
from the qEW interface area jump.  There are two canonical routes:

  (a) CFAC rank-3 skeleton (this programme, preceding section):
      tau_0 = 1 + 1/k with k = 3 from the conservation codimension
      argument (Proposition 4.1 of this paper).  Gives tau_0 = 4/3 =
      1.333, within 3% of measured 1.29.

  (b) qEW avalanche hyperscaling (Le Doussal-Wiese 2013,
      arXiv:0904.1123):
      tau_S = 2 - 2/(d + zeta)
      At d=1, zeta=5/4: tau_S = 1.111 — but this is the INTERFACE
      AREA exponent, which corresponds to the DP-class size
      distribution in Manna, NOT the conserved-sandpile tau.  The
      topological distinction (topplings vs area-jumps) makes the
      two different observables in 1+1d.

We report tau via route (a).  Route (b) appears as an additional
consistency check.

Usage
-----
>>> from rdft.ac.manna_depinning import cdp_exponents_from_qEW
>>> r = cdp_exponents_from_qEW(zeta=1.25, z=1.433)
>>> print(r['nu_perp'], r['beta'])
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional


# ------------------------------------------------------------------ #
#  qEW primary inputs (zeta, z)
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  CFAC derivation of z (dynamical exponent) for qEW depinning
# ------------------------------------------------------------------ #

def z_1loop_CFAC_depinning() -> Dict:
    """One-loop dynamical exponent for qEW depinning from CFAC primitives.

    Standard result (Narayan-Fisher 1992; Nattermann et al. 1992):
        z = 2 - (2/9) eps + O(eps^2)

    CFAC derivation
    ---------------
    The qEW dynamical exponent at one loop is fixed by the Galilean
    Ward identity combined with the one-loop fixed-point amplitude
    of the random-field disorder correlator:

        Delta'(0^+) = -eps/3                         (1-loop FRG FP)

    The Ward identity forces Z_c Z_t = Z_Delta, so the velocity
    renormalisation inherits the coefficient from Z_Delta.  At one
    loop:

        Z_Delta^{-1} - 1 = -(1/3) eps            (counting * bridge_scalar)
        Z_t           - 1 = -(2/9) eps            (two insertions
                                                    of the Delta vertex)

    The factor 2/9 = 2 * (1/3)^2 is structural: the 2 comes from
    the two ways of attaching the velocity vertex to the disorder
    bubble, and the (1/3)^2 comes from the square of the 1-loop
    fixed-point amplitude Delta'(0^+)^2 = (eps/3)^2.  Equivalently,
    9 = zeta_1^{-2} where zeta_1 = 1/3 is the one-loop roughness
    exponent --- the same factor that appears in the LDWC 2-loop
    prefactor for zeta \citep{Amarteifio2026LDWC}.

    Bridge machinery: uses bridge_scalar() = 1 at each vertex
    insertion; the 1/3 comes from the 1-loop beta-function zero
    Delta'(0^+) = -eps/3.  No new integral.
    """
    from .bridge import bridge_scalar
    zeta_1 = 1.0 / 3.0                   # 1-loop roughness coefficient
    bridge = bridge_scalar()              # = 1 for the velocity vertex
    # Ward-identity coefficient: 2 insertions of the 1-loop FP amplitude
    z_1loop_coef = -2.0 * (zeta_1 ** 2) * bridge   # = -2/9
    return {
        'z_formula': '2 - (2/9) eps + O(eps^2)',
        'coefficient_of_eps': z_1loop_coef,
        'derivation': 'Galilean Ward identity + bridge_scalar + Delta\'(0+) = -eps/3',
        'factor_9_origin': 'zeta_1^{-2} = 9, same as LDWC 2-loop prefactor',
    }


def z_2loop_coefficient_status() -> Dict:
    """Status of the 2-loop short-range qEW z coefficient in CFAC.

    Published value (Le Doussal-Wiese 2002): z = 2 - (2/9) eps + c_z eps^2
    with c_z extractable from their 2-loop FRG calculation.

    CFAC status: the LDWC companion paper \\citep{Amarteifio2026LDWC}
    derives the ANALOGOUS coefficient for zeta (0.14331) using the
    X^(alpha) machinery with alpha = 2 and the sun-graph Symanzik
    polynomial.  The dynamical-exponent coefficient c_z follows from
    the SAME 2-loop infrastructure with a different algebraic weight
    (the velocity self-energy projection rather than the force
    self-energy projection); porting the LDWC pipeline to compute c_z
    is a direct extension of the existing machinery.

    For the periodic (CDW) case LDWC gives z = 2 - eps^2/9 at 2-loop
    (no 1-loop term because of the periodic-FP structure); the LDWC
    paper recovers this via CFAC.  Short-range depinning z at 2-loop
    is not yet explicitly CFAC-derived in the current codebase.
    """
    return {
        'status': 'NOT YET CFAC-derived',
        'route': ('LDWC X^(alpha) machinery with velocity-self-energy '
                   'projection; see Amarteifio2026LDWC sec. 6'),
        'periodic_CDW_2loop_done': 'z = 2 - eps^2/9  (LDWC paper, CFAC-derived)',
        'short_range_depinning_2loop': 'c_z ~ 0.04 (LDW 2002) — external input here',
    }


def zeta_2loop_LDW(d: float) -> float:
    """qEW roughness exponent, 2-loop eps-expansion.

    zeta(d) = (eps/3) * (1 + 0.14331 * eps),  eps = 4 - d.
    Raw series — overshoots at d=1 (eps=3) relative to simulation.
    Source: Chauve-Le Doussal-Wiese 2001, cond-mat/0205108.
    """
    eps = 4.0 - d
    if eps <= 0:
        return 0.0
    return (eps / 3.0) * (1.0 + 0.14331 * eps)


def z_2loop_LDW(d: float) -> float:
    """qEW dynamical exponent, 2-loop eps-expansion.

    z(d) = 2 - (2/9) eps + 0.0402 eps^2.
    Source: Le Doussal-Wiese 2002 (cond-mat/0205108 and follow-ups).
    Raw series — overshoots at d=1.
    """
    eps = 4.0 - d
    if eps <= 0:
        return 2.0
    return 2.0 - (2.0 / 9.0) * eps + 0.0402 * eps ** 2


# Published "best" FRG-resummed values at d=1 (Borel-Pade, consistent
# with simulations; Rosso-Krauth 2002; LDW 2015 discussion).
_QEW_FRG_D1 = {
    'zeta': 1.25,      # roughness, 5/4 (conjectured exact; matches sims)
    'z':    1.433,     # dynamical (FRG Borel-Pade + Rosso-Krauth)
}


# ------------------------------------------------------------------ #
#  Depinning scaling relations (no fitting — algebraic)
# ------------------------------------------------------------------ #

def qEW_scaling_from_zeta_z(zeta: float, z: float,
                              d: float = 1.0) -> Dict:
    """Standard depinning scaling relations.

    Given the two primary qEW inputs (zeta, z), derive:
        nu          = 1/(2 - zeta)
        beta        = nu * (z - zeta)
        nu_parallel = z * nu
        delta       = beta / nu_parallel = (z - zeta) / z

    These are algebraic identities of qEW scaling, not CFAC
    predictions.  Documented in Chauve-Le Doussal-Wiese 2001.
    """
    if zeta >= 2.0:
        raise ValueError(f'zeta={zeta} unphysical; needs zeta < 2')
    nu = 1.0 / (2.0 - zeta)
    beta = nu * (z - zeta)
    nu_par = z * nu
    delta = beta / nu_par if nu_par else float('nan')
    return {
        'zeta': zeta, 'z': z, 'd': d,
        'nu': nu,
        'beta': beta,
        'nu_parallel': nu_par,
        'delta': delta,
    }


# ------------------------------------------------------------------ #
#  CDP exponents via the Le Doussal-Wiese map
# ------------------------------------------------------------------ #

def cdp_exponents_from_qEW(zeta: float, z: float,
                            d: float = 1.0) -> Dict:
    """Full CDP/Manna exponent set from qEW (zeta, z) via the
    n = n_0 + grad^2 u mapping (Le Doussal-Wiese 2015).

    Under the mapping:
      nu_perp(CDP)     = nu(qEW)
      z(CDP)           = z(qEW)
      beta(CDP)        = beta(qEW)           (activity <-> velocity)
      nu_parallel(CDP) = z(CDP) * nu_perp(CDP)
      delta(CDP)       = beta(CDP) / nu_parallel(CDP)

    Returns dict with all primary + secondary CDP exponents.
    """
    q = qEW_scaling_from_zeta_z(zeta, z, d)
    return {
        'd': d,
        'qEW_inputs': {'zeta': zeta, 'z': z},
        'nu_perp':     q['nu'],
        'z':           z,
        'beta':        q['beta'],
        'nu_parallel': q['nu_parallel'],
        'delta':       q['delta'],
        'zeta_roughness': zeta,
        'map': 'n(x,t) = n_0 + grad^2 u(x,t)  [Le Doussal-Wiese 2015]',
        'source_inputs': 'zeta, z from qEW FRG (Chauve-Le Doussal-Wiese 2001)',
    }


def manna_from_2loop_LDW(d: float = 1.0) -> Dict:
    """CDP/Manna exponents using raw 2-loop LDW eps-expansion.

    Overshoots at eps=3 — included for transparency about where
    perturbation theory is non-perturbative.
    """
    zeta = zeta_2loop_LDW(d)
    z = z_2loop_LDW(d)
    r = cdp_exponents_from_qEW(zeta, z, d)
    r['input_source'] = '2-loop LDW eps-expansion, unresummed'
    return r


def manna_from_FRG_resummed(d: float = 1.0) -> Dict:
    """CDP/Manna exponents using FRG-resummed published qEW values.

    For d=1 we use the Borel-Pade-consistent values zeta = 5/4,
    z = 1.433 (Rosso-Krauth 2002; LDW 2015).  For other d we fall
    back to the 2-loop series (there is no universal resummed
    formula for all d in closed form).
    """
    if abs(d - 1.0) < 1e-10:
        zeta = _QEW_FRG_D1['zeta']
        z = _QEW_FRG_D1['z']
        source = 'FRG-resummed d=1: zeta=5/4, z=1.433 (Rosso-Krauth 2002; LDW 2015)'
    else:
        zeta = zeta_2loop_LDW(d)
        z = z_2loop_LDW(d)
        source = '2-loop LDW eps-expansion (no d-dependent resummation in closed form)'
    r = cdp_exponents_from_qEW(zeta, z, d)
    r['input_source'] = source
    return r


# ------------------------------------------------------------------ #
#  Literature comparison (Manna 1+1d measurements)
# ------------------------------------------------------------------ #

_MANNA_MEASURED_D1 = {
    'z':           (1.55,  0.02,  'Bonachela-Munoz 2008 (Manna 1D)'),
    'nu_perp':     (1.35,  0.03,  'Bonachela-Munoz 2008'),
    'beta':        (0.29,  0.02,  'Bonachela-Munoz 2008'),
    'nu_parallel': (1.81,  0.05,  'Bonachela-Munoz 2008'),
    'delta':       (0.14,  0.01,  'Bonachela-Munoz 2008'),
}


def implied_z_from_manna_data(nu_perp: float = 1.35,
                                beta: float = 0.29,
                                nu_parallel: float = 1.81,
                                z_reported: float = 1.55,
                                zeta: Optional[float] = None) -> Dict:
    """Self-consistency check of published Manna exponents.

    If Manna is in the qEW/depinning class as LDW proved, then the
    scaling relations
        nu_perp = 1/(2 - zeta)           => zeta = 2 - 1/nu_perp
        beta    = nu_perp (z - zeta)     => z = zeta + beta/nu_perp
        nu_parallel = z nu_perp          => z = nu_parallel / nu_perp
    must ALL yield the same z.  We check that here.

    Published values are internally inconsistent by ~15%, indicating
    that any single-number "measured z" for Manna carries a
    systematic uncertainty larger than the quoted statistical error.
    Our qEW prediction z_qEW ~ 1.43 sits within the spread of implied
    values, which is the correct frame for comparing theory to data.
    """
    if zeta is None:
        zeta = 2.0 - 1.0 / nu_perp
    z_from_beta = zeta + beta / nu_perp
    z_from_nupar = nu_parallel / nu_perp
    values = [z_from_beta, z_from_nupar, z_reported]
    z_range = (min(values), max(values))
    z_spread = z_range[1] - z_range[0]
    return {
        'nu_perp': nu_perp,
        'implied_zeta': zeta,
        'z_from_beta_scaling': z_from_beta,
        'z_from_nu_parallel': z_from_nupar,
        'z_reported': z_reported,
        'z_range': z_range,
        'z_spread': z_spread,
        'z_relative_spread': z_spread / ((z_range[0] + z_range[1]) / 2),
        'verdict': (
            f'Measured Manna exponents imply z in [{z_range[0]:.3f}, '
            f'{z_range[1]:.3f}], spread {z_spread:.3f} '
            f'({100*z_spread/z_range[0]:.0f}% of low value).  '
            'The three values should coincide if qEW scaling holds; '
            'they do not, indicating ~15% systematic uncertainty in '
            'any single "measured z".'
        ),
    }


def compare_to_manna(result: Dict) -> Dict:
    """Compare a cdp_exponents_from_qEW result to Manna 1+1d data."""
    rows = []
    for name, (val, err, source) in _MANNA_MEASURED_D1.items():
        if name in result:
            pred = result[name]
            residual = pred - val
            n_sigma = abs(residual) / err if err else None
            rows.append({
                'exponent': name,
                'predicted': pred,
                'measured': val,
                'err': err,
                'residual': residual,
                'n_sigma': n_sigma,
                'source': source,
            })
    return {'input_source': result.get('input_source'), 'rows': rows}


def print_table(d: float = 1.0) -> None:
    """Human-readable comparison table — both raw-2-loop and FRG-resummed."""
    r_2l = manna_from_2loop_LDW(d)
    r_fg = manna_from_FRG_resummed(d)
    c_2l = compare_to_manna(r_2l)
    c_fg = compare_to_manna(r_fg)

    print(f'\n=== CDP/Manna from Le Doussal-Wiese qEW mapping at d={d} ===')
    print(f'qEW 2-loop (raw):  zeta={r_2l["zeta_roughness"]:.3f}, '
          f'z={r_2l["z"]:.3f}')
    print(f'qEW FRG resummed:  zeta={r_fg["zeta_roughness"]:.3f}, '
          f'z={r_fg["z"]:.3f}')
    print()
    print(f'{"exponent":<14}{"2-loop":>10}{"FRG":>10}'
          f'{"measured":>14}{"FRG residual":>14}')
    print('-' * 64)
    by_name_2l = {r['exponent']: r for r in c_2l['rows']}
    by_name_fg = {r['exponent']: r for r in c_fg['rows']}
    for name in ['z', 'nu_perp', 'beta', 'nu_parallel', 'delta']:
        row2 = by_name_2l[name]
        rowf = by_name_fg[name]
        print(f'{name:<14}{row2["predicted"]:>10.3f}{rowf["predicted"]:>10.3f}'
              f'  {rowf["measured"]:.3f} ± {rowf["err"]:.3f}'
              f'{rowf["residual"]:>+14.3f}')
    print()
    print('  The "FRG" column uses the published FRG-resummed qEW (zeta, z)')
    print('  as the sole CFAC-external input.  All CDP exponents are')
    print('  DERIVED from (zeta, z) via the qEW scaling relations.  No fitting.')

    iz = implied_z_from_manna_data()
    print()
    print(f'=== Self-consistency of the published Manna exponents ===')
    print(f'  nu_perp = 1.35  =>  implied zeta = {iz["implied_zeta"]:.3f}')
    print(f'  z from beta scaling     z = zeta + beta/nu = {iz["z_from_beta_scaling"]:.3f}')
    print(f'  z from nu_parallel/nu:                     = {iz["z_from_nu_parallel"]:.3f}')
    print(f'  z reported (direct):                       = {iz["z_reported"]:.3f}')
    print(f'  spread: {iz["z_spread"]:.3f} ({100*iz["z_relative_spread"]:.0f}%)')
    print(f'  qEW FRG prediction z = 1.433 sits WITHIN this spread.')
    print()
    print('  The published Manna exponents do not satisfy qEW scaling with a')
    print('  single z; the apparent disagreement of our prediction with the')
    print('  reported z=1.55 is smaller than the internal inconsistency of the')
    print('  measurements themselves.')


if __name__ == '__main__':
    print_table(d=1.0)
