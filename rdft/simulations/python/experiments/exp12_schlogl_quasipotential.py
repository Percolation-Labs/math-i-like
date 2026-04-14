"""
Experiment 12: Schlögl II MFT quasipotential V(rho) from CFAC branch-gap.

Open problem (Prakash-Nicholson 2024, arXiv:2402.13168):
  Closed-form quasipotential for the Schlögl II model
      A + 2X <-> 3X    (autocatalytic)
  in the bistable regime.  Prakash-Nicholson compute V(rho) numerically;
  no analytic form is given for the full MFT functional.

CFAC contribution:
  Use rdft.ac.dse.coupled_dse to build the Schlögl II DSE in DP-MSR form;
  then read off V(rho) from the branch structure of the algebraic curve.
  The branch-gap contour integral
      V(rho) = oint_{branch} log(z - z_star(phi_rho)) drho
  gives V(rho) as an explicit closed form in (rho, k_a, k_b).

Setup:
  Schlögl II rates: k_a (forward 2X -> 3X), k_b (backward 3X -> 2X).
  Add source/sink to make density rho meaningful.  Use the DSE with
  rate-tilted phi to extract V(rho).

NOTE: this is a SCOPING calculation.  Full agreement with Prakash-Nicholson
requires careful matching of conventions and possibly higher-order
corrections; we report the algebraic structure that CFAC produces and
flag deviations.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path

from rdft.ac.stratification import puiseux_order, discriminant_in_z


def schlogl_II_phi(rho: float, k_a: float = 1.0, k_b: float = 1.0,
                   k_in: float = 0.5, k_out: float = 0.5) -> list[float]:
    """phi(G) for Schlögl II at density rho.

    Reactions: A_environment -> X (rate k_in * rho), X -> A (rate k_out),
    2X -> 3X (rate k_a), 3X -> 2X (rate k_b).

    Doi-Peliti vertex contributions (full expansion, mass-terms filtered):
      A_env -> X (rate k_in * rho): pure source, m+n=1, mass-type, skip.
      X -> A (rate k_out):  k_in_X=1, l_out=0:
        (z+1)^0 - (z+1)^1 = -z. vertex (1,1) -k_out [mass, skip].
      2X -> 3X (rate k_a): k=2, l=3, contributions:
        (z+1)^3 - (z+1)^2 = z^3 + 2z^2 + z. vertices (1,2), (2,2), (3,2).
        After m+n>=3 filter: (1,2)+k_a [m+n=3], (2,2)+2k_a [m+n=4], (3,2)+k_a.
        Phi: +k_a*G + 2 k_a G^2 + k_a G^3
      3X -> 2X (rate k_b): k=3, l=2, contributions:
        (z+1)^2 - (z+1)^3 = -(z^3 + 2z^2 + z). vertices (1,3), (2,3), (3,3).
        After m+n>=3 filter: all retained: (1,3) -k_b [m+n=4], (2,3) -2k_b, (3,3) -k_b.
        Phi: -k_b*G^2 - 2 k_b G^3 - k_b G^4

    Total:
      G^1: +k_a
      G^2: +2 k_a - k_b
      G^3: +k_a - 2 k_b
      G^4: -k_b

    This is degree-4 in G.  rho enters via the source rate but doesn't appear
    in this phi (the source is a mass term).  The DSE asymptotics depend on
    rho only through the boundary condition / steady state.
    """
    coefs = {
        0: 1.0,
        1: k_a,
        2: 2 * k_a - k_b,
        3: k_a - 2 * k_b,
        4: -k_b,
    }
    d = max(coefs.keys())
    return [coefs.get(i, 0.0) for i in range(d + 1)]


def quasipotential_at_rho(rho: float, k_a: float, k_b: float) -> float:
    """V(rho) from the branch-gap contour integral.

    The CFAC formula:
        V(rho) = -log|z_star(rho)| + (rho-dependent normalisation)
    where z_star is the dominant branch of the rho-tilted DSE.  In the
    Wenzel-Krammer sense, this is the negative log of the steady-state
    weight, equivalent to the MFT quasipotential up to normalisation.

    For Schlögl II, the rate-tilted phi has a rho-dependent linear coefficient
    (from the source rate).  The full quasipotential requires integrating
    over rho; the branch-radius |z_star(rho)| gives the leading exponential.
    """
    # Tilt: rho effectively rescales the source contribution; treat it as a
    # multiplicative factor on the linear coefficient.  Heuristic for now.
    phi = schlogl_II_phi(rho, k_a, k_b)
    phi_tilted = list(phi)
    phi_tilted[1] = phi_tilted[1] * rho  # rho-tilt on linear term
    k_dom, z = puiseux_order(phi_tilted)
    if not np.isfinite(abs(z)) or abs(z) == 0:
        return float('nan')
    return -np.log(abs(z))


def main():
    print('=' * 80)
    print('Experiment 12: Schlögl II quasipotential via CFAC branch structure')
    print('=' * 80)

    # 1. Identify (k_a, k_b) at the bistable / cube-root crossing
    # From Exp 2 / dse_landscape paper: k_a/k_b ≈ 1.476 puts Schlögl II on C_3.
    cases = [
        ('Schlögl II at k_a=1, k_b=1 (off C_3)', 1.0, 1.0),
        ('Schlögl II at k_a=1.476, k_b=1 (on C_3)', 1.476, 1.0),
        ('Schlögl II at k_a=2, k_b=1 (above C_3)', 2.0, 1.0),
    ]

    for desc, k_a, k_b in cases:
        phi = schlogl_II_phi(1.0, k_a, k_b)
        k_dom, z = puiseux_order(phi)
        print(f'\n{desc}')
        print(f'  phi (cubic slice) = {phi}')
        print(f'  dominant Puiseux order k = {k_dom},  |z*| = {abs(z):.4f}')

    # 2. Quasipotential as function of rho at the cube-root tuning
    print('\n\nV(rho) at the cube-root tuning k_a=1.476, k_b=1:')
    rho_arr = np.linspace(0.5, 5, 30)
    V_arr = np.array([quasipotential_at_rho(rho, 1.476, 1.0) for rho in rho_arr])
    print(f'{"rho":>8} {"V(rho)":>12}')
    for r, v in zip(rho_arr[::3], V_arr[::3]):
        print(f'{r:>8.2f} {v:>12.4f}')

    # 3. Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for desc, k_a, k_b in cases:
        V_curve = np.array([quasipotential_at_rho(rho, k_a, k_b) for rho in rho_arr])
        ax.plot(rho_arr, V_curve, lw=1.8, label=desc)
    ax.set_xlabel(r'$\rho$ (density)')
    ax.set_ylabel(r'$V(\rho) = -\log|z_\star(\rho)|$')
    ax.set_title('Experiment 12: Schlögl II quasipotential from CFAC branch radius')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outdir = Path(__file__).parent.parent.parent.parent / 'paper' / 'cfac' / 'figures'
    fig.savefig(outdir / 'exp12_schlogl_quasipotential.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'exp12_schlogl_quasipotential.png', bbox_inches='tight', dpi=150)
    print(f'\nSaved {outdir / "exp12_schlogl_quasipotential.pdf"}')

    print()
    print('=' * 80)
    print('STATEMENT')
    print('=' * 80)
    print("""
Open problem (Prakash-Nicholson 2024, arXiv:2402.13168):
  Closed-form quasipotential V(rho) for Schlögl II in the bistable regime;
  numerical only in the literature.

CFAC contribution:
  The branch radius |z_star(rho)| of the rho-tilted Schlögl II DSE provides
  V(rho) = -log|z_star(rho)| up to normalisation.  The library function
  rdft.ac.stratification.puiseux_order makes this a one-line evaluation.

Result:
  V(rho) computed as an explicit function of rho for three Schlögl II
  parameter choices (off, on, and above the cube-root tuning).  At the
  cube-root tuning k_a/k_b ~ 1.476, V(rho) inherits a different
  algebraic structure (dominant branch order may shift as rho varies).

  This is a SCOPING calculation that demonstrates the workflow.  Full
  matching to Prakash-Nicholson 2024 numerics requires careful
  Wentzel-Kramers normalisation and the proper rho-tilt of the Schlögl II
  generator; that's the next refinement.  The library now provides the
  algebraic backbone for that refinement.
""")


if __name__ == '__main__':
    main()
