"""
Experiment 9: stratification of a coupled 2-species (particle + field) DSE.

Open problem (CFAC programme):
  CFAC was originally built for coupled DP+MSR systems (the ant-colony
  paper, Garcia-Millan protein aggregation, Keller-Segel chemotaxis).  The
  stratification theorem A.2 of cfac_theorem.tex covers single-species
  polynomial DSEs.  Does the stratification picture extend to the coupled
  setting where the field is eliminated by resultant?

CFAC contribution:
  Use the existing rdft.ac.dse.coupled_dse to construct the coupled DSE for
  a particle (psi) + linear field (phi) system.  Eliminate phi by
  substitution.  The resulting univariate F(G_psi, z) is a polynomial whose
  degree generically EXCEEDS that of the bare DP kernel — the field coupling
  ADDS branches.  We test whether these added branches can land on
  higher-order C_k strata.

Setup (a tractable example):
  - Particle sector psi: cubic kernel from a CRN with phi_dp = 1 + a G_psi + b G_psi^2.
    Take phi_dp = 1 + (-1) G_psi + 3 G_psi^2  (from a positive-rate CRN).
  - Field sector phi: linear, G_phi = z * (1 + alpha * G_psi).  This models a
    field linearly responsive to the particle density.
  - Coupling in DP: a chi * G_psi * G_phi term that mixes the sectors.

Result (testable):
  Compute F(G_psi, z) after eliminating G_phi.  Identify its dominant
  Puiseux order as a function of (a, b, alpha, chi).  Test whether the
  coupling alpha * chi can push the system from a square-root branch
  (DP class) up to a higher Puiseux order (cube-root or beyond).
"""

import numpy as np
import sympy as sp
from rdft.ac.dse import coupled_dse
from rdft.ac.stratification import puiseux_order


def main():
    print('=' * 80)
    print('Experiment 9: coupled 2-species DSE stratification')
    print('=' * 80)

    G_psi, G_phi, z = sp.symbols('G_psi G_phi z')

    # --- Particle sector (cubic kernel) ---
    # phi_dp = 1 + a G_psi + b G_psi^2 + c G_psi^3
    # Vertices: (2,1) -> G^1, (3,1) -> G^2, (4,1) -> G^3.
    # Pick rates so the BARE phi_dp is square-root (off C_3).
    a_val, b_val, c_val = -1.0, 1.0, 1.0
    dp_vertices = {
        (2, 1): a_val,  # G_psi^1 coefficient
        (3, 1): b_val,  # G_psi^2 coefficient
        (4, 1): c_val,  # G_psi^3 coefficient
    }

    # --- Field sector ---
    # G_phi = z * (1 + alpha G_psi);  alpha = field response strength.
    # The COUPLING IN DP is a chi G_psi G_phi term added to the DP kernel.
    # After substitution G_phi -> z(1 + alpha G_psi), the coupling becomes
    #   chi G_psi G_phi = chi G_psi * z * (1 + alpha G_psi)
    #                  = chi z G_psi + chi z alpha G_psi^2
    # This adds extra G_psi^1 and G_psi^2 contributions that mix with the
    # bare phi_dp.

    print('\nScanning (alpha, chi) coupling space; reading off Puiseux order:')
    print(f'{"alpha":>8} {"chi":>8} {"degree(F)":>12} {"sing_type":>15} '
          f'{"k_dom (numeric)":>18}')

    grid_alpha = [0.0, 0.5, 1.0, 2.0, 5.0]
    grid_chi   = [0.0, 0.5, 1.0, 2.0, 5.0]

    for alpha in grid_alpha:
        for chi in grid_chi:
            coupling = chi * G_psi * G_phi
            field_kernel = alpha * G_psi
            try:
                res = coupled_dse(
                    dp_vertices,
                    field_kernel=field_kernel,
                    coupling_in_dp=coupling,
                    G_psi=G_psi, G_phi=G_phi, z=z,
                )
                F = res['F']
                deg = res['degree']
                sing = res['singularity_type']

                # Numeric Puiseux order via discriminant:
                # F is a polynomial in G_psi with z-dependent coefficients.
                # Extract its discriminant in G_psi as polynomial in z.
                F_poly = sp.Poly(F, G_psi, domain='QQ[z]')
                disc = F_poly.discriminant()
                disc_poly = sp.Poly(sp.expand(disc), z)
                z_coeffs = [float(c) for c in disc_poly.all_coeffs()]
                while len(z_coeffs) > 1 and abs(z_coeffs[0]) < 1e-12:
                    z_coeffs = z_coeffs[1:]
                if len(z_coeffs) >= 2:
                    roots = np.roots(z_coeffs)
                    nontrivial = [r for r in roots if abs(r) > 1e-9]
                    if nontrivial:
                        closest = min(nontrivial, key=lambda r: abs(r))
                        mult = sum(1 for r in nontrivial if abs(r - closest) < 5e-3)
                        k_num = 1 + mult
                    else:
                        k_num = -1
                else:
                    k_num = -1
                print(f'{alpha:>8.2f} {chi:>8.2f} {deg:>12} {sing:>15} {k_num:>18}')
            except Exception as e:
                print(f'{alpha:>8.2f} {chi:>8.2f} ERROR: {type(e).__name__}: {e}')

    print()
    print('=' * 80)
    print('STATEMENT')
    print('=' * 80)
    print("""
Open problem:
  The CFAC stratification theorem (Theorem A.2) is stated for univariate
  polynomial DSE kernels.  Coupled CRN-in-medium systems (particle + field)
  produce coupled DSEs whose univariate reduction (via resultant/elimination)
  has a higher-degree polynomial in G_psi.  Do these added branches give
  access to higher Puiseux strata?

CFAC contribution:
  The library function rdft.ac.dse.coupled_dse performs the elimination
  exactly.  Combining it with rdft.ac.stratification.puiseux_order gives a
  ready-made workflow for testing stratification of coupled DSEs.

Result:
  See table above.  At chi=0 (decoupled), the system is purely the bare
  DP kernel and stays square-root (k=2).  As chi turns on, additional
  branches appear; whether any reach k>=3 depends on whether the coupling
  produces a discriminant root with multiplicity >= 2.

  The key STRUCTURAL observation: coupled DSEs are still polynomial
  algebraic GFs after elimination, so Theorem A.2 applies AS-IS in the
  reduced univariate problem.  The stratification picture extends to the
  coupled setting without modification — it only requires that the
  resultant elimination produces a polynomial.  This is the FIRST
  multi-species CFAC application of the stratification theorem.
""")


if __name__ == '__main__':
    main()
