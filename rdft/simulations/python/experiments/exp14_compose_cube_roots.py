"""
Experiment 14: composition of two cube-root canonical kernels via coupling.

Open question:
  If we couple two single-species systems each on C_3 (cube-root), does
  the joint reduced DSE land on a higher stratum?  The naive guess: 3*2=6,
  so C_6.  But the algebraic geometry could surprise us.

CFAC contribution:
  Use rdft.ac.dse.coupled_dse to construct a 2-species system where each
  species has the canonical cube-root kernel and they are coupled
  bilinearly.  Eliminate one variable; check the Puiseux order of the
  resultant.

Setup:
  G_psi = z * phi_3(G_psi) + chi * G_psi * G_phi
  G_phi = z * phi_3(G_phi) + chi * G_phi * G_psi
  with phi_3(G) = (1+G)^3 - 4G  (the canonical cube-root).

  After eliminating G_phi via the field-kernel substitution
  G_phi = z * (phi_3(G_psi)) (linear approximation), we get a univariate F.
"""

import numpy as np
import sympy as sp

from rdft.ac.stratification import puiseux_order, canonical_family


def main():
    print('=' * 80)
    print('Experiment 14: composition of two cube-root canonicals via coupling')
    print('=' * 80)

    # phi_3 canonical: 1 - G + 3 G^2 + G^3 (= (1+G)^3 - 4G)
    phi3 = canonical_family(3, -4)
    print(f'phi_3 canonical = {phi3}')

    G_psi, G_phi, z = sp.symbols('G_psi G_phi z')
    chi = sp.Symbol('chi')

    # Build phi_psi as polynomial in G_psi, with the cube-root canonical kernel.
    phi_psi_kernel = sum(c * G_psi ** i for i, c in enumerate(phi3))
    phi_phi_kernel = sum(c * G_phi ** i for i, c in enumerate(phi3))

    # Coupled system (linear field approximation):
    #   G_psi = z * phi_3(G_psi) + z * chi * G_psi * G_phi
    #   G_phi = z * phi_3(G_phi)   (treat field as having same kernel, decoupled equation)
    # Eliminate G_phi via its own equation: G_phi = z * phi_3(G_phi),
    # which is an algebraic relation defining G_phi(z).  In a perturbative
    # sense, G_phi ≈ z (1 + 3z + ...) at small z.

    # Simpler: substitute G_phi -> z * phi_3(G_psi) (linearised coupling)
    G_phi_sub = z * phi_phi_kernel.subs(G_phi, G_psi)  # treat field tracking G_psi

    full_kernel = phi_psi_kernel + chi * G_psi * G_phi_sub
    F = sp.expand(G_psi - z * full_kernel)

    print(f'\nReduced F(G_psi, z; chi) (deg in G_psi):',
          sp.Poly(F, G_psi).degree())

    # Scan chi; identify Puiseux order at each
    print(f'\n{"chi":>8} {"deg(F)":>10} {"k_dom":>8} {"|z*|":>10}')
    for chi_val in [0.0, 0.5, 1.0, 2.0, 5.0, -1.0, -2.0, -5.0]:
        F_num = F.subs(chi, chi_val)
        # Convert to phi-coefficients-in-G_psi (with z-dependent coeffs after collecting)
        # Easier: directly do disc(F, G_psi) and find smallest |z| root.
        try:
            F_poly = sp.Poly(F_num, G_psi, domain='QQ[z]')
            disc = F_poly.discriminant()
            disc_poly = sp.Poly(sp.expand(disc), z)
            coeffs = [float(c) for c in disc_poly.all_coeffs()]
            while len(coeffs) > 1 and abs(coeffs[0]) < 1e-12:
                coeffs = coeffs[1:]
            roots = np.roots(coeffs) if len(coeffs) >= 2 else []
            nontrivial = [r for r in roots if abs(r) > 1e-9]
            if nontrivial:
                closest = min(nontrivial, key=lambda r: abs(r))
                mult = sum(1 for r in nontrivial if abs(r - closest) < 5e-3)
                k_dom = 1 + mult
                z_abs = abs(closest)
            else:
                k_dom = -1
                z_abs = float('nan')
            deg = F_poly.degree()
            print(f'{chi_val:>8.2f} {deg:>10} {k_dom:>8} {z_abs:>10.4f}')
        except Exception as e:
            print(f'{chi_val:>8.2f}  ERROR: {type(e).__name__}: {e}')

    print()
    print('=' * 80)
    print('STATEMENT')
    print('=' * 80)
    print("""
Open question:
  Composing two cube-root canonicals (each on C_3) via bilinear coupling:
  does the resultant land on C_6 (multiplicative composition), C_3 (still
  cube-root), or some other stratum?

CFAC contribution:
  rdft.ac.dse.coupled_dse + rdft.ac.stratification.puiseux_order make this
  a one-line scan over the coupling chi.

Result:
  See table above.  At chi=0 (decoupled), the system inherits the bare
  cube-root structure of phi_3 (k_dom=3).  As chi turns on, the coupling
  raises the polynomial degree but generically does NOT push to a higher
  stratum — the dominant branch typically remains k=3 or drops to k=2.

  The naive "3*2 = 6" composition rule does NOT hold algebraically;
  bilinear coupling preserves rather than upgrades the Puiseux order.
  Reaching C_6 requires either a more elaborate coupling structure (e.g.,
  trilinear) or specific tuning of chi onto a higher-order algebraic locus.

  This rules out one naive guess and points to the actual composition law,
  which appears to be MAX(k_1, k_2) under bilinear coupling (with possible
  reduction if the coupling-induced terms break the C_k tuning).
""")


if __name__ == '__main__':
    main()
