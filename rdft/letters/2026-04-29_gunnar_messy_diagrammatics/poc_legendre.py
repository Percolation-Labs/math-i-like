"""POC: Legendre transform W -> Gamma for 0-d Reggeon DP.

Thin wrapper around ``rdft.crn.legendre.legendre_reggeon_dp``. Reproduces the
3+2 split for 2-loop vertex graphs (3 1PI + 2 reducible) at g^5 by reading
coefficients of W and Gamma directly from the CRN library.

Run: python3 poc_legendre.py
"""
import sympy as sp
from rdft.crn.legendre import legendre_reggeon_dp


def main(N_g: int = 5, J_max: int = 4):
    print("=" * 72)
    print("  Reggeon DP Legendre transform via rdft.crn")
    print("=" * 72)
    r = legendre_reggeon_dp(N_g=N_g, J_max=J_max)

    print()
    print("Selected W coefficients:")
    for n in (1, 2, 3):
        coef = sp.Poly(r.W, r.coupling).coeff_monomial(r.coupling**n)
        print(f"  [g^{n}] W = {sp.expand(coef)}")

    print()
    print("Selected Gamma coefficients:")
    for n in (0, 1, 3, 5):
        coef = sp.Poly(r.Gamma, r.coupling).coeff_monomial(r.coupling**n)
        print(f"  [g^{n}] Gamma = {sp.expand(coef)}")

    print()
    print("2-loop vertex sector (g^5, 3 external legs):")
    W_3pt = r.W_coef(2, 1, 5)
    Gamma_3pt = r.Gamma_coef(2, 1, 5)
    print(f"  [g^5 J^2 Jt]    W     = {W_3pt}     (sum over 5 connected, 3 1PI + 2 reducible)")
    print(f"  [g^5 Phi^2 Phit] Gamma = {Gamma_3pt}     (sum over 3 1PI only)")
    print(f"  W - Gamma                = {W_3pt - Gamma_3pt}     (reducibles cancelled by J*Phi subtraction)")

    assert W_3pt == -7608, f"W coef changed: {W_3pt}"
    assert Gamma_3pt == -504, f"Gamma coef changed: {Gamma_3pt}"
    print()
    print("All numerical checks PASSED (matches gribov.tex §3.3).")


if __name__ == "__main__":
    main()
