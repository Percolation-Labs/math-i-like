"""Verify the 2x2-matrix reduction AND the structural eigenvalue prediction.

Reproduces the boxed claims in note.tex, R2:

Structural prediction (no matrix arithmetic, just the four-term coupling):
  Input 1 (sector dim):  #{(m1,m2): m1+m2=1, m1,m2>=0} = 2.
  Input 2 (trace):       T_diag(m1,m2) = m1 + m2 = N (Term 2 is rail-1 number op).
                         => Tr(T) on N=1 sector = 1 * 2 = 2.
  Input 3 (det):         V annihilates |(0,1)> (no rail-1 particles to remove).
                         => rank(V|_B) = 1, rank(T) = 1, det(T) = 0.
  Conclusion:            eigvals(T) = {0, Tr(T)} = {0, 2}.

Cross-checks computed explicitly:
  - V on B = {|(1,0)>, |(0,1)>}: [[1,0],[-1,0]]
  - V_tilde (rail-flip) on B:    [[0,-1],[0, 1]]
  - T = V + V_tilde:             [[1,-1],[-1, 1]],  T^2 = 2T
  - chain element <(1,0)| T^n V |(1,0)> = 2^n for n = 0..5
  - brute-force sum over occupations agrees through n = 5
"""
import numpy as np


def V(m1, m2, n1, n2):
    """Gunnar's four-term Kronecker coupling."""
    if min(m1, m2, n1, n2) < 0:
        return 0
    out = 0
    if m2 == n2 and m1 - 1 == n1 + 1:
        out += 1
    if m2 == n2 and m1 - 1 == n1 - 1:
        out += n1
    if m2 == n2 + 1 and m1 - 1 == n1 + 1:
        out -= 1
    if m2 == n2 - 1 and m1 - 1 == n1:
        out -= n2
    return out


def main():
    basis = [(1, 0), (0, 1)]

    # ---- Structural prediction (no matrix needed) ----
    print("=== Structural prediction ===")
    dim_pred = len(basis)
    print(f"Input 1 (boundary picks N=1 sector): dim = {dim_pred}")
    # Term 2 is the only diagonal-preserving term; weight n1 = m1 on the diagonal.
    # So T_diag(m1,m2) = m1 + m2 = N. Trace = sum over basis = N * dim.
    trace_pred = sum(m1 + m2 for (m1, m2) in basis)
    print(f"Input 2 (T2 is rail-1 number op): Tr(T) = sum(m1+m2) = {trace_pred}")
    # V annihilates |(0,1)> (no rail-1 particle to remove): rank(V) = 1.
    # rank(T) = 1, so det(T) = 0.
    det_pred = 0
    print(f"Input 3 (V annihilates |(0,1)>): rank(T) = 1, det(T) = {det_pred}")
    eigvals_pred = (0, trace_pred)
    print(f"=> Predicted eigenvalues of T: {eigvals_pred}")

    # ---- Cross-check by explicit construction ----
    print("\n=== Explicit cross-check ===")
    Vmat = np.array(
        [[V(m1, m2, n1, n2) for (m1, m2) in basis] for (n1, n2) in basis],
        dtype=int,
    )
    Vtilde = np.array(
        [[V(m2, m1, n2, n1) for (m1, m2) in basis] for (n1, n2) in basis],
        dtype=int,
    )
    T = Vmat + Vtilde

    print("V on B =")
    print(Vmat)
    print("V_tilde on B =")
    print(Vtilde)
    print("T = V + V_tilde =")
    print(T)
    print(f"Tr(T) = {np.trace(T)}, det(T) = {round(np.linalg.det(T))}")
    print("eigvals(T) =", sorted(np.linalg.eigvals(T).real))
    assert np.array_equal(T @ T, 2 * T), "T^2 = 2T failed"
    assert np.trace(T) == trace_pred, "trace prediction failed"
    assert round(np.linalg.det(T)) == det_pred, "det prediction failed"

    # Chain element <(1,0)| T^n V |(1,0)> -- column convention.
    e1 = np.array([1, 0])
    print("\nMatrix expression <(1,0)| T^n V |(1,0)>:")
    for n in range(6):
        val = e1 @ np.linalg.matrix_power(T, n) @ Vmat @ e1
        assert val == 2**n, f"matrix expression failed at n={n}"
        print(f"  n={n}: {val}  (= 2^{n} = {2**n})")

    # Brute-force sum over internal occupations -- n+1 H-vertices, n brackets.
    # Boundary fixed at (1,0) on both sides. Truncate the sum at L = 4 by Step 1
    # (the conservation argument): higher occupations cannot contribute.
    L = 5
    print("\nBrute-force sum over internal occupations (truncated at L=5):")
    for n in range(1, 6):
        # internal indices ell_1, ..., ell_{2n}
        total = 0
        for indices in np.ndindex(*([L] * (2 * n))):
            ell = list(indices)
            # leftmost vertex: V(1, 0; ell_1, ell_2)
            a = V(1, 0, ell[0], ell[1])
            if a == 0:
                continue
            # middle brackets and rightmost bracket
            prod = a
            for k in range(1, n):  # n-1 middle brackets
                # H(ell_{2k-1}, ell_{2k+1}; ell_{2k}, ell_{2k+2}) + rail-flip
                m1, n1, m2, n2 = ell[2*k-2], ell[2*k], ell[2*k-1], ell[2*k+1]
                bracket = V(m1, m2, n1, n2) + V(m2, m1, n2, n1)
                prod *= bracket
                if prod == 0:
                    break
            if prod == 0:
                continue
            # rightmost bracket: H(ell_{2n-1}, 1; ell_{2n}, 0) + H(ell_{2n}, 0; ell_{2n-1}, 1)
            m1, m2 = ell[2*n-2], ell[2*n-1]
            right = V(m1, m2, 1, 0) + V(m2, m1, 0, 1)
            total += prod * right
        assert total == 2**n, f"brute force failed at n={n}: {total} vs 2^{n}={2**n}"
        print(f"  n={n}: sum = {total}  (= 2^{n} = {2**n})")

    # ---- Polynomial factorisation: V_hat collapses to (z1-z2)(w1-w2) ----
    print("\n=== V_hat polynomial on the linear sector (m1+m2 = n1+n2 = 1) ===")
    try:
        from sympy import symbols, expand, factor, collect
        z1, z2, w1, w2 = symbols('z1 z2 w1 w2')
        Vhat = 0
        for (m1, m2) in basis:
            for (n1, n2) in basis:
                v = V(m1, m2, n1, n2)
                z = z1 if m1 == 1 else z2
                w = w1 if n1 == 1 else w2
                Vhat += v * z * w
        print(f"V_hat|_(1,1) = {Vhat}")
        # Rail-flip: swap z1<->z2 and w1<->w2
        sub = {z1: z2, z2: z1, w1: w2, w2: w1}
        Vhat_flip = Vhat.subs(sub, simultaneous=True)
        print(f"V_tilde_hat|_(1,1) = {Vhat_flip}")
        That = expand(Vhat + Vhat_flip)
        print(f"T_hat = {That}")
        print(f"T_hat factored = {factor(That)}")
        # Antisymmetric input: substitute z1=1, z2=-1
        T_anti = expand(That.subs({z1: 1, z2: -1}))
        print(f"T_hat(z1=1, z2=-1) = {T_anti}  -->  eigenvalue = 2 on (w1-w2)")
        # Symmetric input: substitute z1=1, z2=1 -> kernel
        T_sym = expand(That.subs({z1: 1, z2: 1}))
        print(f"T_hat(z1=1, z2=1)  = {T_sym}   -->  symmetric mode in kernel")
    except ImportError:
        print("(sympy not available; skipping polynomial factorisation check)")

    print("\nAll checks pass. The 2x2 reduction is exact and structural.")


if __name__ == "__main__":
    main()
