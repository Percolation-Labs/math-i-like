"""
Exact MFPT via direct tridiagonal solve of the backward equation.

For the birth-death chain with absorbing at n_* and reflecting at n=V,
the MFPT h(n) from state n to reach n_* satisfies:

  -1 = W_+(n)[h(n+1) - h(n)] + W_-(n)[h(n-1) - h(n)]    for n > n_*

with h(n_*) = 0  and  h(V+1) ≡ h(V)  (reflecting at top).

Rearranged:  W_-(n) h(n-1) - (W_+(n)+W_-(n)) h(n) + W_+(n) h(n+1) = -1

We solve this linear system directly.
"""
import numpy as np
from math import sqrt, pi, log


def mfpt_exact(V, g, alpha=1.0, f_thresh=0.1, f_start=None):
    n_star = int(round(V * f_thresh))
    if n_star < 1: n_star = 1

    disc = sqrt(1 - 4/(alpha*g))
    f_ON = 0.5*(1 + disc)
    if f_start is None:
        n_start = int(round(V * f_ON))
    else:
        n_start = int(round(V * f_start))

    # States n = n_star+1, ..., V.  N unknowns.
    N = V - n_star
    if N <= 0:
        return 0
    A = np.zeros((N, N))
    b = -np.ones(N)
    for i in range(N):
        n = n_star + 1 + i
        f = n / V
        Wp = alpha*g*f*f*(V - n)
        Wm = n * 1.0
        if i > 0:
            A[i, i-1] = Wm
        A[i, i] = -(Wp + Wm)
        if i < N - 1:
            A[i, i+1] = Wp
        elif n == V:
            # Reflecting at V: h(V+1) = h(V), so equation at n=V becomes
            # Wm * h(V-1) - Wm * h(V) = -1
            A[i, i] = -Wm  # override
    # h(n_*) = 0 is boundary (not in the system).
    # At i=0, the equation has a term W_-(n_*+1) * h(n_*) = 0 (dropped).

    h = np.linalg.solve(A, b)
    idx_start = n_start - (n_star + 1)
    if idx_start < 0:
        return 0.0
    if idx_start >= len(h):
        idx_start = len(h) - 1
    return h[idx_start]


def cfac_params(g, alpha=1.0):
    disc = sqrt(1 - 4/(alpha*g))
    f_s = 0.5*(1 + disc)
    f_u = 0.5*(1 - disc)
    Op_s = alpha*g*(2*f_s - 3*f_s*f_s) - 1
    Op_u = alpha*g*(2*f_u - 3*f_u*f_u) - 1
    W_s = f_s; W_u = f_u
    Psi_s = 1/(1-f_s) - 1/f_s
    Psi_u = 1/(1-f_u) - 1/f_u

    fs_grid = np.linspace(f_u + 1e-9, f_s - 1e-9, 3000)
    s = -np.log(alpha*g * fs_grid * (1 - fs_grid))
    S_star = abs(np.trapezoid(s, fs_grid))
    return dict(f_s=f_s, f_u=f_u, Op_s=Op_s, Op_u=Op_u,
                W_s=W_s, W_u=W_u, Psi_s=Psi_s, Psi_u=Psi_u, S=S_star)


def main():
    print("="*72)
    print("EXACT MFPT vs CANDIDATE PREFACTORS")
    print("="*72)
    print("\nTrue A should be nearly V-independent (asymptotic).")

    for g in [4.5, 5.0, 6.0]:
        p = cfac_params(g)
        print(f"\n--- g = {g}, S* = {p['S']:.5f} ---")
        A_langevin = 2*pi/sqrt(abs(p['Op_s']*p['Op_u']))
        A_Phi      = 2*pi/sqrt(abs(p['Psi_s']*p['Psi_u']))
        A_pi2      = pi**2/sqrt(abs(p['Psi_s']*p['Psi_u']))
        print(f"  Candidates: A_Lang = {A_langevin:.3f}, "
              f"A_Φ = {A_Phi:.3f}, A_π² = {A_pi2:.3f}")
        print(f"  {'V':>4s} {'τ_exact':>12s} {'A_extr':>8s}")
        for V in [20, 30, 50, 80, 120, 180]:
            tau = mfpt_exact(V, g)
            if tau > 1e-10 and np.isfinite(tau):
                A = tau / np.exp(V * p['S'])
                print(f"  {V:>4d} {tau:>12.4e} {A:>8.4f}")


if __name__ == '__main__':
    main()
