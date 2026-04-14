"""
Exact MFPT for 1D birth-death chain — first-principles verification
of the CFAC 1-loop prefactor formula.

For a birth-death chain on {0, 1, ..., V} with rates W±(n), the
EXACT MFPT from N_s to reach (below) threshold N_* is (Gardiner
/ van Kampen):

  τ(N_s → N_*) = Σ_{n=N_*}^{N_s-1}  (1/W_-(n+1)) · Σ_{m=N_*+1}^{n+1} π_m/π_{n+1}

where π_m is the stationary distribution (up to normalisation).
For absorbing at N_*, the formula simplifies.

We compute τ_exact(V) for V = 20..200, extract A(V) = τ/exp(VS*),
and compare to the candidate formulas:
  A_1 = 2π/√|Ω'(f_s)Ω'(f_u)|          (naive Langevin)
  A_2 = 2π/√|Φ''(f_s)Φ''(f_u)|         (quasi-potential)
  A_3 = π²/√|Φ''(f_s)Φ''(f_u)|         (empirical fit)
  A_Meerson = 2π/(|Ω'(f_s)||Ω'(f_u)|)^{1/2}·√(W_-(f_u)/W_+(f_s))
"""
import numpy as np
from math import sqrt, pi, log


def cfac_params(g, alpha=1.0):
    """For W_+ = α g f²(1-f), W_- = f.  Returns fixed points,
    derivatives, and candidate prefactors."""
    if alpha*g <= 4:
        return None
    disc = sqrt(1 - 4/(alpha*g))
    f_s = 0.5*(1 + disc)  # stable ON
    f_u = 0.5*(1 - disc)  # unstable middle
    # Ω(f) = αg f²(1-f) - f
    # Ω'(f) = αg(2f - 3f²) - 1
    Op_s = alpha*g*(2*f_s - 3*f_s*f_s) - 1  # < 0 for stable
    Op_u = alpha*g*(2*f_u - 3*f_u*f_u) - 1  # > 0 for unstable
    # W_+(f) = αg f²(1-f) = f at fixed points (W_+ = W_- = f)
    W_s = f_s; W_u = f_u  # since W_+ = W_- = f at fixed points
    # Φ''(f) = 1/(1-f) - 1/f at fixed points
    Psi_pp_s = 1/(1-f_s) - 1/f_s
    Psi_pp_u = 1/(1-f_u) - 1/f_u

    # Instanton action
    # S* = -∫_{f_s}^{f_u} ln(αg f(1-f)) df
    #    (= +integral in reverse direction)
    fs_grid = np.linspace(f_u + 1e-9, f_s - 1e-9, 3000)
    s = -np.log(alpha*g * fs_grid * (1 - fs_grid))
    S_star = abs(np.trapezoid(s, fs_grid))

    # Candidate prefactor formulas
    A1 = 2*pi / sqrt(abs(Op_s*Op_u))
    A2 = 2*pi / sqrt(abs(Psi_pp_s*Psi_pp_u))
    A3 = pi**2 / sqrt(abs(Psi_pp_s*Psi_pp_u))
    # With state-dependent D correction (Langevin with D):
    # D(f) = (W_+ + W_-)/(2) at FP = W_* (density form)
    A_Dcorr = 2*pi*sqrt(W_s*W_u)/sqrt(abs(Op_s*Op_u))
    # Meerson-style (scramble)
    A_meerson = 2*pi*sqrt(W_u/W_s)/sqrt(abs(Op_s*Op_u))

    return dict(f_s=f_s, f_u=f_u, Op_s=Op_s, Op_u=Op_u,
                W_s=W_s, W_u=W_u, Psi_s=Psi_pp_s, Psi_u=Psi_pp_u,
                S=S_star, A_langevin=A1, A_Phi=A2, A_pi2=A3,
                A_Dcorr=A_Dcorr, A_meerson=A_meerson)


def exact_mfpt(V, g, alpha=1.0, f_start=None, f_thresh=0.1):
    """Exact MFPT for the 1D birth-death chain from N_s down to N_*
    using the Gardiner 'backward equation' formula:

    τ(n) = τ(n-1) + (1/W_+(n-1) · Σ ...)  — messier form.

    Simpler: use recursive backward equation:
      τ(n) = 1/[W_+(n) + W_-(n)]  + [W_+(n) τ(n+1) + W_-(n) τ(n-1)] /
              [W_+(n)+W_-(n)]

    with τ(n_*) = 0 (absorbing at threshold).

    We set up the tridiagonal system and solve.
    """
    p = cfac_params(g, alpha)
    if f_start is None:
        f_start = p['f_s']

    # Discretise N from n_* to V
    n_star = int(round(V * f_thresh))
    n_start = int(round(V * f_start))

    # Define rates (numpy arrays)
    def W_plus(n):
        f = n / V
        return alpha*g*f*f*(V - n)
    def W_minus(n):
        return n * 1.0

    # Solve backward equation for τ(n), for n = n_star, ..., V
    # BC: τ(n_*) = 0
    # Reflecting at n = V (or open; since W_+(V) = 0, it's naturally reflecting)
    # For n > n_*:
    #   τ(n) = 1 + p+(n) τ(n+1) + p-(n) τ(n-1)
    # where p± = W_±/(W_++W_-)
    # Rearrange: W_+(n)[τ(n)-τ(n+1)] + W_-(n)[τ(n)-τ(n-1)] = 1
    # At n = V (boundary): W_+(V)=0 ⇒ W_-(V)[τ(V)-τ(V-1)] = 1 ⇒ τ(V) = τ(V-1) + 1/W_-(V)

    N = V - n_star + 1   # indices 0..V-n_star corresponding to n=n_star..V
    # Unknowns: τ(n) for n = n_star+1 .. V, so N-1 unknowns

    # Standard iterative method:
    # τ(n) - τ(n-1) = 1/W_-(n) · [1 + Σ_{k=n}^{V-1} W_+(k)W_+(k-1)...W_+(n)/[W_-(k+1)W_-(k)...W_-(n+1)] · ...]
    # Use recursion (Gardiner eq 5.2.148-ish):
    # Define φ(n) = Π_{j=n_star+1}^{n} W_+(j-1)/W_-(j).  Then
    # τ(n_start) = Σ_{n=n_star+1}^{n_start} 1/(W_-(n) φ(n)) · Σ_{m=n_star+1}^{n} φ(m)

    # Compute π(n) ∝ Π_{j=1}^{n} W_+(j-1)/W_-(j)  (normalised later)
    # Here we want Π from n_star+1
    log_phi = np.zeros(N)
    for idx in range(1, N):
        n = n_star + idx
        # Phi(n) = Phi(n-1) · W_+(n-1)/W_-(n)
        Wp = W_plus(n - 1)
        Wm = W_minus(n)
        if Wp <= 0 or Wm <= 0:
            log_phi[idx] = -np.inf
        else:
            log_phi[idx] = log_phi[idx-1] + log(Wp/Wm)

    # Now MFPT starting from n_start:
    # τ(n_start → n_star) = Σ_{n=n_star+1}^{n_start} [1/(W_-(n)·φ(n))] · Σ_{m=n_star+1}^{n} φ(m)
    # = Σ_n (1/W_-(n)) · e^{-log_phi(n)} · Σ_{m=n_star+1}^{n} e^{log_phi(m)}

    # MFPT from n_start to absorbing at n_star (downward escape).
    # The chain is symmetric under flipping W_+ ↔ W_-: we want mean time
    # to reach n_star when starting from n_start > n_star.
    #
    # Standard formula for 1D birth-death absorbing at n_star from above:
    #   τ(n_start) = Σ_{k=n_star+1}^{n_start} 1/(W_+(k)·π(k))·Σ_{m=k}^{V} π(m)
    # where π(n) = Π_{j=n_star+1}^{n} W_-(j)/W_+(j-1)  (reversed ratio
    # because we're going down-wards in a reflecting-at-top chain).
    #
    # Equivalent to van Kampen / Gardiner Eq. 5.2.158 generalised.

    # Recompute log_phi with the DOWNWARD convention:
    #   π(n) = Π_{j=n_star+1}^{n} W_-(j)/W_+(j-1)
    log_phi = np.zeros(N)
    for idx in range(1, N):
        n = n_star + idx
        Wp = W_plus(n - 1)
        Wm = W_minus(n)
        if Wp <= 0 or Wm <= 0:
            log_phi[idx] = -np.inf
        else:
            log_phi[idx] = log_phi[idx - 1] + log(Wm / Wp)

    # τ = Σ_{k=n_star+1}^{n_start} (1/(W_+(k) π(k))) · Σ_{m=k}^{V} π(m)
    # For efficiency, precompute partial sums of π from the top end.
    idx_start = n_start - n_star
    # π values on arrays for all idx 0..N-1; normalise log_phi so max is 0
    # to avoid overflow
    log_max = np.max(log_phi[np.isfinite(log_phi)])
    pi_arr = np.where(np.isfinite(log_phi),
                       np.exp(log_phi - log_max), 0)
    # Σ_{m=k}^{V} π(m):  cumulative from the top
    cum_top = np.zeros(N + 1)
    for idx in range(N - 1, -1, -1):
        cum_top[idx] = cum_top[idx + 1] + pi_arr[idx]
    # Now τ contribution:
    tau = 0.0
    for k_idx in range(1, idx_start + 1):
        k = n_star + k_idx
        Wp_k = W_plus(k)
        if Wp_k <= 0 or pi_arr[k_idx] <= 0:
            continue
        tau += (1.0 / Wp_k) * (1.0 / pi_arr[k_idx]) * cum_top[k_idx]
    return tau


def main():
    print("="*72)
    print("EXACT MFPT vs CANDIDATE PREFACTOR FORMULAS")
    print("="*72)

    for g in [4.5, 5.0, 6.0]:
        p = cfac_params(g)
        print(f"\n--- g = {g} ---")
        print(f"  f_s = {p['f_s']:.4f}, f_u = {p['f_u']:.4f}")
        print(f"  Ω'(f_s) = {p['Op_s']:.3f}, Ω'(f_u) = {p['Op_u']:.3f}")
        print(f"  W at f_s, f_u: {p['W_s']:.3f}, {p['W_u']:.3f}")
        print(f"  Φ''(f_s), Φ''(f_u): {p['Psi_s']:.3f}, {p['Psi_u']:.3f}")
        print(f"  S* = {p['S']:.5f}")
        print(f"  Candidate prefactors:")
        print(f"    A_Langevin 2π/√|Ω'Ω'|       = {p['A_langevin']:.3f}")
        print(f"    A_Φ        2π/√|Φ''Φ''|     = {p['A_Phi']:.3f}")
        print(f"    A_π²       π²/√|Φ''Φ''|     = {p['A_pi2']:.3f}")
        print(f"    A_D-corr   2π√(WsWu)/√|Ω'Ω'| = {p['A_Dcorr']:.3f}")

        print(f"\n  Exact MFPT and extracted prefactor:")
        print(f"  {'V':>4s} {'τ_exact':>12s} {'A_extr':>8s}")
        for V in [20, 30, 50, 80, 120, 200]:
            tau = exact_mfpt(V, g)
            if tau > 0 and tau < 1e30:
                A = tau / np.exp(V * p['S'])
                print(f"  {V:>4d} {tau:>12.3e} {A:>8.3f}")
            else:
                print(f"  {V:>4d}   (overflow)")


if __name__ == '__main__':
    main()
