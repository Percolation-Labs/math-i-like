"""
Advanced CFAC predictions for the canonical ant.

Beyond ν = 1/2 and β = 1/2, what NON-TRIVIAL numbers can CFAC
predict and verify?

(A) Branch-gap instanton MFPT for the cooperative canonical ant.
    MFPT = τ_0 exp(V S*),  S* = ∫_1^{z*} (x_unst - x_stab)/z dz.
    S* is a SPECIFIC NUMBER (not 1/2) that depends on the rate
    constants in a non-trivial way.

(B) Tunable Puiseux exponent from cooperativity.
    k_+ χ φ^p coupling gives DSE of degree (p+1). For p >= 2, the
    saddle-node is generic (β = 1/2), BUT the amplitude ratio
    and the pre-factor scale with p. More importantly, for p = 3
    one can tune to a tricritical point where β = 1/4 (not 1/2).

(C) Anomalous scaling of <φ|R>/<φ|S> vs λ.
    At strong coupling: <φ|R> saturates at δ/λ_eff,
    <φ|S> decays faster. Their ratio has a specific λ-dependence.

(D) Amplitude constant of two-point correlator.
    Not just the exponent — the AMPLITUDE of <φ(0)φ(r)> has a
    CFAC-computable value in terms of the DP susceptibility.

Let me pick (A) and (C) as most diagnostic.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────
# (A) Branch-gap instanton for cooperative canonical ant
# ─────────────────────────────────────────────────────────────────
def mfpt_gillespie_cooperative(V, k_plus=0.01, k_minus=0.05, chi=1.0,
                                 delta=2.0, lam=0.10, n_trials=200,
                                 max_time=1e6, seed=0):
    """
    Well-mixed Gillespie simulation of the cooperative canonical ant
    in a single cell of volume V:
      S -> R at rate k_+ χ φ²,     R -> S at rate k_-,
      R -> R + φ at rate δ,         φ -> ∅ at rate λ.

    Start in OFF state (N_R = 0, φ = 0).  Measure time to reach
    ON state (N_R > f_on * V).

    MFPT = average over trials.
    """
    rng = np.random.default_rng(seed)
    f_on = 0.3  # threshold for "arrived at ON"
    times = []

    for trial in range(n_trials):
        # Start near ON state to seed bistability then measure
        # escape to OFF — more tractable for large V.
        # Actually start OFF and measure time to cross to ON.
        N = V  # total agents
        N_R = 0
        N_S = N
        phi_particles = 0    # integer pheromone particles (sub-volume V)

        t = 0.0
        reached_on = False
        steps = 0

        while t < max_time and steps < 500_000:
            phi = phi_particles / V  # concentration
            # Rates
            r_SR = k_plus * chi * phi * phi * N_S     # S -> R
            r_RS = k_minus * N_R                       # R -> S
            r_dep = delta * N_R                        # emit pheromone
            r_ev = lam * phi_particles                 # one pheromone disappears
            r_tot = r_SR + r_RS + r_dep + r_ev
            if r_tot <= 0:
                break
            dt = -np.log(rng.random()) / r_tot
            t += dt
            u = rng.random() * r_tot
            if u < r_SR:
                N_S -= 1; N_R += 1
            elif u < r_SR + r_RS:
                N_R -= 1; N_S += 1
            elif u < r_SR + r_RS + r_dep:
                phi_particles += 1
            else:
                phi_particles -= 1
            steps += 1

            if N_R / N > f_on:
                reached_on = True
                break

        if reached_on:
            times.append(t)
        # Otherwise trial was capped — we treat as right-censored

    if not times:
        return np.nan, np.nan
    mfpt = float(np.mean(times))
    se = float(np.std(times) / np.sqrt(len(times)))
    return mfpt, se


def branch_gap_S_cooperative(alpha, g_c_ratio=1.5):
    """
    Compute the CFAC branch-gap action for the cooperative ant.

    Self-consistent equation: x = z * F(x) where F(x) = g χ φ²
    with the identification x = f_R.  For the cooperative ant
    without baseline, we have:
      f_R = α g f_R² (1 - f_R) / [α g f_R² (1 - f_R) + 1]
      (wait — this isn't quite right; let me re-derive)

    Actually the kinetic MF is:
      f_R(1 - f_R) = 1/(α g),   plus OFF root f_R = 0.

    The branch-gap contour integral requires parametrising the
    DP rate function.  For the cooperative ant, the effective
    Doi-Peliti action has the population-level deterministic
    dynamics:
      df_R/dt = k_+ χ <φ²> (1-f_R) - k_- f_R
    With φ = δ n f_R/λ at MF:
      df_R/dt = α k_- g_n f_R² (1-f_R) - k_- f_R
        where g_n = α g = α χ (δ n/λ)² k_+ / k_- ... wait this is
    getting tangled.

    The branch-gap is a specific integral of the self-consistent
    equation.  For our purposes, use that x = f_R solves
      α g x² (1-x) = x    (at steady-state, dividing by k_-)
      => x (1 - α g x (1-x)) = 0
      => x = 0 or  α g x(1-x) = 1
      => α g x² - α g x + 1 = 0
      => x = (α g ± √(α²g² - 4αg))/(2αg)
      x_stab = (αg - √D)/(2αg) vs x_unst = (αg + √D)/(2αg)
      No wait, (1 - √(1 - 4/αg))/2, (1 + √(1 - 4/αg))/2.

    Hmm, let me just compute the effective potential landscape.
    The deterministic ODE:
      df/dt = F(f) = α g f² (1-f) - f  (in units of k_-)

    Potential V(f) = -∫ F(f) df = ∫(f - α g f²(1-f)) df
                   = f²/2 - α g f³/3 + α g f⁴/4

    Barrier height: V(f_unst) - V(f_stab_OFF) where
       f_stab_OFF = 0, f_unst is the middle root.

    At the saddle-node g_c = 4/α:  f_unst = f_stab = 1/2,
    barrier = 0.
    For g > g_c, barrier grows.

    CFAC branch-gap S* is proportional to this barrier.
    For the cooperative case with quadratic coupling and our
    rate constants:
      V_barrier(g) = f_unst²/2 - αg f_unst³/3 + αg f_unst⁴/4
                    - V(0=0) = f_unst²/2 - αg f_unst³/3 + αg f_unst⁴/4

    The instanton MFPT scales as τ ~ exp(V · V_barrier) where
    the SECOND V is the system volume.

    Let me just compute and return V_barrier * V.
    """
    g_c = 4.0 / alpha
    g = g_c * g_c_ratio
    # f_unst is the middle root
    disc = 1 - 4/(alpha*g)
    if disc <= 0:
        return np.nan, None
    f_unst = (1 - np.sqrt(disc)) / 2
    f_stab = 0.0  # OFF state

    # Potential barrier (non-dimensional)
    V_at = lambda f: f**2/2 - alpha*g*f**3/3 + alpha*g*f**4/4
    barrier = V_at(f_unst) - V_at(f_stab)

    return barrier, f_unst


# ─────────────────────────────────────────────────────────────────
# (C) <φ|R> and <φ|S> scaling with λ at fixed n, σ
# ─────────────────────────────────────────────────────────────────
def phi_conditional_vs_lambda(ns=500):
    """Non-trivial prediction: the RATIO <φ|R>/<φ|S> has a specific
    λ-dependence set by the Markov chain dwell-time statistics.

    MF (single-site, no diffusion): a cell cycles A↔B with rates
    k_+(1+χφ) and k_-.  Field φ decays at rate λ.

    <φ|R> = δ/λ · (1 - e^(-λ τ_R))  where τ_R = 1/k_- is R-dwell
    <φ|S> = <φ|R> · e^(-λ τ_S)  where τ_S ≈ 1/k_+
    (heuristic — field decays from peak during S-state dwell)

    Compare <φ|R>/<φ|S> = e^(λ τ_S)  grows exponentially with λ.
    """
    pass


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    print("=" * 72)
    print("ADVANCED CFAC PREDICTIONS — NON-1/2 NUMBERS")
    print("=" * 72)

    # (A) MFPT exponent in the cooperative ant
    print("\n(A) Branch-gap MFPT in cooperative canonical ant")
    print("  τ_MFPT ~ exp(V · S*),  S* is CFAC instanton action")
    alpha = 0.2
    g_ratios = [1.2, 1.5, 2.0, 3.0]
    Vs = [20, 40, 80, 160]

    print(f"\n  CFAC predicts (barrier heights):")
    print(f"{'g/g_c':>8s} {'f_unst':>8s} {'V_bar':>10s}  (bigger g → bigger barrier)")
    for gr in g_ratios:
        bar, f_unst = branch_gap_S_cooperative(alpha, g_c_ratio=gr)
        print(f"{gr:>8.2f} {f_unst:>8.4f} {bar:>10.5f}")

    # Gillespie measurement: at fixed g/g_c = 2.0, vary V, measure MFPT
    # τ(V) = τ_0 exp(V · S*) → slope of log τ vs V gives S*
    print(f"\n  Measuring MFPT vs volume V (g/g_c = 2.0, OFF -> ON):")
    g_ratio = 2.0
    alpha = 0.2
    g_c = 4.0 / alpha
    g = g_c * g_ratio
    # Relate g to n: g = chi (δ n/λ)² → n = (λ/δ) √(g/chi)
    chi, delta, lam = 1.0, 2.0, 0.10
    n_fixed = (lam/delta) * np.sqrt(g/chi)
    print(f"    (n = {n_fixed:.4f} in continuum terms)")

    S_theory, f_unst = branch_gap_S_cooperative(alpha, g_c_ratio=g_ratio)
    print(f"    CFAC predicted barrier: V_bar = {S_theory:.4f}")
    print(f"    f_unst = {f_unst:.4f}")

    print(f"\n{'V':>5s} {'MFPT (Gillespie)':>18s} {'log τ':>8s}   (expect log τ = V · S + const)")
    log_taus = []
    for V in Vs:
        mfpt, se = mfpt_gillespie_cooperative(V, k_plus=0.01, k_minus=0.05,
                                                chi=1.0, delta=2.0, lam=0.10,
                                                n_trials=100, max_time=1e7)
        if not np.isnan(mfpt):
            log_taus.append((V, np.log(mfpt)))
            print(f"{V:>5d} {mfpt:>12.2f} ± {se:>4.1f} {np.log(mfpt):>8.3f}")
        else:
            print(f"{V:>5d}   (no transitions observed)")

    if len(log_taus) >= 2:
        Varr, logtau = zip(*log_taus)
        Varr = np.array(Varr); logtau = np.array(logtau)
        S_fit = np.polyfit(Varr, logtau, 1)[0]
        print(f"\n    Fitted S (Gillespie slope): {S_fit:.4f}")
        print(f"    CFAC branch-gap prediction:  {S_theory:.4f}")
        err = 100 * abs(S_fit - S_theory) / S_theory
        print(f"    Error: {err:.1f}%")


if __name__ == '__main__':
    main()
