"""
RARE-EVENT / INSTANTON prediction for the cooperative canonical ant.

In a well-mixed container of volume V, the cooperative ant
  S -> R at rate k_+ χ φ²,  R -> S at rate k_-,
  R emits φ at rate δ, φ decays at rate λ

has two stable states (ON, OFF) in the bistable regime.  The
mean first-passage time to cross from ON basin to OFF
(or vice-versa) scales as
    τ_MFPT ~ exp(V · S*),
where S* is the CFAC branch-gap instanton action — a SPECIFIC
non-trivial number that depends on all the rate constants.

This is the SIGNATURE prediction of CFAC for rare events.
We verify:
  (a) exponential scaling with V
  (b) the fitted slope matches the large-deviation rate function

In the quasi-stationary limit (fast φ equilibration compared to
S↔R dynamics), φ = δ f_R / λ (algebraic), and the f_R dynamics
reduce to a 1D birth-death process:
  N_R -> N_R+1  at rate W_+(N_R) = α (δ/λ)² (N_R/V)² (V - N_R)
  N_R -> N_R-1  at rate W_-(N_R) = N_R        (in units of k_-)

The large-deviation action for the escape from ON basin to OFF:
  S_ON→OFF = ∫_{f_ON}^{f_unst} ln(W_-(f)/W_+(f)) df
  (in the continuum limit V → ∞)

This is a SPECIFIC INTEGRAL that we can compute analytically AND
verify by Gillespie simulation at several V.

Biological relevance: a colony in the ON (recruiting) state rare-
switches to OFF — the MFPT scaling quantifies the persistence of
recruitment.  Measurable by counting switches per hour in
controlled colony experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, exp


def cfac_action_on_to_off(g, alpha=1.0, n_pts=500):
    """
    Large-deviation rate function integrand (CFAC branch-gap):
      s(f) = ln(W_-(f) / W_+(f))
    with W_+ = α g f² (1-f), W_- = f
    (in units of k_-, per cell).

    Action for ON → OFF escape:
      S = ∫_{f_unst}^{f_ON} s(f) df      (detailed balance — the
    integral from ON down to the unstable middle root)

    Note: the sign conventions vary; we take the convention that
    S > 0 is the barrier height.  For the rate W_+/W_-:
      ratio > 1 for f < f_unst (backward drift dominates)
      ratio < 1 for f > f_unst (forward drift dominates in ON-to-OFF escape)
    """
    # Three roots of W_+ = W_- => α g f²(1-f) = f => α g f (1-f) = 1
    # f² - f + 1/(αg) = 0 => f_unst = (1-√(1-4/(αg)))/2,  f_ON = (1+√(...))/2
    if alpha * g <= 4:
        return np.nan, None, None, None
    disc = np.sqrt(1 - 4/(alpha*g))
    f_unst = 0.5 * (1 - disc)
    f_ON = 0.5 * (1 + disc)

    # Integrate log(W_-/W_+) from f_unst to f_ON (should be negative → flip sign)
    # Actually CFAC convention: S_escape = ∫ ln(W_-/W_+) df over the escape path.
    # The integrand s(f) = ln(f/(α g f²(1-f))) = -ln(α g f (1-f))
    # For f near f_unst or f_ON: s(f_*)=0 because αg f_*(1-f_*) = 1.
    # Between them: αg f(1-f) > 1 so s(f) < 0.
    # So ∫_{f_unst}^{f_ON} s(f) df < 0.  Take |·| for barrier height.
    fs = np.linspace(f_unst + 1e-9, f_ON - 1e-9, n_pts)
    s = -np.log(alpha * g * fs * (1 - fs))
    S = np.trapezoid(s, fs)
    return abs(S), f_unst, f_ON, (fs, s)


def gillespie_ON_to_OFF(V, g, alpha, k_minus=1.0, f_OFF_thresh=0.1,
                         n_trials=150, max_steps=5_000_000, seed=0):
    """
    Well-mixed Gillespie (quasi-stationary φ: φ = δ N_R/(λV) adiabatic).
    Start at ON basin; measure time to reach f_R < f_OFF_thresh.

    In time units of 1/k_-:
      W_+(N_R) = α g (N_R/V)² (V - N_R)       [S→R transition]
      W_-(N_R) = N_R                            [R→S transition]
    """
    rng = np.random.default_rng(seed)
    # Initial state: near f_ON
    disc = np.sqrt(1 - 4/(alpha*g))
    f_ON = 0.5 * (1 + disc)
    N_R_init = int(round(V * f_ON))

    times = []
    for trial in range(n_trials):
        N_R = N_R_init
        t = 0.0
        for _ in range(max_steps):
            f = N_R / V
            W_plus = alpha * g * f * f * (V - N_R)
            W_minus = N_R * 1.0       # in units of k_-
            W = W_plus + W_minus
            if W <= 0:
                break
            dt = -np.log(rng.random()) / W
            t += dt
            if rng.random() * W < W_plus:
                N_R += 1
            else:
                N_R -= 1
            if N_R / V < f_OFF_thresh:
                times.append(t)
                break
    if not times:
        return np.nan, np.nan, 0
    return float(np.mean(times)), float(np.std(times) / np.sqrt(len(times))), len(times)


def main():
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    print("=" * 72)
    print("CFAC RARE-EVENT PREDICTION: instanton MFPT for cooperative ant")
    print("=" * 72)

    # Just-above-saddle-node regime (barrier small, MFPT tractable)
    alpha = 1.0
    g = 4.5       # only slightly above g_c = 4; barrier ~ 0.05-0.1

    S_pred, f_unst, f_ON, prof = cfac_action_on_to_off(g, alpha)
    print(f"\n  Parameters: α = {alpha},  g = {g}  (bistable, g_c = 4)")
    print(f"  OFF basin at f=0,  f_unst = {f_unst:.4f},  f_ON = {f_ON:.4f}")
    print(f"  CFAC instanton action S* = {S_pred:.4f}"
          f"   (a non-trivial number — NOT 1/2)")

    # Gillespie at several V (wide range for clean exponential scaling)
    Vs = [30, 50, 80, 120, 180, 250, 350]
    log_taus = []
    print(f"\n  Measuring τ_MFPT by Gillespie:")
    print(f"{'V':>4s} {'τ_MFPT':>10s}  {'±SE':>6s}  {'log(τ)':>8s}  {'log(τ)/V':>10s}")
    for V in Vs:
        tau, se, n_ok = gillespie_ON_to_OFF(V, g, alpha,
                                              n_trials=200, seed=42+V)
        if not np.isnan(tau):
            lt = log(tau)
            print(f"{V:>4d} {tau:>10.2f}  {se:>6.2f}  {lt:>8.3f}  {lt/V:>10.4f}  "
                  f"({n_ok}/200 trials)")
            log_taus.append((V, lt))
        else:
            print(f"{V:>4d}  (no transitions observed)")

    if len(log_taus) >= 3:
        Varr, logtau = zip(*log_taus)
        Varr = np.array(Varr); logtau = np.array(logtau)
        slope, intercept = np.polyfit(Varr, logtau, 1)
        print(f"\n  FITTED slope (= S_instanton from Gillespie): {slope:.4f}")
        print(f"  CFAC predicted S* (analytical):               {S_pred:.4f}")
        err = 100 * abs(slope - S_pred) / S_pred
        print(f"  Agreement: {err:.1f}%")

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                                constrained_layout=True)

        ax1 = ax[0]
        if prof is not None:
            fs, s = prof
            ax1.plot(fs, s, lw=2, color='teal')
            ax1.fill_between(fs, s, 0, alpha=0.3, color='teal',
                              label=rf'$\int = -S^*$')
            ax1.axvline(f_unst, color='r', ls='--', alpha=0.7,
                         label=rf'$f_{{\rm unst}}={f_unst:.3f}$')
            ax1.axvline(f_ON, color='g', ls='--', alpha=0.7,
                         label=rf'$f_{{\rm ON}}={f_ON:.3f}$')
        ax1.axhline(0, color='k', lw=0.5)
        ax1.set_xlabel(r'$f_R$')
        ax1.set_ylabel(r'Rate-function integrand $s(f_R)$')
        ax1.set_title(rf'CFAC instanton integrand, $S^*={S_pred:.3f}$')
        ax1.legend(frameon=False, fontsize=9)
        ax1.grid(alpha=0.3)

        ax2 = ax[1]
        ax2.semilogy(Varr, np.exp(logtau), 'bo', ms=8, mfc='none',
                      mew=2, label='Gillespie')
        Vfit = np.linspace(Varr.min(), Varr.max(), 100)
        ax2.semilogy(Vfit, np.exp(slope * Vfit + intercept),
                      'b-', alpha=0.6,
                      label=rf'fit: slope = {slope:.3f}')
        ax2.semilogy(Vfit, np.exp(S_pred * Vfit + intercept -
                                    (slope - S_pred) * Varr[0]),
                      'k--', label=rf'CFAC: $S^* = {S_pred:.3f}$')
        ax2.set_xlabel(r'Volume $V$')
        ax2.set_ylabel(r'MFPT $\tau$')
        ax2.set_title(r'Exponential MFPT scaling with volume')
        ax2.legend(frameon=False)
        ax2.grid(alpha=0.3, which='both')

        out = os.path.join(figdir, 'canonical_ant_instanton.pdf')
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved → {out}")

    return S_pred


if __name__ == '__main__':
    main()
