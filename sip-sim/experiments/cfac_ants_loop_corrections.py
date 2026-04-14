"""
1-loop CFAC corrections for the cooperative canonical ant.

Three non-trivial loop-level predictions we can compute and verify:

(A) Kramers (Eyring) prefactor of the rare-event MFPT
    τ = A_CFAC · exp(V S*)
    A_CFAC = 2π / sqrt(|Ω'(f_ON) · Ω'(f_unst)|)
    where Ω(f) = W_+(f) - W_-(f) is the deterministic drift.
    This is a SPECIFIC NON-TRIVIAL formula, not a symmetry number.

(B) Relaxation-time divergence near the saddle-node
    τ_relax(g) ∝ (g - g_c)^{-γ}
    CFAC prediction from linearised dynamics: γ = 1/2 (sqrt-cusp
    characteristic of saddle-node).  Measurable from decay of
    autocorrelation near the bifurcation.

(C) Prefactor scaling across the bistable regime
    As we tune g above g_c = 4, the barrier S*(g) goes from 0 to
    a maximum around g ≈ 4.5-5, then saturates. The prefactor
    A_CFAC(g) also varies. Check both predictions against Gillespie
    at several g.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def S_star(g, alpha=1.0):
    """CFAC instanton action, cooperative ant without baseline."""
    if alpha * g <= 4:
        return 0.0, None, None
    disc = np.sqrt(1 - 4/(alpha*g))
    f_unst = 0.5 * (1 - disc)
    f_ON = 0.5 * (1 + disc)
    fs = np.linspace(f_unst + 1e-9, f_ON - 1e-9, 2000)
    s = -np.log(alpha * g * fs * (1 - fs))
    S = abs(np.trapezoid(s, fs))
    return S, f_unst, f_ON


def kramers_prefactor(g, alpha=1.0):
    """
    1-loop Kramers prefactor for the cooperative ant MFPT:
      A = 2π / sqrt(|Ω'(f_ON) · Ω'(f_unst)|)
    where Ω = W_+ - W_- is the deterministic drift.
    """
    if alpha * g <= 4:
        return np.nan
    disc = np.sqrt(1 - 4/(alpha*g))
    f_unst = 0.5 * (1 - disc)
    f_ON = 0.5 * (1 + disc)
    # Ω(f) = α g f²(1-f) - f,  Ω'(f) = α g (2f(1-f) - f²) - 1
    #                               = α g (2f - 3f²) - 1
    def omega_prime(f):
        return alpha * g * (2 * f - 3 * f * f) - 1.0
    O_ON = omega_prime(f_ON)
    O_unst = omega_prime(f_unst)
    A = 2 * np.pi / np.sqrt(abs(O_ON * O_unst))
    return A, O_ON, O_unst


def gillespie_mfpt(V, g, alpha=1.0, f_OFF_thresh=0.1,
                    n_trials=100, max_steps=5_000_000, seed=0):
    rng = np.random.default_rng(seed)
    disc = np.sqrt(1 - 4/(alpha*g))
    f_ON = 0.5 * (1 + disc)
    N_R_init = int(round(V * f_ON))
    times = []
    for _ in range(n_trials):
        N_R = N_R_init
        t = 0.0
        for _ in range(max_steps):
            f = N_R / V
            W_plus = alpha * g * f * f * (V - N_R)
            W_minus = N_R * 1.0
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
        return np.nan, np.nan
    return float(np.mean(times)), float(np.std(times)/np.sqrt(len(times)))


def analyse_existing_prefactor():
    """Use data from earlier run (g=4.5, α=1) to extract A."""
    print("\n" + "=" * 72)
    print("(A) KRAMERS PREFACTOR — existing g=4.5 data")
    print("=" * 72)

    V = np.array([30, 50, 80, 120, 180])
    tau = np.array([17.42, 29.98, 61.03, 177.36, 807.97])

    S, f_unst, f_ON = S_star(4.5)
    A_cfac, O_ON, O_unst = kramers_prefactor(4.5)

    print(f"  g = 4.5, α = 1:")
    print(f"    f_unst = {f_unst:.4f},  f_ON = {f_ON:.4f}")
    print(f"    Ω'(f_ON) = {O_ON:.4f},  Ω'(f_unst) = {O_unst:.4f}")
    print(f"    CFAC instanton action S* = {S:.5f}")
    print(f"    CFAC Kramers prefactor A_CFAC = {A_cfac:.4f}")

    A_meas = tau / np.exp(V * S)
    print(f"\n  Gillespie extraction  A = τ/exp(VS*):")
    for i, v in enumerate(V):
        print(f"    V={v:3d}: A = {A_meas[i]:.3f}")
    A_avg = A_meas.mean()
    print(f"\n  Mean measured A = {A_avg:.3f}")
    print(f"  CFAC prediction: {A_cfac:.3f}")
    print(f"  Agreement:       {100*abs(A_avg - A_cfac)/A_cfac:.1f}%")

    return A_cfac, A_avg, V, A_meas


def sweep_g_verify():
    """Sweep g and check both S* and A_CFAC vs Gillespie."""
    print("\n" + "=" * 72)
    print("(B) COUPLING-STRENGTH SWEEP: S*(g) AND A(g)")
    print("=" * 72)

    gs = [4.2, 4.5, 5.0, 6.0, 8.0]
    V = 80  # fixed

    print(f"\n  V = {V}")
    print(f"\n{'g':>5s} {'S_CFAC':>8s} {'A_CFAC':>8s} {'τ_G':>10s} {'S_meas':>8s} {'A_meas':>8s}")
    print("-" * 60)

    results = []
    for g in gs:
        S_cfac, f_unst, f_ON = S_star(g)
        A_cfac, _, _ = kramers_prefactor(g)
        tau_G, se = gillespie_mfpt(V, g, n_trials=500, seed=42)
        if np.isnan(tau_G):
            print(f"{g:>5.2f} {S_cfac:>8.4f} {A_cfac:>8.3f}   (no trans)")
            continue
        # log τ_G = log A_meas + V S_meas.  Can't get both from one V.
        # But if we measured many V, we'd fit both.  Here just compare
        # to tau_predict = A_CFAC · exp(V · S_CFAC):
        tau_pred = A_cfac * np.exp(V * S_cfac)
        A_if_S_cfac = tau_G / np.exp(V * S_cfac)
        print(f"{g:>5.2f} {S_cfac:>8.4f} {A_cfac:>8.3f} {tau_G:>10.2f}     -   {A_if_S_cfac:>8.3f}")
        results.append((g, S_cfac, A_cfac, tau_G, A_if_S_cfac))
    return results


def relaxation_time_near_saddle():
    """Measure autocorrelation time of f_R as g → g_c⁺."""
    print("\n" + "=" * 72)
    print("(C) CRITICAL SLOWING AT SADDLE-NODE: τ_relax(g)")
    print("  CFAC prediction: τ_relax ∝ (g - g_c)^{-1/2} (mean-field)")
    print("=" * 72)

    # Run Gillespie for a long time starting near f_ON; measure decay
    # of autocorrelation of f_R.
    gs = [4.2, 4.5, 5.0, 6.0, 8.0, 12.0]
    V = 200
    alpha = 1.0
    tau_relax = []
    for g in gs:
        if alpha * g <= 4:
            tau_relax.append(np.nan)
            continue
        disc = np.sqrt(1 - 4/(alpha*g))
        f_ON = 0.5 * (1 + disc)
        # CFAC prediction for relaxation rate:
        # τ_relax = 1/|Ω'(f_ON)| = 1/|α g (2 f_ON - 3 f_ON²) - 1|
        omega_p = alpha * g * (2 * f_ON - 3 * f_ON**2) - 1
        tau_relax_cfac = 1 / abs(omega_p)

        # Direct Gillespie: start at f_ON, trace f_R(t), estimate
        # autocorrelation decay time.
        rng = np.random.default_rng(42)
        N_R_init = int(round(V * f_ON))
        N_R = N_R_init
        ts = [0.0]; fs = [f_ON]
        t = 0.0
        for _ in range(200_000):
            f = N_R / V
            W_plus = alpha * g * f * f * (V - N_R)
            W_minus = N_R * 1.0
            W = W_plus + W_minus
            if W <= 0:
                break
            dt = -np.log(rng.random()) / W
            t += dt
            if rng.random() * W < W_plus:
                N_R += 1
            else:
                N_R -= 1
            # don't let it cross into OFF basin
            if N_R / V < (f_ON - disc * 1.5):
                N_R = N_R_init
            ts.append(t); fs.append(N_R / V)
        ts = np.array(ts); fs = np.array(fs)

        # Autocorr: sample f at uniform time grid
        if len(ts) < 100:
            tau_relax.append(np.nan)
            continue
        t_uniform = np.linspace(ts[0], ts[-1], 5000)
        f_uniform = np.interp(t_uniform, ts, fs)
        f_c = f_uniform - f_uniform.mean()
        # Autocorr via FFT
        F = np.fft.rfft(f_c)
        corr = np.fft.irfft(F * np.conj(F))[:len(f_uniform)//2]
        corr = corr / corr[0]
        # 1/e time
        idx = np.where(corr < np.exp(-1))[0]
        if len(idx) == 0:
            tau_relax.append(np.nan)
            continue
        dt_sample = t_uniform[1] - t_uniform[0]
        tau_auto = idx[0] * dt_sample
        tau_relax.append((g, tau_relax_cfac, tau_auto))
        print(f"  g={g:>5.2f}: τ_CFAC={tau_relax_cfac:>7.2f}  "
              f"τ_Gillespie={tau_auto:>7.2f}")

    # Fit: log τ_CFAC = -γ log(g - g_c) + const
    valid = [(g, tc, tg) for row in tau_relax
             if isinstance(row, tuple)
             for (g, tc, tg) in [row]]
    if valid:
        gs_v, tcs_v, tgs_v = zip(*valid)
        eps = np.array(gs_v) - 4.0
        sl_cfac = -np.polyfit(np.log(eps), np.log(tcs_v), 1)[0]
        print(f"\n  CFAC prediction slope γ (τ_relax ∝ (g-g_c)^{{-γ}}): "
              f"{sl_cfac:.3f}")
        print(f"  Theoretical (MF saddle-node): γ = 1/2 = 0.5")
    return tau_relax


if __name__ == '__main__':
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    print("=" * 72)
    print("CFAC LOOP-LEVEL CORRECTIONS — NON-TRIVIAL NUMBERS")
    print("=" * 72)

    A_cfac, A_meas, V_data, Ameas_data = analyse_existing_prefactor()
    sweep_results = sweep_g_verify()
    slowing_results = relaxation_time_near_saddle()

    # Final figure: Kramers prefactor agreement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                              constrained_layout=True)

    ax = axes[0]
    ax.plot(V_data, Ameas_data, 'bo', ms=9, mfc='none', mew=2,
            label='measured $A = \\tau/e^{VS^*}$')
    ax.axhline(A_cfac, color='k', ls='--', lw=2,
               label=rf'CFAC: $A = 2\pi/\sqrt{{|\Omega\'(f_{{\rm ON}})\Omega\'(f_{{\rm unst}})|}} = {A_cfac:.3f}$')
    ax.axhline(A_meas.mean(), color='r', ls=':', lw=1.5,
               label=rf'mean measured = {A_meas.mean():.3f}')
    ax.set_xlabel('Volume $V$')
    ax.set_ylabel('Kramers prefactor $A$')
    ax.set_title('1-loop prefactor: CFAC formula vs Gillespie')
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 12)

    ax = axes[1]
    # Relaxation curves
    valid = [r for r in slowing_results if isinstance(r, tuple)]
    if valid:
        gs_v = np.array([r[0] for r in valid])
        tcs_v = np.array([r[1] for r in valid])
        tgs_v = np.array([r[2] for r in valid])
        eps = gs_v - 4.0
        ax.loglog(eps, tcs_v, 'k-', lw=2,
                  label=r'CFAC: $\tau_{\rm relax} = 1/|\Omega\'(f_{\rm ON})|$')
        ax.loglog(eps, tgs_v, 'rs', ms=9, mfc='none', mew=2,
                  label='Gillespie autocorrelation')
        # Reference slope 1/2
        ref = tcs_v[0] * np.sqrt(eps[0] / eps)
        ax.loglog(eps, ref, 'k:', alpha=0.5,
                  label=r'$\propto (g - g_c)^{-1/2}$ reference')
    ax.set_xlabel(r'$g - g_c$')
    ax.set_ylabel(r'$\tau_{\rm relax}$')
    ax.set_title('Critical slowing: saddle-node relaxation')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which='both')

    out = os.path.join(figdir, 'canonical_ant_loop_corrections.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")
