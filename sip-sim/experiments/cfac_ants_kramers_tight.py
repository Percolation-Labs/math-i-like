"""
Tighter Kramers prefactor measurement at stronger bistability.

At g=4.5, S*=0.0265 so V·S* ≈ 5 at V=180 — only marginally asymptotic.
At g=6, S*≈0.109 so V·S* ≈ 5 at V=50 — asymptotic reached at smaller V,
allowing cleaner extrapolation.

CFAC prediction:
  A = 2π / sqrt(|Ω'(f_ON) · Ω'(f_unst)|)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log

FIGDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'rdft', 'paper', 'cfac', 'figures_ants'))
os.makedirs(FIGDIR, exist_ok=True)


def cfac_S_and_A(g, alpha=1.0, n_pts=2000):
    if alpha * g <= 4:
        return np.nan, np.nan, None, None
    disc = np.sqrt(1 - 4/(alpha*g))
    f_unst = 0.5 * (1 - disc)
    f_ON = 0.5 * (1 + disc)
    fs = np.linspace(f_unst + 1e-9, f_ON - 1e-9, n_pts)
    s = -np.log(alpha * g * fs * (1 - fs))
    S = abs(np.trapezoid(s, fs))
    # Ω'(f) = α g (2f - 3f²) - 1
    O_ON = alpha*g*(2*f_ON - 3*f_ON*f_ON) - 1
    O_unst = alpha*g*(2*f_unst - 3*f_unst*f_unst) - 1
    A = 2 * np.pi / np.sqrt(abs(O_ON * O_unst))
    return S, A, f_ON, f_unst


def gillespie_mfpt(V, g, alpha=1.0, f_thresh=0.1, n_trials=400,
                    max_steps=20_000_000, seed=0):
    rng = np.random.default_rng(seed)
    _, _, f_ON, _ = cfac_S_and_A(g, alpha)
    N_init = int(round(V * f_ON))
    times = []
    for _ in range(n_trials):
        N_R = N_init
        t = 0.0
        for _ in range(max_steps):
            f = N_R / V
            Wp = alpha*g*f*f*(V-N_R)
            Wm = N_R*1.0
            W = Wp + Wm
            if W <= 0:
                break
            dt = -np.log(rng.random()) / W
            t += dt
            if rng.random()*W < Wp:
                N_R += 1
            else:
                N_R -= 1
            if N_R/V < f_thresh:
                times.append(t)
                break
    if not times:
        return np.nan, np.nan
    return float(np.mean(times)), float(np.std(times)/np.sqrt(len(times)))


def main():
    # Stronger bistability: g=6 so S* ≈ 0.11, asymptotic reached at moderate V
    g = 6.0
    alpha = 1.0
    S, A_cfac, f_ON, f_unst = cfac_S_and_A(g, alpha)
    print(f"Params: α={alpha}, g={g}")
    print(f"  f_ON = {f_ON:.4f},  f_unst = {f_unst:.4f}")
    print(f"  CFAC S* = {S:.5f}")
    print(f"  CFAC A  = {A_cfac:.4f}")

    # Test at wider V range (expecting better asymptotic)
    Vs = [15, 25, 40, 60, 90, 130]
    print(f"\n{'V':>4s} {'τ':>10s} {'±SE':>7s} {'log τ':>8s} "
          f"{'A_if_S*':>9s}")
    print("-" * 50)
    results = []
    for V in Vs:
        tau, se = gillespie_mfpt(V, g, alpha, n_trials=500, seed=42+V)
        if np.isnan(tau):
            print(f"{V:>4d}  (no transitions)")
            continue
        A_meas = tau / np.exp(V*S)
        print(f"{V:>4d} {tau:>10.2f} {se:>7.2f} {log(tau):>8.3f} "
              f"{A_meas:>9.3f}")
        results.append((V, tau, se, A_meas))

    if len(results) >= 3:
        Varr = np.array([r[0] for r in results])
        tauarr = np.array([r[1] for r in results])
        Aarr = np.array([r[3] for r in results])

        # Fit A(V) = A_∞ + c/V asymptotic
        p = np.polyfit(1/Varr, Aarr, 1)
        A_inf = p[1]
        c1 = p[0]
        print(f"\nAsymptotic extrapolation A(V) = A_∞ + c/V:")
        print(f"  A_∞ = {A_inf:.3f}  (CFAC = {A_cfac:.3f})")
        print(f"  c₁ = {c1:.2f}")
        err = 100*abs(A_inf - A_cfac)/A_cfac
        print(f"  Error after 1/V extrapolation: {err:.1f}%")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                                  constrained_layout=True)

        # (a) τ(V) on log scale
        ax = axes[0]
        ax.semilogy(Varr, tauarr, 'bo', ms=10, mfc='none', mew=2,
                     label='Gillespie (400 trials each)')
        Vfit = np.linspace(Varr.min(), Varr.max(), 100)
        ax.semilogy(Vfit, A_cfac*np.exp(Vfit*S), 'k-', lw=2,
                     label=rf'CFAC: $A = {A_cfac:.2f}$, $S^* = {S:.3f}$')
        ax.set_xlabel(r'Volume $V$')
        ax.set_ylabel(r'MFPT $\tau$')
        ax.set_title(rf'Stronger bistability ($g={g}$): $V S^* \geq {V*S:.1f}$')
        ax.legend(frameon=False, fontsize=10, loc='lower right')
        ax.grid(alpha=0.3, which='both')

        # (b) A_meas(V) asymptotic extrapolation
        ax = axes[1]
        ax.plot(1/Varr, Aarr, 'bo', ms=10, mfc='none', mew=2,
                 label='measured')
        xr = np.linspace(0, (1/Varr).max()*1.1, 50)
        ax.plot(xr, A_inf + c1*xr, 'b-', alpha=0.6,
                 label=rf'fit $A_\infty + c/V$, $A_\infty={A_inf:.2f}$')
        ax.axhline(A_cfac, color='k', ls='--', lw=2,
                    label=rf'CFAC: $A = {A_cfac:.2f}$')
        ax.set_xlabel(r'$1/V$')
        ax.set_ylabel(r'$A = \tau / e^{VS^*}$')
        ax.set_title(rf'Asymptotic extrapolation $A_\infty \to A_{{\rm CFAC}}$ (err {err:.1f}\%)')
        ax.legend(frameon=False, fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)

        out = os.path.join(FIGDIR, 'canonical_ant_kramers_tight.pdf')
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
