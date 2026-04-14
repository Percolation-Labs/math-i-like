"""
Visualise state-space entropy reduction in the canonical ant.

Classical ants show SPATIAL entropy reduction: trails (channels).
The canonical ant lacks biased movement, so spatial channels
don't form; instead it shows STATE-SPACE entropy reduction.

Two visual panels:
  (a) Time series of f_R in cooperative ant (bistable switching)
  (b) Histogram of f_R over long simulation: bimodal distribution
      — peaks at OFF (f_R = 0) and ON (f_R ≈ 2/3)
  (c) Field φ conditional distributions: P(φ | R) vs P(φ | S)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def run_cooperative(N=80, V=80, alpha=1.0, g=5.5, n_steps=200_000,
                    seed=42):
    """Well-mixed Gillespie recorded at regular intervals."""
    rng = np.random.default_rng(seed)
    disc = np.sqrt(1 - 4/(alpha*g))
    f_ON = 0.5 * (1 + disc)
    N_R = int(V * f_ON)
    t = 0.0
    ts, fs = [], []
    record_every = 1.0  # time units

    t_record = 0.0
    for step in range(n_steps):
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
        if t >= t_record:
            ts.append(t); fs.append(N_R / V)
            t_record += record_every
    return np.array(ts), np.array(fs), f_ON, 0.5 * (1 - disc)


def main():
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    # Bistable regime, not too strong (so we see switches in finite time)
    g = 4.6
    V = 50
    print(f"Running cooperative ant: V = {V}, g = {g}")
    ts, fs, f_ON, f_unst = run_cooperative(V=V, alpha=1.0, g=g,
                                             n_steps=3_000_000, seed=0)
    print(f"  f_ON = {f_ON:.3f}, f_unst = {f_unst:.3f}, "
          f"total sim time = {ts[-1]:.0f}, samples = {len(fs)}")

    # Shannon entropy reduction
    # Uniform prior entropy: log(V+1) = log(51) ≈ 3.93
    # Observed entropy: compute from histogram
    bins = np.linspace(0, 1, V + 2)
    hist, _ = np.histogram(fs, bins=bins, density=False)
    p = hist / hist.sum()
    H = -np.sum(p[p > 0] * np.log(p[p > 0]))
    H_uniform = np.log(V + 1)
    print(f"  H(f_R) = {H:.3f} nats")
    print(f"  H_uniform = {H_uniform:.3f} nats")
    print(f"  Entropy reduction: {H_uniform - H:.3f} nats  "
          f"({100*(H_uniform-H)/H_uniform:.1f}% of max)")

    # Now run linear (non-cooperative) canonical ant for comparison:
    # f_R just follows a smooth distribution around the MF value
    print("\nRunning linear canonical ant for comparison...")
    # Linear: S→R at rate k_+(1+χφ), always monostable
    # Quick well-mixed sim
    def run_linear(V=50, alpha=1.0, g_eff=5.0, seed=1):
        rng = np.random.default_rng(seed)
        # Steady-state f from linear MF
        # α(1 + g f) = f(1 + α + α g f)
        # Same params as above, just linear
        N_R = int(V * 0.5)
        t = 0.0
        ts_l, fs_l = [], []
        t_rec = 0.0
        for _ in range(3_000_000):
            f = N_R / V
            phi = g_eff * f  # effective "field" (linear version)
            W_plus = alpha * (1 + phi) * (V - N_R)   # linear baseline
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
            if t >= t_rec:
                ts_l.append(t); fs_l.append(f)
                t_rec += 1.0
        return np.array(ts_l), np.array(fs_l)

    ts_lin, fs_lin = run_linear(V=V, alpha=1.0, g_eff=3.0)
    hist_lin, _ = np.histogram(fs_lin, bins=bins, density=False)
    p_lin = hist_lin / hist_lin.sum()
    H_lin = -np.sum(p_lin[p_lin > 0] * np.log(p_lin[p_lin > 0]))
    print(f"  Linear ant:  H(f_R) = {H_lin:.3f} nats")
    print(f"  Cooperative: H(f_R) = {H:.3f} nats")
    print(f"  Extra reduction from bistability: {H_lin - H:.3f} nats")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                              constrained_layout=True)

    # (a) Time series — cooperative
    ax = axes[0, 0]
    # Sub-sample for plotting
    skip = max(1, len(ts) // 5000)
    ax.plot(ts[::skip], fs[::skip], lw=0.6, color='crimson', alpha=0.9)
    ax.axhline(f_ON, color='green', ls='--', lw=1.5, alpha=0.7,
               label=rf'$f_{{\rm ON}} = {f_ON:.2f}$')
    ax.axhline(f_unst, color='orange', ls=':', lw=1.5, alpha=0.7,
               label=rf'$f_{{\rm unst}} = {f_unst:.2f}$')
    ax.axhline(0, color='blue', ls='--', lw=1.5, alpha=0.7,
               label=r'$f_{\rm OFF} = 0$')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'$f_R$')
    ax.set_title(rf'Cooperative ant (bistable, $g = {g}$): rare switches')
    ax.legend(frameon=False, fontsize=9, loc='right')
    ax.set_ylim(-0.05, 1.05)

    # (b) Histogram — cooperative (bimodal)
    ax = axes[0, 1]
    ax.hist(fs, bins=40, density=True, color='crimson', alpha=0.7,
            label=rf'cooperative: $H = {H:.2f}$ nats')
    ax.hist(fs_lin, bins=40, density=True, color='steelblue', alpha=0.5,
            label=rf'linear: $H = {H_lin:.2f}$ nats')
    ax.set_xlabel(r'$f_R$')
    ax.set_ylabel('Probability density')
    ax.set_title('State distribution: cooperative is bimodal')
    ax.legend(frameon=False, fontsize=9)

    # (c) Autocorrelation
    ax = axes[1, 0]
    # Sample at uniform grid
    t_unif = np.linspace(ts[0], ts[-1], 8000)
    f_unif = np.interp(t_unif, ts, fs)
    f_c = f_unif - f_unif.mean()
    corr = np.correlate(f_c, f_c, mode='full')[len(f_c)-1:]
    corr = corr / corr[0]
    tau_plot = np.arange(len(corr)) * (t_unif[1] - t_unif[0])
    mask = tau_plot < 1000
    ax.plot(tau_plot[mask], corr[mask], lw=1.5, color='crimson')
    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel(r'$C(\tau)$')
    ax.set_title('Autocorrelation: long-time tail from bistable switching')
    ax.axhline(0, color='k', lw=0.3)

    # (d) Summary text panel
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        rf"\textbf{{State-space entropy reduction}}" + "\n" +
        rf"Cooperative ant, $V={V}$, $g={g}$:" + "\n"
        rf"$H_{{\rm uniform}} = \ln(V+1) = {H_uniform:.2f}$ nats" + "\n"
        rf"$H(f_R)_{{\rm cooperative}} = {H:.2f}$ nats" + "\n"
        rf"$\Delta H = {H_uniform - H:.2f}$ nats "
        rf"(= {100*(H_uniform-H)/H_uniform:.0f}\% of uniform)" + "\n"
        rf"$H(f_R)_{{\rm linear}} = {H_lin:.2f}$ nats" + "\n\n"
        rf"The canonical ant does NOT form spatial" + "\n"
        rf"trails (no biased movement, by design)." + "\n"
        rf"Instead: bistable state-space. The" + "\n"
        rf"cooperative variant sits in ON or OFF" + "\n"
        rf"for long times; rare switches between" + "\n"
        rf"basins are governed by the CFAC" + "\n"
        rf"instanton action $S^* = 0.0265$."
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            verticalalignment='top', fontsize=11, family='sans-serif')

    out = os.path.join(figdir, 'canonical_ant_entropy.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
