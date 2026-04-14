"""
Clean visualization of state-space order.

ONE main panel: f_R(t) time series showing dwell-switch-dwell.
ONE small panel: histogram alongside.
That's it.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'rdft', 'paper', 'wip', 'figures_ants'))
os.makedirs(FIGDIR, exist_ok=True)


def run_long_gillespie(V=60, alpha=1.0, g=4.5, eps0=0.002,
                        n_steps=10_000_000, seed=42):
    rng = np.random.default_rng(seed)
    disc = np.sqrt(1 - 4/(alpha*g))
    f_ON = 0.5 * (1 + disc)
    f_unst = 0.5 * (1 - disc)
    N_R = int(V * f_ON)
    t = 0.0
    ts, fs = [t], [N_R/V]
    record_every = 2.0
    t_record = record_every
    state = 'ON' if N_R/V > f_unst else 'OFF'
    switch_times = []
    for _ in range(n_steps):
        f = N_R / V
        W_plus = alpha * (eps0 + g * f * f) * (V - N_R)
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
        new_state = 'ON' if N_R/V > f_unst else 'OFF'
        if new_state != state:
            switch_times.append(t)
            state = new_state
        if t >= t_record:
            ts.append(t); fs.append(N_R/V)
            t_record += record_every
    return (np.array(ts), np.array(fs), f_ON, f_unst,
            np.array(switch_times))


def main():
    print("Running LONG Gillespie (10^7 steps)...")
    ts, fs, f_ON, f_unst, sw = run_long_gillespie(V=60, g=4.5,
                                                    n_steps=10_000_000)
    print(f"  Total time: {ts[-1]:.0f}, switches: {len(sw)}")

    # Clean 2-panel figure: time series + histogram
    fig = plt.figure(figsize=(11, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 5)
    ax_main = fig.add_subplot(gs[0, :4])
    ax_hist = fig.add_subplot(gs[0, 4], sharey=ax_main)

    # Time series
    skip = max(1, len(ts)//8000)
    ax_main.plot(ts[::skip]/1000, fs[::skip], lw=0.6, color='#c0392b')
    ax_main.axhline(f_ON, color='#27ae60', ls='--', lw=1.2)
    ax_main.axhline(f_unst, color='#f39c12', ls=':', lw=1.2)
    ax_main.axhline(0, color='#2980b9', ls='--', lw=1.2)
    ax_main.text(ts[-1]/1000 * 0.99, f_ON + 0.02, r'$f_{\rm ON}$',
                  ha='right', color='#27ae60', fontsize=11)
    ax_main.text(ts[-1]/1000 * 0.99, f_unst - 0.06, r'barrier',
                  ha='right', color='#f39c12', fontsize=10)
    ax_main.text(ts[-1]/1000 * 0.99, 0.03, r'$f_{\rm OFF}$',
                  ha='right', color='#2980b9', fontsize=11)
    ax_main.set_xlabel(r'Time ($\times 10^3$)')
    ax_main.set_ylabel(r'$f_R(t)$')
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.grid(alpha=0.2)
    ax_main.set_title(
        rf'Colony state over $\sim {int(ts[-1]/1000)}$k time units: '
        rf'{len(sw)} ON$\leftrightarrow$OFF switches, '
        rf'typical dwell $\sim$ {int(ts[-1]/max(len(sw),1)/1000)}k units')

    # Histogram
    hist, edges = np.histogram(fs, bins=40, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax_hist.fill_betweenx(centres, 0, hist, color='#c0392b', alpha=0.75)
    ax_hist.set_xlabel(r'$P(f_R)$')
    ax_hist.tick_params(labelleft=False)
    ax_hist.grid(alpha=0.2)
    ax_hist.set_title('Distribution')

    out = os.path.join(FIGDIR, 'canonical_ant_order_visualisation.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
