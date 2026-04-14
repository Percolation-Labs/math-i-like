"""
SIR / Borel total-progeny: showing what AC adds beyond the leading n^{-3/2}.

Three panels:
  (1) 1/n correction curve at criticality (R0=1)
  (2) Subcritical amplitude C(R0) and decay rho(R0) from AC
  (3) Scaling collapse of P(n; R0) near criticality

For every panel we overlay:
  - "classical" leading answer
  - AC prediction (Puiseux / singularity analysis)
  - direct simulation of the Galton-Watson branching process with Poisson(R0) offspring
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

rng = np.random.default_rng(20260414)


def borel_tanner_logpmf(n, R0):
    # P(n) = (R0 n)^(n-1) exp(-R0 n) / n!
    return (n - 1) * np.log(R0 * n) - R0 * n - gammaln(n + 1)


def simulate_total_progeny(R0, n_trials, max_size=10_000_000):
    """Galton-Watson with Poisson(R0) offspring; return total progeny per trial."""
    sizes = np.empty(n_trials, dtype=np.int64)
    for t in range(n_trials):
        alive = 1
        total = 1
        while alive > 0 and total < max_size:
            # sum of `alive` independent Poisson(R0) draws == Poisson(alive*R0)
            new = rng.poisson(alive * R0)
            total += new
            alive = new
        sizes[t] = total
    return sizes


def simulate_size_and_height(R0, n_trials, max_size=200_000):
    """Galton-Watson; return (total progeny, generations until extinction)."""
    sizes = np.empty(n_trials, dtype=np.int64)
    heights = np.empty(n_trials, dtype=np.int64)
    for t in range(n_trials):
        alive = 1
        total = 1
        gen = 0
        while alive > 0 and total < max_size:
            new = rng.poisson(alive * R0)
            total += new
            if new > 0:
                gen += 1
            alive = new
        sizes[t] = total
        heights[t] = gen
    return sizes, heights


def density_2maxe(y, kmax=80):
    """
    Density of Y = 2*M where M = max of normalized Brownian excursion.
    Aldous limit law for H_n/sqrt(n) at criticality with sigma=1.
    Uses Kennedy's series via numerical differentiation of the CDF
    P(M <= x) = 1 + 2 sum_{k>=1} (1 - 4k^2 x^2) e^{-2 k^2 x^2}.
    Numerical differentiation is more robust against series-cancellation
    noise than differentiating term by term.
    """
    x = y / 2.0
    h = 1e-3

    def cdf_M(xx):
        out = np.ones_like(xx, dtype=float)
        for k in range(1, kmax + 1):
            out = out + 2.0 * (1.0 - 4 * k**2 * xx**2) * np.exp(-2 * k**2 * xx**2)
        return np.clip(out, 0.0, 1.0)

    fM = (cdf_M(x + h) - cdf_M(x - h)) / (2 * h)
    fM = np.clip(fM, 0.0, None)
    return 0.5 * fM


# ---------- Panel 1: 1/n correction at criticality ----------
def panel1(ax):
    ns = np.arange(2, 301)
    # exact Borel at R0=1
    logP = borel_tanner_logpmf(ns, 1.0)
    y_exact = np.sqrt(2 * np.pi * ns**3) * np.exp(logP)

    y_leading = np.ones_like(ns, dtype=float)                       # classical
    y_ac1 = 1 - 1 / (12 * ns)                                        # AC to 1/n
    # next Puiseux term: the O(1/n^2) coefficient for Borel is +1/288
    y_ac2 = 1 - 1 / (12 * ns) + 1 / (288 * ns**2)

    # simulation: bin into discrete n buckets
    N_trials = 400_000
    sizes = simulate_total_progeny(1.0, N_trials, max_size=5000)
    sizes = sizes[sizes < 5000]
    counts = np.bincount(sizes, minlength=ns.max() + 1)
    emp = counts[ns] / N_trials
    emp_y = np.sqrt(2 * np.pi * ns**3) * emp
    # smooth by log-binning for visual clarity
    edges = np.unique(np.round(np.logspace(np.log10(2), np.log10(300), 25)).astype(int))
    centres, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ns >= lo) & (ns < hi)
        if emp[sel].sum() > 0:
            centres.append(np.sqrt(lo * hi))
            means.append(emp_y[sel].mean())
            sems.append(emp_y[sel].std() / np.sqrt(sel.sum()))

    ax.axhline(1.0, color="0.6", ls=":", label="Classical leading (Borel $n^{-3/2}$)")
    ax.plot(ns, y_ac1, "b-", lw=2, label=r"AC: $1-\frac{1}{12n}$ (Puiseux $O(1/n)$)")
    ax.plot(ns, y_ac2, "g--", lw=1.5, label=r"AC: $1-\frac{1}{12n}+\frac{1}{288 n^2}$")
    ax.plot(ns, y_exact, "k-", lw=0.8, alpha=0.6, label="Exact Borel $P(n)$")
    ax.errorbar(centres, means, yerr=sems, fmt="o", color="crimson",
                ms=4, label=f"Simulation ($N={N_trials:,}$)")

    ax.set_xscale("log")
    ax.set_xlabel("$n$ (outbreak size)")
    ax.set_ylabel(r"$\sqrt{2\pi n^{3}}\, P(n)$")
    ax.set_title("Panel 1 — $1/n$ correction at criticality $R_0=1$")
    ax.set_ylim(0.92, 1.01)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)


# ---------- Panel 2: subcritical amplitude & decay ----------
def panel2(axL, axR):
    R0s = np.linspace(0.5, 0.98, 11)

    # AC closed forms from singularity analysis of T(z)=z exp(R0(T-1))
    #   rho(R0) = 1/z* = R0 exp(1-R0)
    #   C(R0)   = 1 / (R0 sqrt(2 pi))     [amplitude in P(n) ~ C rho^n n^{-3/2}]
    rho_ac = R0s * np.exp(1 - R0s)
    C_ac = 1.0 / (R0s * np.sqrt(2 * np.pi))

    rho_sim, C_sim = [], []
    for R0 in R0s:
        N = 200_000
        sizes = simulate_total_progeny(R0, N, max_size=200_000)
        ns = np.arange(5, 400)
        counts = np.bincount(sizes, minlength=ns.max() + 1)
        emp = counts[ns] / N
        mask = emp > 50 / N
        ns_m, emp_m = ns[mask], emp[mask]
        # fit log P = log C + n log rho - (3/2) log n
        y = np.log(emp_m) + 1.5 * np.log(ns_m)
        A = np.vstack([np.ones_like(ns_m, dtype=float), ns_m.astype(float)]).T
        (logC, logrho), *_ = np.linalg.lstsq(A, y, rcond=None)
        C_sim.append(np.exp(logC))
        rho_sim.append(np.exp(logrho))

    axL.plot(R0s, rho_ac, "b-", lw=2, label=r"AC: $\rho=R_0 e^{1-R_0}$")
    axL.plot(R0s, rho_sim, "o", color="crimson", label="Sim fit")
    axL.set_xlabel("$R_0$")
    axL.set_ylabel(r"decay rate $\rho(R_0)$")
    axL.set_title("Panel 2a — subcritical decay rate")
    axL.legend(fontsize=9)
    axL.grid(alpha=0.3)

    axR.plot(R0s, C_ac, "b-", lw=2, label=r"AC: $C=1/(R_0\sqrt{2\pi})$")
    axR.plot(R0s, C_sim, "o", color="crimson", label="Sim fit")
    axR.set_xlabel("$R_0$")
    axR.set_ylabel(r"amplitude $C(R_0)$")
    axR.set_title("Panel 2b — subcritical amplitude (NOT in standard tail bounds)")
    axR.legend(fontsize=9)
    axR.grid(alpha=0.3)


# ---------- Panel 3: scaling collapse near criticality ----------
def panel3(ax):
    # For R0 = 1 - delta, delta small: P(n;R0) ~ (1/sqrt(2 pi n^3)) exp(-n delta^2 / 2)
    # => n^{3/2} P(n;R0) ~ F(xi) with xi = n delta^2, F(xi) = (1/sqrt(2pi)) exp(-xi/2).
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, 5))
    deltas = [0.30, 0.20, 0.12, 0.07, 0.04]

    for delta, c in zip(deltas, colors):
        R0 = 1 - delta
        N = 300_000
        sizes = simulate_total_progeny(R0, N, max_size=200_000)
        ns = np.arange(2, int(15 / delta**2))
        counts = np.bincount(sizes, minlength=ns.max() + 1)
        emp = counts[ns] / N
        mask = emp > 30 / N
        ns_m, emp_m = ns[mask], emp[mask]
        xi = ns_m * delta**2
        yplot = ns_m**1.5 * emp_m
        ax.plot(xi, yplot, "o", ms=3, color=c, alpha=0.7,
                label=rf"$\delta=1-R_0={delta}$")

    xi_grid = np.linspace(0.01, 5, 400)
    F_ac = np.exp(-xi_grid / 2) / np.sqrt(2 * np.pi)
    ax.plot(xi_grid, F_ac, "k-", lw=2.2,
            label=r"AC scaling function $\mathcal{F}(\xi)=\frac{1}{\sqrt{2\pi}}e^{-\xi/2}$")

    # classical "leading" prediction makes no R0 statement at all
    ax.axhline(1 / np.sqrt(2 * np.pi), color="0.6", ls=":",
               label=r"Classical criticality-only value $1/\sqrt{2\pi}$")

    ax.set_xlabel(r"scaling variable $\xi = n(1-R_0)^2$")
    ax.set_ylabel(r"$n^{3/2} P(n; R_0)$")
    ax.set_title("Panel 3 — scaling collapse: five different $R_0$ values fall on one AC curve")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)


# ---------- Panel 4: Aldous Brownian-excursion height collapse ----------
def panel4(ax):
    """At R0=1, conditioned on size n, H_n/sqrt(n) -> 2*max(e). The limit law
    is the same across size bins -- a scaling collapse for an observable
    that the Borel-Tanner pmf is silent about."""
    R0 = 1.0
    N_trials = 1_500_000
    sizes, heights = simulate_size_and_height(R0, N_trials, max_size=20_000)

    bins = [(60, 200), (300, 1000), (1500, 5000)]
    colors = ["tab:purple", "tab:cyan", "tab:orange"]
    bw = 0.12

    edges = np.arange(0, 7 + bw, bw)
    centres = 0.5 * (edges[:-1] + edges[1:])

    for (lo, hi), c in zip(bins, colors):
        sel = (sizes >= lo) & (sizes < hi)
        n_sel = sel.sum()
        if n_sel < 50:
            continue
        y = heights[sel] / np.sqrt(sizes[sel])
        hist, _ = np.histogram(y, bins=edges, density=True)
        ax.step(centres, hist, where="mid", color=c, lw=1.6,
                label=fr"size $\in [{lo},{hi})$ ($N={n_sel:,}$)")

    y_grid = np.linspace(0.6, 6.5, 400)
    f_aldous = density_2maxe(y_grid)
    ax.plot(y_grid, f_aldous, "k-", lw=2.2,
            label=r"AC/Aldous limit: density of $2\max_{[0,1]} e(t)$")

    ax.axvline(np.sqrt(2 * np.pi), color="0.5", ls="--",
               label=fr"mean $= \sqrt{{2\pi}} \approx {np.sqrt(2*np.pi):.3f}$")

    ax.set_xlabel(r"rescaled duration $H_n / \sqrt{n}$")
    ax.set_ylabel("density")
    ax.set_title(r"Panel 4 — duration collapse: $H_n/\sqrt{n} \to 2\max e$ "
                 "(NOT in Borel--Tanner)")
    ax.set_xlim(0.5, 6.0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)


def main():
    fig = plt.figure(figsize=(13, 14.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.1, 1.1],
                          hspace=0.42, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, :])
    ax2L = fig.add_subplot(gs[1, 0])
    ax2R = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])

    panel1(ax1)
    panel2(ax2L, ax2R)
    panel3(ax3)
    panel4(ax4)

    fig.suptitle("SIR / Borel total-progeny: calibration (panels 1-3) + "
                 "shape statistics beyond Borel (panel 4)",
                 fontsize=13, y=0.995)
    out = "simulations/output/sir_ac_demo.png"
    import os
    os.makedirs("simulations/output", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
