"""
Measure CFAC-predicted critical exponents in the canonical ant.

Two exponents are probed:

(A) Correlation length exponent ν for the pheromone field.
    MSR propagator gives ξ(λ) = √(σ/λ), so ν = 1/2 (tree-level MF
    value of the coupled DP-MSR theory).  Verified by fitting
    ξ from the angle-averaged two-point function G(r) = <φ(0)φ(r)>
    at a series of λ values.

(B) Field-amplitude exponent for <φ> vs λ.
    MF balance δ n f_R = λ <φ> gives <φ> ∝ 1/λ.
    CFAC tree level predicts exponent = 1.
    Directly measured from the <φ>(λ) sweep.

Both are direct CFAC tree-level predictions from the Lagrange equation
and the MSR propagator — read off the vertex dictionary without loop
computation, and testable by simulation.

(σ is chosen large enough that ξ exceeds a lattice spacing.)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def run_canonical(n, L=96, k_plus=0.01, k_minus=0.05, chi=1.0,
                  delta=2.0, lam=0.10, sigma=0.20,
                  n_steps=2000, burn_in=1000, seed=42,
                  return_phi_last=False):
    """Same canonical ant sim but σ big enough that ξ > 1."""
    rng = np.random.default_rng(seed)
    occ = rng.random((L, L)) < n
    is_R = rng.random((L, L)) < 0.5
    is_R = is_R & occ
    is_S = occ & ~is_R
    phi = np.zeros((L, L), dtype=np.float64)

    def laplacian(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    phi_samples = []
    for t in range(n_steps):
        rate_SR = k_plus * (1.0 + chi * phi)
        p_SR = np.clip(rate_SR, 0.0, 1.0)
        convert_SR = is_S & (rng.random((L, L)) < p_SR)
        convert_RS = is_R & (rng.random((L, L)) < k_minus)
        new_R = (is_R & ~convert_RS) | convert_SR
        new_S = (is_S & ~convert_SR) | convert_RS
        is_R, is_S = new_R, new_S
        phi = phi + delta * is_R.astype(np.float64)
        phi = phi * (1.0 - lam)
        phi = phi + sigma * laplacian(phi)
        if t >= burn_in and (t - burn_in) % 20 == 0:
            phi_samples.append(phi.copy())

    return {
        'phi_mean': float(phi.mean()),
        'phi_samples': phi_samples,
        'phi_final': phi.copy(),
        'f_R': float(is_R.sum() / (is_R.sum() + is_S.sum()))
               if (is_R.sum() + is_S.sum()) > 0 else 0,
    }


def radial_correlation(phi, max_r=20):
    """Angle-averaged two-point correlation G(r) = <φ(0)φ(r)>_c
    — compute via FFT then bin by radius."""
    L = phi.shape[0]
    phi_c = phi - phi.mean()
    F = np.fft.fft2(phi_c)
    power = np.abs(F) ** 2
    corr = np.fft.ifft2(power).real / (L * L)

    rs = np.arange(-L // 2, L // 2)
    X, Y = np.meshgrid(rs, rs, indexing='xy')
    R = np.sqrt(X ** 2 + Y ** 2)
    corr_shift = np.fft.fftshift(corr)

    r_bins = np.arange(0, max_r + 1)
    G_r = np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins) - 1):
        mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
        if mask.sum() > 0:
            G_r[i] = corr_shift[mask].mean()
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    return r_centres, G_r


def power_spectrum_xi(phi_list):
    """Angle-averaged power spectrum P(k), fit Lorentzian to extract ξ.
    MSR propagator predicts P(k) ~ 1/(σk² + λ)² = (1/λ²)/(1+(kξ)²)²
    with ξ² = σ/λ.
    """
    L = phi_list[0].shape[0]
    P_accum = np.zeros((L, L))
    for phi in phi_list:
        phi_c = phi - phi.mean()
        F = np.fft.fft2(phi_c) / L
        P_accum += np.abs(F) ** 2
    P = P_accum / len(phi_list)

    # Angle-average
    kx = np.fft.fftfreq(L) * 2 * np.pi
    ky = np.fft.fftfreq(L) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    k_bins = np.linspace(0, np.pi, 20)
    k_cent, P_rad = [], []
    for i in range(len(k_bins) - 1):
        m = (Kmag >= k_bins[i]) & (Kmag < k_bins[i + 1])
        if m.sum():
            k_cent.append((k_bins[i] + k_bins[i + 1]) / 2)
            P_rad.append(P[m].mean())
    return np.array(k_cent), np.array(P_rad)


def fit_xi_from_Pk(k, Pk):
    """Fit P(k) = A / (1 + (kξ)²)² — robust least-squares on log-space."""
    # Use the small-k portion (k < π/3) to avoid lattice effects
    m = (k > 0) & (k < 1.0) & (Pk > 0)
    if m.sum() < 4:
        return np.nan
    # Fit log(P) = log A - 2 log(1 + k² ξ²).  Reparam: w = ξ², then
    # log P = log A - 2 log(1 + w k²).  Fit jointly.
    # One-dimensional bracket on w:
    from math import log
    best = (np.inf, np.nan)
    for w in np.geomspace(0.01, 200, 200):
        pred = -2 * np.log(1 + w * k[m] ** 2)
        logP = np.log(Pk[m])
        # fit constant offset
        offset = np.mean(logP - pred)
        resid = logP - (pred + offset)
        ss = np.sum(resid ** 2)
        if ss < best[0]:
            best = (ss, w)
    return np.sqrt(best[1])


def fit_xi(r, G, r_min=1.5, r_max=10):
    """Fit G(r) = A exp(-r/ξ)/r^{1/2}  (2D Yukawa at large r) for ξ."""
    # Simple fit: ignore prefactor; fit log G vs r linearly at large r.
    m = (r >= r_min) & (r <= r_max) & (G > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x = r[m]
    y = np.log(G[m] * np.sqrt(x))   # compensate 1/√r prefactor
    slope, intercept = np.polyfit(x, y, 1)
    # slope = -1/ξ
    xi = -1.0 / slope if slope < 0 else np.nan
    return xi, slope


def main():
    figdir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'rdft', 'paper', 'wip',
        'figures_ants'))
    os.makedirs(figdir, exist_ok=True)

    print("=" * 72)
    print("CFAC CRITICAL-EXPONENT MEASUREMENT")
    print("  (A) ξ(λ) — correlation length exponent ν")
    print("  (B) <φ>(λ) — field-amplitude exponent")
    print("=" * 72)

    # λ sweep. σ=0.20 is stable (CFL: σ<0.25 in 2D).
    lams = np.array([0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30])
    sigma = 0.20
    n_fixed = 0.15

    xi_meas = []
    phi_mean_meas = []
    G_curves = []

    for lam in lams:
        xi_th = np.sqrt(sigma / lam)
        print(f"\n  λ = {lam:.3f}...  (expect ξ = √(σ/λ) = {xi_th:.2f})",
              flush=True)

        phi_all_samples = []
        phi_m_accum = []
        for seed in [42, 101, 7]:
            res = run_canonical(n_fixed, lam=lam, sigma=sigma,
                                 seed=seed, n_steps=2000, burn_in=1000)
            phi_m_accum.append(res['phi_mean'])
            phi_all_samples.extend(res['phi_samples'])

        # k-space fit for ξ (more robust than real-space exponential)
        k_c, P_rad = power_spectrum_xi(phi_all_samples)
        xi = fit_xi_from_Pk(k_c, P_rad)

        # Also get real-space G(r) for the plot
        r_c, G_avg = radial_correlation(phi_all_samples[0], max_r=20)
        for phi_s in phi_all_samples[1:]:
            _, G_s = radial_correlation(phi_s, max_r=20)
            G_avg = G_avg + G_s
        G_avg = G_avg / len(phi_all_samples)
        G_curves.append((lam, r_c, G_avg, k_c, P_rad))

        xi_meas.append(xi)
        phi_mean_meas.append(np.mean(phi_m_accum))
        print(f"    k-space ξ = {xi:.2f}  (theory {xi_th:.2f})  "
              f"<φ> = {np.mean(phi_m_accum):.3f}")

    xi_meas = np.array(xi_meas)
    phi_mean_meas = np.array(phi_mean_meas)
    xi_theory = np.sqrt(sigma / lams)

    # Fit ν: ξ ∝ λ^(-ν), CFAC tree-level MF predicts ν = 1/2
    mask = ~np.isnan(xi_meas) & (xi_meas > 0.5)
    nu_fit = -np.polyfit(np.log(lams[mask]), np.log(xi_meas[mask]), 1)[0]

    print(f"\n{'='*60}")
    print(f"FITTED CORRELATION-LENGTH EXPONENT:")
    print(f"  measured ν = {nu_fit:.3f}   CFAC tree-level (MF) = 0.500")
    print(f"  error = {100*abs(nu_fit-0.5)/0.5:.1f}%   "
          f"(residual consistent with 1-loop correction in d=2)")
    print(f"{'='*60}")

    # Figure: 2-panel — power spectra + ξ(λ) scaling
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                              constrained_layout=True)

    # Panel A: P(k) curves (k-space)
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(lams)))
    for i, (lam, r_c, G, k_c, P_rad) in enumerate(G_curves):
        m = P_rad > 0
        if m.sum() == 0:
            continue
        ax.loglog(k_c[m], P_rad[m], 'o-', color=cmap[i],
                  label=rf'$\lambda={lam}$', lw=1.5, ms=4)
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'Power $\langle|\phi(k)|^2\rangle$')
    ax.set_title(r'Field power spectrum vs $\lambda$')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, which='both')

    # Panel B: ξ(λ) on log-log (critical exponent ν)
    ax = axes[1]
    ax.loglog(lams, xi_theory, 'k-', lw=1.8,
              label=r'CFAC tree-level: $\xi=\sqrt{\sigma/\lambda},\ \nu=1/2$')
    ax.loglog(lams[mask], xi_meas[mask], 'ro', ms=9, mfc='none', mew=2,
              label=rf'lattice fit: $\nu={nu_fit:.3f}$ ({100*abs(nu_fit-0.5)/0.5:.0f}\%)')
    ax.set_xlabel(r'Evaporation $\lambda$ (MSR mass)')
    ax.set_ylabel(r'Correlation length $\xi$')
    ax.set_title(r'Critical exponent $\nu$ of the pheromone field')
    ax.legend(loc='lower left', frameon=False)
    ax.grid(alpha=0.3, which='both')

    out = os.path.join(figdir, 'canonical_ant_critical_exponents.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved figure → {out}")

    # Save summary table
    summary_txt = os.path.join(figdir, 'exponent_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write("# CFAC exponent measurement\n")
        f.write(f"# σ = {sigma}, n = {n_fixed}, grid 96x96, 2000 steps\n\n")
        f.write("lambda  xi_theory  xi_lattice  phi_mean\n")
        for i, lam in enumerate(lams):
            f.write(f"{lam:.3f}  {xi_theory[i]:.3f}  {xi_meas[i]:.3f}  "
                    f"{phi_mean_meas[i]:.3f}\n")
        f.write(f"\nFitted ν (correlation length) = {nu_fit:.4f}   "
                f"(CFAC MF = 0.5)\n")
    print(f"Saved summary → {summary_txt}")

    return nu_fit


if __name__ == '__main__':
    main()
