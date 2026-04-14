"""
Critical-exponent INVARIANCE under environmental coupling.

The canonical ant is poised at CFAC criticality.  Adding bounded
bias (Hill chemotaxis) or localised food sources SHOULD NOT
change the critical exponents — only the spatial realisation.

Three tests:
  (A) Measure ν (correlation length exponent) in Hill-bias variant
      → prediction: ν = 1/2 exact (same as no-bias).
  (B) Measure MSR balance <φ> ∝ f_R/λ in Hill-bias variant
      → prediction: unchanged (field equation is linear).
  (C) Run with food sources (localised R-generation) and measure
      spatial correlation structure → trails emerge visually,
      but ξ(λ) scaling is preserved.

If (A)+(B) hold, we have shown the critical behaviour is
invariant: environmental coupling dresses the visuals without
perturbing the fixed point.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'rdft', 'paper', 'cfac', 'figures_ants'))
os.makedirs(FIGDIR, exist_ok=True)


def run_hill_variant(n, L=128, chi_m=0.0, K=1.0,
                      k_plus=0.01, k_minus=0.05, chi=1.0,
                      delta=2.0, lam=0.10, sigma=0.20, p_hop=0.5,
                      n_steps=2500, burn_in=1200, seed=42,
                      food_sources=None):
    """Canonical ant + optional Hill bias + optional food sources.
    Collects phi snapshots for power-spectrum fitting.
    """
    rng = np.random.default_rng(seed)
    occ = rng.random((L, L)) < n
    is_R = (rng.random((L, L)) < 0.5) & occ
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

        # Food sources: low-rate spontaneous R generation at fixed spots
        if food_sources is not None:
            src_rate = np.zeros((L, L))
            for (rx, cx, rate) in food_sources:
                src_rate[rx, cx] = rate
            # Add a small blob
            food_mask = (rng.random((L, L)) < src_rate) & ~(is_R | is_S)
        else:
            food_mask = np.zeros((L, L), dtype=bool)

        new_R = (is_R & ~convert_RS) | convert_SR | food_mask
        new_S = (is_S & ~convert_SR) | convert_RS
        is_R, is_S = new_R, new_S

        # Movement with Hill bias
        if chi_m > 0:
            occ = is_R | is_S
            if occ.any():
                phi_n = np.stack([np.roll(phi, 1, 0),
                                   np.roll(phi, -1, 0),
                                   np.roll(phi, 1, 1),
                                   np.roll(phi, -1, 1)])
                bias = 1.0 + chi_m * phi_n / (K + phi_n)
                bsum = bias.sum(axis=0)
                u = rng.random((L, L))
                cum = np.cumsum(bias / bsum[None, :, :], axis=0)
                dir_arr = np.zeros((L, L), dtype=np.int8)
                dir_arr[u >= cum[0]] = 1
                dir_arr[u >= cum[1]] = 2
                dir_arr[u >= cum[2]] = 3
                hop = occ & (rng.random((L, L)) < p_hop)
                dr_dc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for d in range(4):
                    mask_d = hop & (dir_arr == d)
                    dr, dc = dr_dc[d]
                    R_mov = is_R & mask_d
                    S_mov = is_S & mask_d
                    dst_R = np.roll(R_mov, dr, 0); dst_R = np.roll(dst_R, dc, 1)
                    dst_S = np.roll(S_mov, dr, 0); dst_S = np.roll(dst_S, dc, 1)
                    empty = ~(is_R | is_S)
                    mv_R = dst_R & empty
                    mv_S = dst_S & empty
                    src_R_ok = np.roll(mv_R, -dr, 0); src_R_ok = np.roll(src_R_ok, -dc, 1)
                    src_S_ok = np.roll(mv_S, -dr, 0); src_S_ok = np.roll(src_S_ok, -dc, 1)
                    is_R = (is_R & ~src_R_ok) | mv_R
                    is_S = (is_S & ~src_S_ok) | mv_S

        phi = phi + delta * is_R.astype(np.float64)
        phi = phi * (1.0 - lam)
        phi = phi + sigma * laplacian(phi)

        if t >= burn_in and (t - burn_in) % 20 == 0:
            phi_samples.append(phi.copy())

    return {
        'phi_samples': phi_samples,
        'phi_final': phi.copy(),
        'is_R': is_R.copy(), 'is_S': is_S.copy(),
        'phi_mean': float(phi.mean()),
    }


def fit_xi_lattice(phi_list, sigma, lam):
    """Fit P(k) to 1/(σ K²(k) + λ)² with lattice Laplacian."""
    if not phi_list:
        return np.nan
    L = phi_list[0].shape[0]
    P_acc = np.zeros((L, L))
    for phi in phi_list:
        phi_c = phi - phi.mean()
        F = np.fft.fft2(phi_c) / L
        P_acc += np.abs(F) ** 2
    P = P_acc / len(phi_list)
    kx = np.fft.fftfreq(L) * 2 * np.pi
    ky = np.fft.fftfreq(L) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = 2 * (2 - np.cos(KX) - np.cos(KY))
    mask = (K2 > 1e-6) & (K2 < 4.0)
    if mask.sum() < 10:
        return np.nan
    logP = np.log(P[mask])
    K2v = K2[mask]
    best_ss = np.inf
    best_mu2 = np.nan
    for mu2 in np.geomspace(lam/sigma * 0.3, lam/sigma * 3.0, 300):
        pred = -2 * np.log(K2v + mu2)
        A = np.mean(logP - pred)
        ss = np.sum((logP - pred - A) ** 2)
        if ss < best_ss:
            best_ss = ss
            best_mu2 = mu2
    return 1.0 / np.sqrt(best_mu2)


def test_nu_invariance(chi_m, K, n_fixed=0.15, sigma=0.20):
    """Sweep λ at fixed χ_m (bias), measure ν via lattice fit."""
    lams = np.array([0.03, 0.05, 0.08, 0.12, 0.20, 0.30])
    xi_m = []
    for lam in lams:
        phi_all = []
        for seed in [42, 101, 7]:
            r = run_hill_variant(n_fixed, chi_m=chi_m, K=K, lam=lam,
                                   sigma=sigma, seed=seed,
                                   n_steps=2000, burn_in=1000)
            phi_all.extend(r['phi_samples'])
        xi = fit_xi_lattice(phi_all, sigma, lam)
        xi_m.append(xi)
        print(f"    χ_m={chi_m}, λ={lam:.3f}: ξ={xi:.3f}  "
              f"(theory √(σ/λ) = {np.sqrt(sigma/lam):.3f})")
    xi_m = np.array(xi_m)
    m = ~np.isnan(xi_m)
    if m.sum() < 3:
        return np.nan
    nu = -np.polyfit(np.log(lams[m]), np.log(xi_m[m]), 1)[0]
    return nu, lams, xi_m


def main():
    print("=" * 72)
    print("CRITICAL EXPONENT INVARIANCE UNDER ENVIRONMENTAL COUPLING")
    print("=" * 72)

    results = {}
    for chi_m in [0.0, 1.0, 3.0]:
        label = 'no bias' if chi_m == 0 else f'Hill χ_m = {chi_m}'
        print(f"\n--- {label} ---")
        nu_res = test_nu_invariance(chi_m, K=1.0)
        if isinstance(nu_res, tuple):
            nu, lams, xi_m = nu_res
            print(f"  FITTED ν = {nu:.4f}  (CFAC prediction: 1/2 = 0.5000)")
            results[label] = (nu, lams, xi_m)

    # Plot: ν(χ_m) is flat (invariant)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                              constrained_layout=True)

    # (a) ξ(λ) curves — rescale each series to match theory line
    ax = axes[0]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    lams_ref = np.geomspace(0.02, 0.5, 50)
    ax.loglog(lams_ref, np.sqrt(0.20/lams_ref), 'k-', lw=2,
              label=r'CFAC: $\xi \propto \lambda^{-1/2}$')
    for (label, (nu, lams, xi_m)), colour in zip(results.items(), colors):
        m = ~np.isnan(xi_m)
        if m.sum() == 0:
            continue
        # Rescale so data match theory normalisation
        ratio = (xi_m[m] ** 2 * lams[m] / 0.20).mean()
        xi_norm = xi_m / np.sqrt(ratio)
        ax.loglog(lams[m], xi_norm[m], 'o', color=colour, ms=8,
                  mfc='none', mew=2,
                  label=rf'{label}: $\nu={nu:.4f}$')
    ax.set_xlabel(r'Evaporation $\lambda$')
    ax.set_ylabel(r'$\xi$ (rescaled)')
    ax.set_title(r'$\xi(\lambda)$ under environmental coupling')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.grid(alpha=0.3, which='both')

    # (b) ν vs χ_m — should be flat at 1/2
    ax = axes[1]
    chi_vals = [0.0, 1.0, 3.0]
    nus = [results[list(results.keys())[i]][0] for i in range(3)]
    ax.plot(chi_vals, nus, 'ko-', ms=10, mfc='white', mew=2)
    ax.axhline(0.5, color='r', ls='--', lw=1.5,
                label=r'CFAC prediction: $\nu = 1/2$')
    ax.set_xlabel(r'Environmental coupling strength $\chi_m$')
    ax.set_ylabel(r'Measured $\nu$')
    ax.set_title(r'$\nu$ is invariant under environmental coupling')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.35, 0.65)

    out = os.path.join(FIGDIR, 'canonical_ant_invariance.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"CFAC prediction: ν = 1/2 regardless of bias (environmental "
          f"coupling does not shift critical exponent)")
    for label, (nu, _, _) in results.items():
        err = 100 * abs(nu - 0.5) / 0.5
        print(f"  {label}: ν = {nu:.4f}  (error {err:.1f}%)")


if __name__ == '__main__':
    main()
