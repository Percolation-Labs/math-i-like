"""
Measure Keller–Segel chemotactic collapse in the Hill-bias
CFAC-compliant canonical ant.

Model:
  S ⇌ R state transitions as canonical (rates k±, etc.).
  Movement bias: w_{i→j} ∝ 1 + χ_m φ_j / (K + φ_j)   (Hill, bounded)
  Field φ: deposited by R, decays λ, diffuses σ.

KS prediction:
  Uniform density n is linearly unstable when
     n > n_c = D_ψ λ / (χ_eff δ),
  where χ_eff = χ_m/K is the small-φ linear response coefficient
  and D_ψ = p_hop / 4 is the ant diffusion constant.

  Above n_c: ants aggregate (concentration ratio grows).
  Below n_c: uniform distribution.

We sweep n and measure clustering as:
  B(n) = Var(N_block) / <N_block>²   (block variance normalised)
Poisson baseline: B = 1/<N_block>.  Clustering: B >> Poisson.

If KS class holds, we expect:
  Below n_c: B ~ 1/n (Poisson)
  Above n_c: B > Poisson, grows with n
  Transition near n_c.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'rdft', 'paper', 'wip', 'figures_ants'))
os.makedirs(FIGDIR, exist_ok=True)


def run_hill_bias_ant(n, L=80, chi_m=0.0, K=0.5,
                       k_plus=0.01, k_minus=0.05, chi=1.0,
                       delta=2.0, lam=0.10, sigma=0.005, p_hop=0.3,
                       n_steps=4000, burn_in=2000, seed=42):
    """Canonical ant with Hill-saturating movement bias.

    Simple movement rule:
      For each occupied cell, hop with prob p_hop.
      If hopping, pick direction in {up, down, left, right} with
        probability ∝ 1 + χ_m φ_neighbour / (K + φ_neighbour).
      If destination empty, move; else stay.
    """
    rng = np.random.default_rng(seed)
    occ = rng.random((L, L)) < n
    is_R = (rng.random((L, L)) < 0.5) & occ
    is_S = occ & ~is_R
    phi = np.zeros((L, L), dtype=np.float64)

    def laplacian(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    concentrations = []  # sample every N steps after burn-in

    for t in range(n_steps):
        # State transitions
        rate_SR = k_plus * (1.0 + chi * phi)
        p_SR = np.clip(rate_SR, 0.0, 1.0)
        convert_SR = is_S & (rng.random((L, L)) < p_SR)
        convert_RS = is_R & (rng.random((L, L)) < k_minus)
        new_R = (is_R & ~convert_RS) | convert_SR
        new_S = (is_S & ~convert_SR) | convert_RS
        is_R, is_S = new_R, new_S

        # Movement
        occ = is_R | is_S
        if occ.any():
            # Biased direction probabilities at each cell from its 4 neighbours
            phi_n = np.stack([np.roll(phi, 1, 0),     # up (from i+1)
                              np.roll(phi, -1, 0),    # down
                              np.roll(phi, 1, 1),     # left
                              np.roll(phi, -1, 1)])   # right
            # bias at neighbour
            bias = 1.0 + chi_m * phi_n / (K + phi_n)   # shape (4, L, L)
            bias_sum = bias.sum(axis=0)
            # Pick direction for each cell:
            u = rng.random((L, L))
            cum = np.cumsum(bias / bias_sum[None, :, :], axis=0)
            # Manual: dir = 0 if u<cum[0], 1 if u<cum[1], 2 if u<cum[2], 3 else
            dir_arr = np.zeros((L, L), dtype=np.int8)
            dir_arr[u >= cum[0]] = 1
            dir_arr[u >= cum[1]] = 2
            dir_arr[u >= cum[2]] = 3

            # Hop decision
            hop = occ & (rng.random((L, L)) < p_hop)

            # Execute: process each direction sequentially
            dr_dc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for d in range(4):
                mask_d = hop & (dir_arr == d)
                dr, dc = dr_dc[d]
                # Compute destinations
                R_movers = is_R & mask_d
                S_movers = is_S & mask_d
                # Destination cells
                dst_R = np.roll(R_movers, dr, axis=0)
                dst_R = np.roll(dst_R, dc, axis=1)
                dst_S = np.roll(S_movers, dr, axis=0)
                dst_S = np.roll(dst_S, dc, axis=1)
                # Only move to currently-empty cells
                dest_empty = ~(is_R | is_S)
                moved_R = dst_R & dest_empty
                moved_S = dst_S & dest_empty
                # Find source cells whose move succeeded
                src_R_success = np.roll(moved_R, -dr, axis=0)
                src_R_success = np.roll(src_R_success, -dc, axis=1)
                src_S_success = np.roll(moved_S, -dr, axis=0)
                src_S_success = np.roll(src_S_success, -dc, axis=1)
                # Update
                is_R = (is_R & ~src_R_success) | moved_R
                is_S = (is_S & ~src_S_success) | moved_S

        # Field dynamics
        phi = phi + delta * is_R.astype(np.float64)
        phi = phi * (1.0 - lam)
        phi = phi + sigma * laplacian(phi)

        # Sample concentration every 50 steps
        if t >= burn_in and t % 50 == 0:
            # Block variance: divide lattice into 5x5 blocks
            bs = 5
            bh = L // bs
            ant = (is_R | is_S).astype(float)
            ant_block = ant[:bh*bs, :bh*bs].reshape(bh, bs, bh, bs).sum(axis=(1, 3))
            mean_b = ant_block.mean()
            var_b = ant_block.var()
            if mean_b > 0.1:
                B = var_b / (mean_b * mean_b)
            else:
                B = 0.0
            concentrations.append((B, mean_b, ant.mean(),
                                    float(phi.max()/phi.mean()) if phi.mean()>0.01 else 1.0))

    if not concentrations:
        return dict(n=n, B=0, B_poisson=0, mean_b=0, conc_phi=1.0)
    arr = np.array(concentrations)
    return dict(
        n=n,
        B=float(arr[:, 0].mean()),
        B_std=float(arr[:, 0].std()),
        mean_b=float(arr[:, 1].mean()),
        density=float(arr[:, 2].mean()),
        conc_phi=float(arr[:, 3].mean()),
    )


def main():
    print("=" * 72)
    print("KS COLLAPSE: BIAS vs NO-BIAS CFAC-COMPLIANT ANT")
    print("=" * 72)

    # Parameters: reduce δ and χ_eff to put n_c in middle of sweep
    K = 1.0
    delta, lam, sigma = 0.5, 0.10, 0.005
    p_hop = 0.5
    D_psi = p_hop / 4   # 0.125

    # Two configs: no bias (control) and moderate Hill bias
    configs = [
        ('no bias', 0.0),
        ('Hill bias χ_m=1', 1.0),
        ('Hill bias χ_m=3', 3.0),
    ]
    ns = np.array([0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.25])

    all_results = {}
    for label, chi_m in configs:
        chi_eff = chi_m / K if chi_m > 0 else 0
        n_c_KS = (D_psi * lam / (chi_eff * delta)) if chi_eff > 0 else np.inf
        print(f"\n--- {label} (χ_eff = {chi_eff}, n_c_KS = {n_c_KS:.3f}) ---")
        print(f"{'n':>7s} {'B':>8s} {'B_pois':>8s} {'B_exc':>8s}")
        res = []
        for n in ns:
            r = run_hill_bias_ant(n, chi_m=chi_m, K=K, delta=delta,
                                   lam=lam, sigma=sigma, p_hop=p_hop,
                                   n_steps=2500, burn_in=1200, seed=42)
            B_poisson = 1.0 / r['mean_b'] if r['mean_b'] > 0.1 else 0
            B_excess = r['B'] - B_poisson
            print(f"{n:>7.3f} {r['B']:>8.4f} {B_poisson:>8.4f} "
                  f"{B_excess:>8.4f}")
            res.append(dict(n=n, B=r['B'], B_pois=B_poisson,
                             B_excess=B_excess,
                             conc_phi=r['conc_phi']))
        all_results[label] = (res, n_c_KS)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                              constrained_layout=True)
    colors = ['#7f8c8d', '#3498db', '#e74c3c']

    ax = axes[0]
    for (label, _), colour in zip(configs, colors):
        res, n_c_KS = all_results[label]
        ns_arr = np.array([r['n'] for r in res])
        Bs = np.array([r['B'] for r in res])
        Bpois = np.array([r['B_pois'] for r in res])
        lbl = label
        if np.isfinite(n_c_KS):
            lbl += rf' ($n_c={n_c_KS:.3f}$)'
        ax.loglog(ns_arr, Bs, 'o-', color=colour, ms=7, mfc='none',
                  mew=1.8, label=lbl)
    ax.loglog(ns_arr, Bpois, 'k:', lw=1.3, label='Poisson baseline')
    ax.set_xlabel(r'Ant density $n$')
    ax.set_ylabel(r'Block variance $B$')
    ax.set_title('Block variance: bias increases clustering')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    for (label, _), colour in zip(configs, colors):
        res, n_c_KS = all_results[label]
        ns_arr = np.array([r['n'] for r in res])
        Be = np.array([r['B_excess'] for r in res])
        ax.semilogx(ns_arr, Be, 'o-', color=colour, ms=7, mfc='none',
                    mew=1.8, label=label)
        if np.isfinite(n_c_KS):
            ax.axvline(n_c_KS, color=colour, ls=':', lw=1.2, alpha=0.6)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'Ant density $n$')
    ax.set_ylabel(r'Clustering excess $B - B_{\rm Poisson}$')
    ax.set_title(r'Clustering vs CFAC-KS prediction $n_c$')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    out = os.path.join(FIGDIR, 'canonical_ant_ks_collapse.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")

    # Summary
    print("\n=== SUMMARY ===")
    for (label, chi_m) in configs:
        res, n_c_KS = all_results[label]
        Be = np.array([r['B_excess'] for r in res])
        print(f"  {label}: max B_excess = {Be.max():.3f}, "
              f"at n = {res[Be.argmax()]['n']:.3f}  "
              f"(n_c_KS = {n_c_KS:.3f})")

    return all_results


if __name__ == '__main__':
    main()
