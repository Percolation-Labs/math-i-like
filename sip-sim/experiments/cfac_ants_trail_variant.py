"""
A CFAC-COMPLIANT variant of the canonical ant that exhibits spatial
patterns.

Idea: keep state-switching AS-IS (I1–I5 satisfied by rates), and
add a BOUNDED movement bias:
  w_{i→j} ∝ 1 + χ_m · φ_j/(K + φ_j)

Saturating Hill-type bias, derivative bounded by χ_m/K — this is (I1)
compliant.  Linear regime (φ << K) is Keller–Segel-like chemotaxis;
saturation regime (φ >> K) is bounded so no trail-lock.

Expect:
  - At low density + small χ_m: homogeneous random-walk + state-switching
  - At moderate density + moderate χ_m: chemotactic collapse begins
    (KS universality, tunable via χ_m/K)
  - With food-like spatial inhomogeneity: trail patterns can form
    while remaining perturbative

Run three density / coupling combinations and snapshot the result
to demonstrate trails CAN form in CFAC class.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

FIGDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'rdft', 'paper', 'wip', 'figures_ants'))
os.makedirs(FIGDIR, exist_ok=True)


def run_saturating_bias(n=0.10, L=80, k_plus=0.01, k_minus=0.05,
                          chi=1.0, delta=2.0, lam=0.10, sigma=0.005,
                          chi_m=0.0, K=1.0, n_steps=3000, seed=42,
                          food_sources=None):
    """
    Canonical ant with ADDITIONAL bounded chemotactic bias on
    the S-state ants (searchers follow φ gradient, mildly).
    food_sources: list of (r, c, rate) for localised R-deposition.
    """
    rng = np.random.default_rng(seed)
    occ = rng.random((L, L)) < n
    is_R = rng.random((L, L)) < 0.5
    is_R = is_R & occ
    is_S = occ & ~is_R
    phi = np.zeros((L, L), dtype=np.float64)

    def laplacian(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    def neighbour_phis(phi):
        """Return 4 arrays: φ at up/down/left/right neighbours."""
        return (np.roll(phi, 1, 0), np.roll(phi, -1, 0),
                np.roll(phi, 1, 1), np.roll(phi, -1, 1))

    # Food sources: fixed-position R-generation (mimics real foraging)
    source_rate = np.zeros((L, L))
    if food_sources:
        for r, c, rate in food_sources:
            source_rate[r, c] = rate

    for t in range(n_steps):
        # State transitions
        rate_SR = k_plus * (1.0 + chi * phi)
        p_SR = np.clip(rate_SR, 0.0, 1.0)
        convert_SR = is_S & (rng.random((L, L)) < p_SR)
        convert_RS = is_R & (rng.random((L, L)) < k_minus)
        # Food source: spontaneous R-emergence at fixed points
        if food_sources:
            spontaneous = (rng.random((L, L)) < source_rate) & ~(is_R | is_S)
            # Place R at these cells (creating a slow influx of ants)
        else:
            spontaneous = np.zeros((L, L), dtype=bool)

        new_R = (is_R & ~convert_RS) | convert_SR | spontaneous
        new_S = (is_S & ~convert_SR) | convert_RS
        is_R, is_S = new_R, new_S

        # Movement with Hill-saturating bias (for BOTH S and R,
        # but only non-empty cells move)
        # Each occupied cell moves to one of 4 neighbours with
        # probability proportional to 1 + χ_m φ/(K+φ) at that neighbour
        if chi_m > 0:
            occ = is_R | is_S
            phi_up, phi_dn, phi_lf, phi_rt = neighbour_phis(phi)
            # Hill bias at each neighbour
            bias = lambda p: 1.0 + chi_m * p / (K + p)
            w_up = bias(phi_up); w_dn = bias(phi_dn)
            w_lf = bias(phi_lf); w_rt = bias(phi_rt)
            w_tot = w_up + w_dn + w_lf + w_rt
            # Cumulative probabilities
            c_up = w_up / w_tot
            c_dn = c_up + w_dn / w_tot
            c_lf = c_dn + w_lf / w_tot
            u = rng.random((L, L))
            # Direction per cell
            dir_up = occ & (u < c_up)
            dir_dn = occ & (u >= c_up) & (u < c_dn)
            dir_lf = occ & (u >= c_dn) & (u < c_lf)
            dir_rt = occ & (u >= c_lf)
            # Attempted new positions: use roll to move everything
            # For simplicity, process one direction at a time with collisions
            # handled by "move-or-stay" Markov step. We approximate with
            # a small hop probability (random walk rate 0.5 per step).
            hop_prob = 0.3
            move_mask = rng.random((L, L)) < hop_prob

            # Collect destinations
            new_is_R = np.zeros_like(is_R)
            new_is_S = np.zeros_like(is_S)
            # Simple approach: for each cell, if moving, place at destination
            # if destination is empty; else stay. Do sequentially for directions.
            # This is an approximation; sufficient for visualisation.

            # Direction arrays of destination indices
            for (dr, dc, mask) in [(-1, 0, dir_up), (1, 0, dir_dn),
                                     (0, -1, dir_lf), (0, 1, dir_rt)]:
                moving = mask & move_mask
                # Destinations
                dst_r_is_R = np.roll(moving & is_R, dr, axis=0)
                dst_r_is_R = np.roll(dst_r_is_R, dc, axis=1)
                dst_s_is_S = np.roll(moving & is_S, dr, axis=0)
                dst_s_is_S = np.roll(dst_s_is_S, dc, axis=1)
                # Accept only if destination empty
                dest_empty = ~(is_R | is_S | new_is_R | new_is_S)
                place_R = dst_r_is_R & dest_empty
                place_S = dst_s_is_S & dest_empty
                new_is_R |= place_R
                new_is_S |= place_S
                # Remove source cells whose move succeeded
                source_success_R = np.roll(place_R, -dr, axis=0)
                source_success_R = np.roll(source_success_R, -dc, axis=1)
                source_success_S = np.roll(place_S, -dr, axis=0)
                source_success_S = np.roll(source_success_S, -dc, axis=1)
                is_R = is_R & ~source_success_R
                is_S = is_S & ~source_success_S
            # Non-movers stay
            is_R = is_R | new_is_R
            is_S = is_S | new_is_S

        # Field dynamics
        phi = phi + delta * is_R.astype(np.float64)
        phi = phi * (1.0 - lam)
        phi = phi + sigma * laplacian(phi)
        phi = np.clip(phi, 0, 1e6)

    return {
        'is_R': is_R.copy(),
        'is_S': is_S.copy(),
        'phi': phi.copy(),
        'n': n,
        'chi_m': chi_m,
        'K': K,
    }


def main():
    print("Running CFAC-compliant trail variants...")

    # Parameters: saturating Hill bias — bounded, so (I1) OK
    configs = [
        ('No bias (canonical)', dict(chi_m=0.0, K=1.0, n=0.10,
                                       sigma=0.02, n_steps=2500)),
        ('Moderate Hill bias', dict(chi_m=3.0, K=0.5, n=0.10,
                                       sigma=0.02, n_steps=2500)),
        ('Strong Hill bias\n(saturating, still bounded)',
         dict(chi_m=10.0, K=0.1, n=0.10, sigma=0.02, n_steps=2500)),
    ]

    results = []
    for label, cfg in configs:
        print(f"  {label}... χ_m = {cfg['chi_m']}")
        r = run_saturating_bias(**cfg, seed=42)
        results.append((label, r))
        n_R = int(r['is_R'].sum())
        n_S = int(r['is_S'].sum())
        print(f"    n_R = {n_R}, n_S = {n_S}, "
              f"f_R = {n_R/(n_R+n_S):.3f}, "
              f"<φ> = {r['phi'].mean():.2f}, max φ = {r['phi'].max():.1f}")

    # Figure: side-by-side snapshots
    fig, axes = plt.subplots(2, len(configs), figsize=(4.5*len(configs), 7),
                              constrained_layout=True)

    for col, (label, res) in enumerate(results):
        # Agents
        ax = axes[0, col]
        L = res['is_R'].shape[0]
        rgb = np.ones((L, L, 3)) * 0.94
        rgb[res['is_R']] = [0.85, 0.2, 0.2]
        rgb[res['is_S']] = [0.2, 0.4, 0.85]
        ax.imshow(rgb, origin='upper', interpolation='nearest')
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Agents (red = R, blue = S)')

        # Field
        ax = axes[1, col]
        phi = res['phi']
        im = ax.imshow(phi, cmap='hot', origin='upper',
                        interpolation='nearest',
                        vmin=0, vmax=max(phi.max(), 0.5))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$\langle\phi\rangle$={phi.mean():.2f}   "
                      rf"max={phi.max():.1f}", fontsize=9)
        if col == 0:
            ax.set_ylabel(r'Pheromone $\phi$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle('CFAC-compliant bias variants: Hill saturation '
                 r'$w \propto 1 + \chi_m \phi/(K+\phi)$ keeps (I1) and '
                 'can produce spatial concentration', fontsize=11)
    out = os.path.join(FIGDIR, 'canonical_ant_trail_variant.pdf')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
