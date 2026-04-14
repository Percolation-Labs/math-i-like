"""Snapshots of the engineered B⇌A + field lattice at different densities."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine
from ac_plus_lattice import make_spec


def run_and_snapshot(n_density, seed=42, max_steps=2000):
    spec = SimulationSpec(**make_spec(n_density=n_density, seed=seed,
                                       max_steps=max_steps))
    engine = build_engine(spec)
    engine.run()

    kind_grid = engine.state.kind_grid
    field = engine.state.env_grids['field']

    A_idx = engine.kind_index['active']
    B_idx = engine.kind_index['passive']

    n_A = (kind_grid == A_idx).sum()
    n_B = (kind_grid == B_idx).sum()
    f_A = n_A / (n_A + n_B) if (n_A + n_B) > 0 else 0

    return {
        'n_density': n_density,
        'kind_grid': kind_grid.copy(),
        'field': field.copy(),
        'n_A': n_A,
        'n_B': n_B,
        'f_A': f_A,
        'A_idx': A_idx,
        'B_idx': B_idx,
    }


def make_figure(snapshots, outpath):
    n_snap = len(snapshots)
    fig, axes = plt.subplots(2, n_snap, figsize=(3.2 * n_snap, 6),
                              constrained_layout=True)
    if n_snap == 1:
        axes = axes[:, np.newaxis]

    for col, snap in enumerate(snapshots):
        # Top row: agents (A red, B blue, empty grey)
        ax = axes[0, col]
        rgb = np.ones((*snap['kind_grid'].shape, 3)) * 0.92
        A_mask = snap['kind_grid'] == snap['A_idx']
        B_mask = snap['kind_grid'] == snap['B_idx']
        rgb[A_mask] = [0.9, 0.2, 0.2]   # red = A
        rgb[B_mask] = [0.2, 0.4, 0.9]   # blue = B
        ax.imshow(rgb, origin='upper', interpolation='nearest')
        ax.set_title(f"n={snap['n_density']:.2f}\n"
                     f"f_A = {snap['f_A']:.3f}  "
                     f"(A={snap['n_A']}, B={snap['n_B']})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Agents\n(red=A, blue=B)', fontsize=10)

        # Bottom row: pheromone/field heatmap
        ax = axes[1, col]
        phi = snap['field']
        im = ax.imshow(phi, cmap='hot', origin='upper',
                       interpolation='nearest',
                       vmin=0, vmax=max(phi.max(), 0.1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"<φ>={phi.mean():.2f}  max={phi.max():.1f}",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel('Field φ', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle('Engineered lattice: B ⇌ A coupled to field φ\n'
                 'Below critical n: mostly B (blue), little field; '
                 'Above: more A (red), field accumulates',
                 fontsize=11)

    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")


if __name__ == '__main__':
    print("Running snapshots at 4 representative densities...")
    densities = [0.03, 0.10, 0.20, 0.40]
    snapshots = []
    for n in densities:
        print(f"  n = {n:.2f}...", flush=True)
        snap = run_and_snapshot(n, max_steps=1500)
        snapshots.append(snap)
        print(f"    f_A = {snap['f_A']:.3f}, "
              f"<φ> = {snap['field'].mean():.2f}, "
              f"max(φ) = {snap['field'].max():.1f}")

    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'ac_plus_snapshots.png')
    make_figure(snapshots, outpath)
    print(f"\nOpen: {outpath}")
