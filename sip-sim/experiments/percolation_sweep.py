"""
Percolation-focused sweep: smaller grid, MORE patches, closer spacing.

The previous sweep failed because 32 patches on 160x160 = patches
28 cells apart, too far for trails to connect.

This version: 64 patches on 100x100 = patches ~12 cells apart.
Within reach of ant persistence length.

Also measures the DISTRIBUTION of component sizes, not just the largest,
which is the proper percolation order parameter.
"""

import numpy as np
import time
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def label_components(binary_grid):
    H, W = binary_grid.shape
    labeled = np.zeros_like(binary_grid, dtype=np.int32)
    comp_id = 0
    for sy in range(H):
        for sx in range(W):
            if binary_grid[sy, sx] and labeled[sy, sx] == 0:
                comp_id += 1
                stack = [(sy, sx)]
                labeled[sy, sx] = comp_id
                while stack:
                    y, x = stack.pop()
                    # Moore 8-connectivity (matches simulation neighbourhood)
                    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),
                                   (1,-1),(1,0),(1,1)]:
                        ny, nx = (y+dy) % H, (x+dx) % W
                        if binary_grid[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = comp_id
                            stack.append((ny, nx))
    return labeled, comp_id


def run_one(lam, n_patches=64, patch_radius=2, grid=100,
            ant_density=0.06, seed=42, max_steps=2500):
    spec = {
        'name': 'x',
        'topology': {'kind': 'grid', 'width': grid, 'height': grid,
                     'wrap': True, 'neighbourhood': 'moore'},
        'agent_kinds': [
            {'name': 'unladen', 'color': '#3498db', 'initial_fraction': ant_density,
             'mobile': True, 'state_schema': {'dir_r': 'float', 'dir_c': 'float'},
             'initial_state': {'dir_r': 0, 'dir_c': 0}},
            {'name': 'laden', 'color': '#e74c3c', 'initial_fraction': 0.0,
             'mobile': True, 'state_schema': {'dir_r': 'float', 'dir_c': 'float'},
             'initial_state': {'dir_r': 0, 'dir_c': 0}},
            {'name': 'food', 'color': '#27ae60', 'initial_fraction': 0.0, 'mobile': False},
        ],
        'env_fields': [{'name': 'pheromone', 'initial_value': 0.0}],
        'field_operations': [
            {'field': 'pheromone', 'kind': 'decay', 'rate': lam},
            {'field': 'pheromone', 'kind': 'diffuse', 'rate': 0.008},
        ],
        'rules': [
            {'name': 'pick', 'condition': {'self_kind': 'unladen',
             'neighbour_count': {'kind': 'food', 'gte': 1}, 'random_threshold': 0.3},
             'action': {'kind': 'transform', 'new_kind': 'laden'}, 'priority': 4},
            {'name': 'hoard', 'condition': {'self_kind': 'laden',
             'neighbour_count': {'kind': 'food', 'gte': 1}, 'random_threshold': 0.5},
             'action': {'kind': 'transform', 'new_kind': 'unladen'}, 'priority': 3},
            {'name': 'drop', 'condition': {'self_kind': 'laden', 'random_threshold': 0.01},
             'action': {'kind': 'transform', 'new_kind': 'unladen'}, 'priority': 2},
            {'name': 'dep', 'condition': {'self_kind': 'laden'},
             'action': {'kind': 'modify_state', 'state_update': {'pheromone': '+6.0'}},
             'priority': 1},
            {'name': 'umov', 'condition': {'self_kind': 'unladen'},
             'action': {'kind': 'move', 'params': {
                 'bias_field': 'pheromone', 'bias_exponent': 2.0, 'bias_base': 0.3,
                 'momentum': 2.0, 'heading_alpha': 0.4, 'noise': 0.05}}, 'priority': 0},
            {'name': 'lmov', 'condition': {'self_kind': 'laden'},
             'action': {'kind': 'move', 'params': {
                 'bias_field': 'pheromone', 'bias_exponent': 2.0, 'bias_base': 0.3,
                 'momentum': 2.0, 'heading_alpha': 0.4, 'noise': 0.02}}, 'priority': 0},
        ],
        'observables': [
            {'name': 'L', 'kind': 'trail_linearity',
             'params': {'field': 'pheromone', 'threshold_frac': 0.3}},
        ],
        'max_steps': max_steps, 'seed': seed,
    }

    engine = build_engine(SimulationSpec(**spec))

    # Place food patches on a slightly jittered grid for more regular spacing
    rng = np.random.default_rng(seed + 1000)
    fi = engine.kind_index['food']
    patch_centers = []

    # Use stratified placement: divide grid into cells and place one patch per cell
    n_side = int(np.ceil(np.sqrt(n_patches)))
    cell_size = grid / n_side
    placed = 0
    for i in range(n_side):
        for j in range(n_side):
            if placed >= n_patches:
                break
            # Jitter within cell
            cy = int((i + 0.5 + 0.3 * (rng.random() - 0.5)) * cell_size) % grid
            cx = int((j + 0.5 + 0.3 * (rng.random() - 0.5)) * cell_size) % grid
            patch_centers.append((cy, cx))
            for dy in range(-patch_radius, patch_radius + 1):
                for dx in range(-patch_radius, patch_radius + 1):
                    y, x = (cy + dy) % grid, (cx + dx) % grid
                    if engine.state.kind_grid[y, x] == 0:
                        engine.state.kind_grid[y, x] = fi
            placed += 1

    trace = engine.run()
    phi = engine.state.env_grids['pheromone']
    mean_phi = phi.mean()
    max_phi = phi.max()

    L_vals = [f.observables.get('L', 0) for f in trace.frames[-300:]
              if 'L' in f.observables]
    linearity = np.mean(L_vals) if L_vals else 0

    # Connected components of hot cells (Moore 8-connected)
    # Use 0.5×mean threshold: captures the trail structure
    # (2×mean was too strict — fragmented everything into tiny islands)
    threshold = max(0.5 * mean_phi, 0.05)
    hot = phi > threshold
    labeled, n_components = label_components(hot)

    # Component sizes
    if n_components > 0:
        sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
        sizes_sorted = np.sort(sizes)[::-1]
        largest = sizes_sorted[0] if len(sizes_sorted) > 0 else 0
        second = sizes_sorted[1] if len(sizes_sorted) > 1 else 0
    else:
        sizes = np.array([])
        largest = 0
        second = 0

    # Patches connected by trails
    patch_to_component = {}
    for idx, (cy, cx) in enumerate(patch_centers):
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                y, x = (cy + dy) % grid, (cx + dx) % grid
                if labeled[y, x] > 0:
                    patch_to_component[idx] = labeled[y, x]
                    break
            if idx in patch_to_component:
                break

    # Count patches per component
    comp_patch_count = {}
    for patch_idx, comp_id in patch_to_component.items():
        comp_patch_count[comp_id] = comp_patch_count.get(comp_id, 0) + 1

    # Largest patch-cluster
    if comp_patch_count:
        largest_patch_cluster = max(comp_patch_count.values())
    else:
        largest_patch_cluster = 0

    f_connected = largest_patch_cluster / n_patches

    # Block variance
    bs = 10
    bh, bw = grid // bs, grid // bs
    tr = phi[:bh * bs, :bw * bs]
    blocks = tr.reshape(bh, bs, bw, bs).mean(axis=(1, 3))
    B = float(np.var(blocks) / mean_phi ** 2) if mean_phi > 0.01 else 0

    return {
        'evaporation': lam,
        'seed': seed,
        'linearity': linearity,
        'f_connected': f_connected,
        'largest_patch_cluster': largest_patch_cluster,
        'n_components': n_components,
        'largest_component_size': largest,
        'second_component_size': second,
        'block_variance': B,
        'mean_phi': mean_phi,
        'max_phi': max_phi,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("PERCOLATION SWEEP: 100×100 grid, 64 food patches on 8×8 lattice")
    print("  Mean patch spacing: 12.5 cells")
    print("=" * 72, flush=True)

    # Dense sweep
    lam_values = np.concatenate([
        np.arange(0.02, 0.08, 0.02),    # 3 points, extra low
        np.arange(0.08, 0.40, 0.01),    # 32 points, dense
        np.arange(0.40, 0.70, 0.03),    # 10 points, sparse
    ])
    n_seeds = 4

    total = len(lam_values) * n_seeds
    print(f"λ values: {len(lam_values)}, seeds: {n_seeds}, total: {total}", flush=True)

    results = []
    t0 = time.time()

    for i, lam in enumerate(lam_values):
        seed_results = []
        for s in range(n_seeds):
            result = run_one(lam, seed=42 + s * 17)
            results.append(result)
            seed_results.append(result)

        done = (i + 1) * n_seeds
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate

        fc_mean = np.mean([r['f_connected'] for r in seed_results])
        fc_std = np.std([r['f_connected'] for r in seed_results])
        L_mean = np.mean([r['linearity'] for r in seed_results])
        n_comp = np.mean([r['n_components'] for r in seed_results])

        print(f"[{done:>4d}/{total}] λ={lam:.3f}  L={L_mean:.3f}  "
              f"f_conn={fc_mean:.3f}±{fc_std:.3f}  n_comp={n_comp:.0f}  "
              f"({eta/60:.0f}min left)", flush=True)

    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'percolation_sweep.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    elapsed_total = time.time() - t0
    print(f"\nSaved {len(results)} results to {outpath}")
    print(f"Total time: {elapsed_total/3600:.2f} hours", flush=True)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'=' * 72}")
    print("ANALYSIS")
    print(f"{'=' * 72}", flush=True)

    by_lam = {}
    for r in results:
        by_lam.setdefault(r['evaporation'], []).append(r)

    print(f"\n{'λ':>6s} {'L':>6s} {'f_conn':>7s} {'±':>5s} {'n_c':>5s} "
          f"{'largest':>8s} {'B':>6s}")
    print("-" * 48)

    lams_sorted = sorted(by_lam.keys())
    Ls, fcs, fc_stds = [], [], []

    for lam in lams_sorted:
        runs = by_lam[lam]
        L = np.mean([r['linearity'] for r in runs])
        fc = np.mean([r['f_connected'] for r in runs])
        fc_s = np.std([r['f_connected'] for r in runs])
        nc = np.mean([r['n_components'] for r in runs])
        lg = np.mean([r['largest_component_size'] for r in runs])
        B = np.mean([r['block_variance'] for r in runs])
        Ls.append(L)
        fcs.append(fc)
        fc_stds.append(fc_s)
        print(f"{lam:>6.3f} {L:>6.3f} {fc:>7.3f} {fc_s:>5.3f} {nc:>5.0f} "
              f"{lg:>8.1f} {B:>6.3f}")

    lams_arr = np.array(lams_sorted)
    fcs_arr = np.array(fcs)
    Ls_arr = np.array(Ls)

    # Find critical λ from f_connected
    if max(fcs_arr) > 0.3 and min(fcs_arr) < 0.1:
        for i in range(len(fcs_arr) - 1):
            if fcs_arr[i] > 0.5 * max(fcs_arr) and fcs_arr[i+1] <= 0.5 * max(fcs_arr):
                lam_c = lams_arr[i] + (0.5 * max(fcs_arr) - fcs_arr[i]) * \
                    (lams_arr[i+1] - lams_arr[i]) / (fcs_arr[i+1] - fcs_arr[i])
                print(f"\n*** f_connected crosses half-max at λ_c ≈ {lam_c:.4f} ***")

                # Transition width
                f_90 = 0.9 * max(fcs_arr)
                f_10 = 0.1 * max(fcs_arr)
                lam_90 = lam_10 = None
                for j in range(len(fcs_arr) - 1):
                    if fcs_arr[j] > f_90 and fcs_arr[j+1] <= f_90 and lam_90 is None:
                        lam_90 = lams_arr[j]
                    if fcs_arr[j] > f_10 and fcs_arr[j+1] <= f_10 and lam_10 is None:
                        lam_10 = lams_arr[j]
                if lam_90 and lam_10:
                    width = lam_10 - lam_90
                    print(f"Transition width (10%-90%): Δλ = {width:.4f}")
                    print(f"Normalised: Δλ/λ_c = {width/lam_c:.3f}")
                    if width/lam_c < 0.3:
                        print("SHARP transition — extract critical exponent")
                    else:
                        print("Broad crossover")
                break
        else:
            print("\nNo clear f_connected transition in this λ range")
    else:
        print(f"\nf_connected range: [{min(fcs_arr):.3f}, {max(fcs_arr):.3f}]")
        print("Not a clean transition — try different parameters")

    # Inflection point of f_connected
    if len(fcs_arr) > 5:
        dfc = np.gradient(fcs_arr, lams_arr)
        i_inf = np.argmin(dfc)
        print(f"Inflection of f_connected: λ = {lams_arr[i_inf]:.3f}, "
              f"steepest slope = {dfc[i_inf]:.3f}")

    print(f"\nDone.")
