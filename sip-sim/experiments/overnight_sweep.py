"""
Overnight sweep: dense λ scan on 160×160 grid with 32 food patches.

Measures:
  - Trail linearity (are trails linear?)
  - Largest connected component of hot cells (do trails connect patches?)
  - Laden fraction (foraging activity)
  - Block variance (spatial heterogeneity)

The order parameter: fraction of food patches connected by trails.
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
    """Simple connected component labeling (BFS, 4-connected).
    Returns (labeled_grid, n_components)."""
    H, W = binary_grid.shape
    labeled = np.zeros_like(binary_grid, dtype=int)
    comp_id = 0
    for sy in range(H):
        for sx in range(W):
            if binary_grid[sy, sx] and labeled[sy, sx] == 0:
                comp_id += 1
                queue = [(sy, sx)]
                labeled[sy, sx] = comp_id
                while queue:
                    y, x = queue.pop()
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = (y+dy) % H, (x+dx) % W
                        if binary_grid[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = comp_id
                            queue.append((ny, nx))
    return labeled, comp_id


def run_one(lam, n_patches=32, patch_radius=2, grid=160, ant_density=0.04,
            seed=42, max_steps=3000):
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
            {'field': 'pheromone', 'kind': 'diffuse', 'rate': 0.005},
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
             'action': {'kind': 'modify_state', 'state_update': {'pheromone': '+5.0'}},
             'priority': 1},
            {'name': 'umov', 'condition': {'self_kind': 'unladen'},
             'action': {'kind': 'move', 'params': {
                 'bias_field': 'pheromone', 'bias_exponent': 1.5, 'bias_base': 0.5,
                 'momentum': 1.0, 'heading_alpha': 0.4, 'noise': 0.10}}, 'priority': 0},
            {'name': 'lmov', 'condition': {'self_kind': 'laden'},
             'action': {'kind': 'move', 'params': {
                 'bias_field': 'pheromone', 'bias_exponent': 1.5, 'bias_base': 0.5,
                 'momentum': 1.0, 'heading_alpha': 0.4, 'noise': 0.03}}, 'priority': 0},
        ],
        'observables': [
            {'name': 'unladen', 'kind': 'count', 'params': {'kind': 'unladen'}},
            {'name': 'laden', 'kind': 'count', 'params': {'kind': 'laden'}},
            {'name': 'L', 'kind': 'trail_linearity',
             'params': {'field': 'pheromone', 'threshold_frac': 0.3}},
        ],
        'max_steps': max_steps, 'seed': seed,
    }

    engine = build_engine(SimulationSpec(**spec))

    # Place food patches
    rng = np.random.default_rng(seed + 1000)
    fi = engine.kind_index['food']
    patch_centers = []
    for _ in range(n_patches):
        cy, cx = rng.integers(0, grid, size=2)
        patch_centers.append((cy, cx))
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                y, x = (cy + dy) % grid, (cx + dx) % grid
                if engine.state.kind_grid[y, x] == 0:
                    engine.state.kind_grid[y, x] = fi

    trace = engine.run()
    phi = engine.state.env_grids['pheromone']
    mean_phi = phi.mean()
    max_phi = phi.max()

    # Linearity from trace
    L_vals = [f.observables.get('L', 0) for f in trace.frames[-500:]
              if 'L' in f.observables]
    linearity = np.mean(L_vals) if L_vals else 0

    # Laden fraction from trace (last 500 steps)
    f_laden_vals = []
    for f in trace.frames[-500:]:
        nU = f.observables.get('unladen', 0)
        nL = f.observables.get('laden', 0)
        if nU + nL > 0:
            f_laden_vals.append(nL / (nU + nL))
    f_laden = np.mean(f_laden_vals) if f_laden_vals else 0

    # Connected component analysis of hot cells
    # "Hot" = pheromone > 2× mean (trail cells)
    threshold = max(2.0 * mean_phi, 0.1)
    hot = phi > threshold
    labeled, n_components = label_components(hot)

    # How many food patches are connected by trails?
    # A food patch is "connected" if its cells are in the same component
    # as at least one other food patch
    patch_components = set()
    for cy, cx in patch_centers:
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                y, x = (cy + dy) % grid, (cx + dx) % grid
                if labeled[y, x] > 0:
                    patch_components.add(labeled[y, x])

    # Largest component that contains food patches
    if patch_components:
        component_sizes = {}
        for comp_id in patch_components:
            component_sizes[comp_id] = (labeled == comp_id).sum()
        largest_component = max(component_sizes.values())
        # How many patches are in the largest component?
        largest_comp_id = max(component_sizes, key=component_sizes.get)
        patches_in_largest = 0
        for cy, cx in patch_centers:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    y, x = (cy + dy) % grid, (cx + dx) % grid
                    if labeled[y, x] == largest_comp_id:
                        patches_in_largest += 1
                        break
                else:
                    continue
                break
        f_connected = patches_in_largest / n_patches
    else:
        largest_component = 0
        f_connected = 0

    # Block variance
    bs = 16
    bh, bw = grid // bs, grid // bs
    tr = phi[:bh * bs, :bw * bs]
    blocks = tr.reshape(bh, bs, bw, bs).mean(axis=(1, 3))
    B = float(np.var(blocks) / mean_phi ** 2) if mean_phi > 0.01 else 0

    # Concentration ratio
    conc = max_phi / mean_phi if mean_phi > 0.01 else 1

    return {
        'evaporation': lam,
        'seed': seed,
        'linearity': linearity,
        'f_laden': f_laden,
        'f_connected': f_connected,
        'n_components': n_components,
        'largest_component': largest_component,
        'block_variance': B,
        'conc_ratio': conc,
        'mean_phi': mean_phi,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("OVERNIGHT SWEEP: 160×160 grid, 32 food patches")
    print("  Dense λ scan: 60 values × 5 seeds = 300 simulations")
    print("  Estimated time: ~3 hours")
    print("=" * 72, flush=True)

    # Dense λ grid: fine near expected transition, coarser at extremes
    lam_values = np.concatenate([
        np.arange(0.02, 0.10, 0.02),   # 4 points, low λ (frozen)
        np.arange(0.10, 0.50, 0.01),   # 40 points, transition region
        np.arange(0.50, 0.80, 0.02),   # 15 points, high λ (dissolved)
    ])
    n_seeds = 5

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
        rate = done / elapsed if elapsed > 0 else 1
        eta = (total - done) / rate

        # Average over seeds
        avg_L = np.mean([r['linearity'] for r in seed_results])
        avg_fc = np.mean([r['f_connected'] for r in seed_results])
        avg_fl = np.mean([r['f_laden'] for r in seed_results])
        std_fc = np.std([r['f_connected'] for r in seed_results])

        print(f"[{done:>4d}/{total}] λ={lam:.3f}  L={avg_L:.3f}  "
              f"f_conn={avg_fc:.3f}±{std_fc:.3f}  f_laden={avg_fl:.3f}  "
              f"({eta/60:.0f}min left)", flush=True)

    # Save raw results
    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'overnight_sweep.csv')
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

    # Average over seeds
    by_lam = {}
    for r in results:
        lam = r['evaporation']
        if lam not in by_lam:
            by_lam[lam] = []
        by_lam[lam].append(r)

    print(f"\n{'λ':>6s} {'L':>6s} {'f_conn':>7s} {'±':>5s} {'f_laden':>7s} "
          f"{'B':>6s} {'n_comp':>7s}")
    print("-" * 50)

    lams_sorted = sorted(by_lam.keys())
    Ls_avg = []
    fc_avg = []
    fc_std = []

    for lam in lams_sorted:
        runs = by_lam[lam]
        L = np.mean([r['linearity'] for r in runs])
        fc = np.mean([r['f_connected'] for r in runs])
        fc_s = np.std([r['f_connected'] for r in runs])
        fl = np.mean([r['f_laden'] for r in runs])
        B = np.mean([r['block_variance'] for r in runs])
        nc = np.mean([r['n_components'] for r in runs])
        Ls_avg.append(L)
        fc_avg.append(fc)
        fc_std.append(fc_s)
        print(f"{lam:>6.3f} {L:>6.3f} {fc:>7.3f} {fc_s:>5.3f} {fl:>7.3f} "
              f"{B:>6.3f} {nc:>7.0f}")

    lams_arr = np.array(lams_sorted)
    fc_arr = np.array(fc_avg)
    Ls_arr = np.array(Ls_avg)

    # Find critical λ from f_connected
    # Midpoint: where f_connected crosses 0.5
    for i in range(len(fc_arr) - 1):
        if fc_arr[i] > 0.5 and fc_arr[i+1] <= 0.5:
            lam_c = lams_arr[i] + (0.5 - fc_arr[i]) * (lams_arr[i+1] - lams_arr[i]) / (fc_arr[i+1] - fc_arr[i])
            print(f"\nf_connected crosses 0.5 at λ_c ≈ {lam_c:.4f}")
            break
    else:
        # Try from linearity
        L_mid = (max(Ls_arr) + min(Ls_arr)) / 2
        for i in range(len(Ls_arr) - 1):
            if Ls_arr[i] > L_mid and Ls_arr[i+1] <= L_mid:
                lam_c = lams_arr[i] + (L_mid - Ls_arr[i]) * (lams_arr[i+1] - lams_arr[i]) / (Ls_arr[i+1] - Ls_arr[i])
                print(f"\nLinearity crosses midpoint at λ_c ≈ {lam_c:.4f}")
                break

    # Inflection point of f_connected
    dfc = np.gradient(fc_arr, lams_arr)
    i_inf = np.argmin(dfc)
    print(f"Inflection of f_connected: λ = {lams_arr[i_inf]:.3f}, "
          f"df/dλ = {dfc[i_inf]:.3f}")

    # Transition width
    if max(fc_arr) > 0.9 and min(fc_arr) < 0.1:
        fc_90 = fc_10 = None
        for i in range(len(fc_arr) - 1):
            if fc_arr[i] > 0.9 and fc_arr[i+1] <= 0.9 and fc_90 is None:
                fc_90 = lams_arr[i]
            if fc_arr[i] > 0.1 and fc_arr[i+1] <= 0.1 and fc_10 is None:
                fc_10 = lams_arr[i]
        if fc_90 and fc_10:
            print(f"Transition width (10%-90%): Δλ = {fc_10 - fc_90:.3f}")
            print(f"  Normalized: Δλ/λ_c = {(fc_10 - fc_90)/lam_c:.3f}")

    print(f"\nDone. Results in {outpath}")
