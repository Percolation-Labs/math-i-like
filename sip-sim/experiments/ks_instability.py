"""
Test the Keller-Segel instability by varying ant density ρ.

The pure KS model (no food, unconditional deposition) has a tree-level
critical density:

    n_c = D_A κ / (χ μ)

Below n_c: uniform ant distribution (no aggregation).
Above n_c: chemotactic collapse (dense cluster forms).

AC+ prediction: d_c = 2 (at the physical dimension),
mean-field exponents exact up to log corrections,
β = 1/2 for the aggregation order parameter.

Order parameter: concentration ratio max(ρ)/⟨ρ⟩ of ant positions,
or equivalently max(φ)/⟨φ⟩ of pheromone (which tracks ants).
Uniform state: ratio ≈ 1. Collapsed: ratio >> 1.

Fixed: λ=0.15 (evaporation), σ=0.005 (chem diffusion),
       δ=6 (deposit rate), n=2.5 (bias exponent).
Vary: ρ ∈ [0.005, 0.20]
"""

import numpy as np
import time
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def run_one(rho, lam=0.15, grid=80, seed=42, max_steps=2000):
    spec = {
        'name': 'x',
        'topology': {'kind': 'grid', 'width': grid, 'height': grid,
                     'wrap': True, 'neighbourhood': 'moore'},
        'agent_kinds': [{
            'name': 'ant', 'color': '#e74c3c',
            'initial_fraction': rho, 'mobile': True,
            'state_schema': {'dir_r': 'float', 'dir_c': 'float'},
            'initial_state': {'dir_r': 0, 'dir_c': 0},
        }],
        'env_fields': [{'name': 'pheromone', 'initial_value': 0.0}],
        'field_operations': [
            {'field': 'pheromone', 'kind': 'decay', 'rate': lam},
            {'field': 'pheromone', 'kind': 'diffuse', 'rate': 0.005},
        ],
        'rules': [
            {'name': 'dep', 'condition': {'self_kind': 'ant'},
             'action': {'kind': 'modify_state',
                        'state_update': {'pheromone': '+6.0'}}, 'priority': 1},
            {'name': 'mov', 'condition': {'self_kind': 'ant'},
             'action': {'kind': 'move', 'params': {
                 'bias_field': 'pheromone', 'bias_exponent': 2.5,
                 'bias_base': 0.3, 'momentum': 2.0,
                 'heading_alpha': 0.4, 'noise': 0.04}}, 'priority': 0},
        ],
        'observables': [],
        'max_steps': max_steps, 'seed': seed,
    }
    engine = build_engine(SimulationSpec(**spec))
    engine.run()

    phi = engine.state.env_grids['pheromone']
    mean_phi = phi.mean()
    max_phi = phi.max()
    conc_ratio = max_phi / mean_phi if mean_phi > 0.01 else 1.0

    # Ant density concentration
    ant_kind_idx = engine.kind_index['ant']
    ant_grid = (engine.state.kind_grid == ant_kind_idx).astype(float)

    # Smooth ant density by convolution (window size = 5)
    from itertools import product
    ws = 5
    smoothed = np.zeros_like(ant_grid)
    for dy, dx in product(range(-ws//2, ws//2+1), repeat=2):
        smoothed += np.roll(np.roll(ant_grid, dy, axis=0), dx, axis=1)
    smoothed /= ws*ws
    mean_ant = smoothed.mean()
    max_ant = smoothed.max()
    ant_conc = max_ant / mean_ant if mean_ant > 0.01 else 1.0

    # Block variance of ant density (measures aggregation)
    bs = 8
    bh, bw = grid//bs, grid//bs
    tr = ant_grid[:bh*bs, :bw*bs]
    blocks = tr.reshape(bh, bs, bw, bs).sum(axis=(1, 3))
    B_ant = float(np.var(blocks) / blocks.mean()**2) if blocks.mean() > 0.01 else 0

    return {
        'ant_density': rho,
        'seed': seed,
        'conc_ratio_phi': conc_ratio,
        'conc_ratio_ant': ant_conc,
        'block_var_ant': B_ant,
        'mean_phi': mean_phi,
        'max_phi': max_phi,
    }


if __name__ == '__main__':
    print("="*72)
    print("KS INSTABILITY: vary ant density, look for aggregation")
    print("  Pure KS model (no food), fixed λ=0.15, σ=0.005")
    print("  Tree-level prediction: aggregation above n_c")
    print("="*72, flush=True)

    # Dense density sweep
    rho_values = np.concatenate([
        np.arange(0.005, 0.03, 0.003),   # very low
        np.arange(0.03, 0.10, 0.005),    # transition region
        np.arange(0.10, 0.25, 0.01),     # above transition
    ])
    n_seeds = 3

    total = len(rho_values) * n_seeds
    print(f"ρ values: {len(rho_values)}, seeds: {n_seeds}, total: {total}",
          flush=True)

    results = []
    t0 = time.time()

    for i, rho in enumerate(rho_values):
        seed_results = []
        for s in range(n_seeds):
            result = run_one(rho, seed=42 + s * 17)
            results.append(result)
            seed_results.append(result)

        done = (i + 1) * n_seeds
        elapsed = time.time() - t0
        eta = (total - done) * elapsed / done if done > 0 else 0

        c_phi = np.mean([r['conc_ratio_phi'] for r in seed_results])
        c_ant = np.mean([r['conc_ratio_ant'] for r in seed_results])
        B = np.mean([r['block_var_ant'] for r in seed_results])
        c_phi_std = np.std([r['conc_ratio_phi'] for r in seed_results])

        print(f"[{done:>3d}/{total}] ρ={rho:.4f}  "
              f"C_φ={c_phi:.2f}±{c_phi_std:.2f}  "
              f"C_ant={c_ant:.2f}  B_ant={B:.3f}  "
              f"({eta/60:.0f}min left)", flush=True)

    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'ks_instability.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {outpath}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min", flush=True)

    # Analysis
    print(f"\n{'='*72}\nANALYSIS\n{'='*72}", flush=True)

    by_rho = {}
    for r in results:
        by_rho.setdefault(r['ant_density'], []).append(r)

    print(f"\n{'ρ':>7s} {'C_φ':>7s} {'±':>5s} {'C_ant':>6s} {'B_ant':>6s}")
    print("-"*36)
    rhos_sorted = sorted(by_rho.keys())
    Cs_phi = []
    Bs_ant = []
    for rho in rhos_sorted:
        runs = by_rho[rho]
        c_phi = np.mean([r['conc_ratio_phi'] for r in runs])
        c_phi_s = np.std([r['conc_ratio_phi'] for r in runs])
        c_ant = np.mean([r['conc_ratio_ant'] for r in runs])
        B = np.mean([r['block_var_ant'] for r in runs])
        Cs_phi.append(c_phi)
        Bs_ant.append(B)
        print(f"{rho:>7.4f} {c_phi:>7.2f} {c_phi_s:>5.2f} {c_ant:>6.2f} {B:>6.3f}")

    rhos_arr = np.array(rhos_sorted)
    Cs = np.array(Cs_phi)
    Bs = np.array(Bs_ant)

    # Look for a jump in B_ant (aggregation signature)
    if len(Bs) > 5:
        dB = np.gradient(Bs, rhos_arr)
        i_max = np.argmax(dB)
        print(f"\nSteepest rise in B_ant: ρ = {rhos_arr[i_max]:.4f}, "
              f"dB/dρ = {dB[i_max]:.2f}")

    # Fit B_ant ~ (ρ - ρ_c)^β above threshold
    print(f"\nFitting B_ant = A·(ρ-ρ_c)^β above transition:")
    best = (0, 999, 0)
    for rho_c in np.arange(0.01, 0.08, 0.005):
        above = [(r, b) for r, b in zip(rhos_arr, Bs)
                 if r > rho_c and b > 0.05]
        if len(above) < 4:
            continue
        x = np.log([r - rho_c for r, _ in above])
        y = np.log([b for _, b in above])
        slope, intercept = np.polyfit(x, y, 1)
        rmse = np.sqrt(np.mean((slope*x + intercept - y)**2))
        if rmse < best[1]:
            best = (rho_c, rmse, slope)

    rc, rmse, beta = best
    print(f"  BEST: ρ_c = {rc:.4f}, β = {beta:.3f}, RMSE = {rmse:.4f}")
    print(f"  Predictions:")
    print(f"    AC+ KS mean-field:  β = 1/2 = 0.500")
    print(f"    Error: {abs(beta - 0.5)/0.5*100:.1f}%")
