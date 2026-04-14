"""
Thread 1: Linear-response ant model — weak coupling regime.

Replace the nonlinear bias (β + φ)^n with the linear form β + φ
(i.e., n = 1) and take β large. This puts the system in weak-
coupling:

    w_j = β + φ_j,   p_j ∝ (1 + δφ_j/β) / N + O(δφ²)

The effective chemotactic coupling is χ_eff = 1/β. Large β →
small χ_eff → perturbative regime where AC+ should apply.

Sweep: ant density ρ at fixed parameters, look for a genuine
KS-like density transition (aggregation above n_c).
"""

import numpy as np
import time
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def run_one(rho, bias_base=5.0, lam=0.15, grid=80, seed=42, max_steps=1500):
    """Linear response: n=1, large β to stay in weak coupling."""
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
                        'state_update': {'pheromone': '+2.0'}}, 'priority': 1},
            {'name': 'mov', 'condition': {'self_kind': 'ant'},
             'action': {'kind': 'move', 'params': {
                 'bias_field': 'pheromone',
                 'bias_exponent': 1.0,       # LINEAR response
                 'bias_base': bias_base,     # LARGE β for weak coupling
                 'momentum': 0.5,            # low momentum
                 'heading_alpha': 0.4,
                 'noise': 0.05}}, 'priority': 0},
        ],
        'observables': [],
        'max_steps': max_steps, 'seed': seed,
    }
    engine = build_engine(SimulationSpec(**spec))
    engine.run()

    phi = engine.state.env_grids['pheromone']
    mean_phi = phi.mean()
    max_phi = phi.max()
    conc_phi = max_phi / mean_phi if mean_phi > 0.01 else 1.0

    ant_kind = engine.kind_index['ant']
    ant_grid = (engine.state.kind_grid == ant_kind).astype(float)
    bs = 8
    bh, bw = grid // bs, grid // bs
    blocks = ant_grid[:bh*bs, :bw*bs].reshape(bh, bs, bw, bs).sum(axis=(1, 3))
    mean_n = blocks.mean()
    B_ant = float(np.var(blocks) / mean_n**2) if mean_n > 0.01 else 0
    # Poisson baseline: Var/mean = 1, so B_ant ~ 1/mean_n for Poisson
    # Aggregation: B_ant >> 1/mean_n
    poisson_B = 1.0 / mean_n if mean_n > 0 else 0
    B_excess = B_ant - poisson_B  # signal of aggregation above Poisson

    return {
        'ant_density': rho,
        'bias_base': bias_base,
        'seed': seed,
        'conc_ratio_phi': conc_phi,
        'block_var_ant': B_ant,
        'poisson_baseline': poisson_B,
        'B_excess': B_excess,
        'mean_phi': mean_phi,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("LINEAR RESPONSE SWEEP (Thread 1)")
    print("  n=1 (linear bias), β=5.0 (weak coupling)")
    print("  Sweep ρ; look for KS-like aggregation above n_c")
    print("=" * 72, flush=True)

    # Test a few β values first to find weak-coupling regime
    print("\n--- Diagnostic: B_excess vs β at ρ=0.04 ---")
    print(f"{'β':>5s} {'C_φ':>6s} {'B_ant':>7s} {'B_poisson':>10s} {'B_excess':>9s}")
    for beta in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        r = run_one(0.04, bias_base=beta, max_steps=1200)
        print(f"{beta:>5.1f} {r['conc_ratio_phi']:>6.2f} {r['block_var_ant']:>7.3f} "
              f"{r['poisson_baseline']:>10.3f} {r['B_excess']:>9.3f}", flush=True)

    # Main sweep with β = 5.0
    print("\n--- Main sweep: ρ at β=5 ---")
    rho_values = np.concatenate([
        np.arange(0.005, 0.03, 0.005),
        np.arange(0.03, 0.12, 0.01),
        np.arange(0.12, 0.22, 0.02),
    ])
    n_seeds = 3
    total = len(rho_values) * n_seeds
    print(f"ρ values: {len(rho_values)}, seeds: {n_seeds}, total: {total}",
          flush=True)

    results = []
    t0 = time.time()
    print(f"\n{'ρ':>7s} {'C_φ':>6s} {'±':>5s} {'B_ant':>7s} {'B_excess':>9s} {'ETA':>6s}")
    print("-" * 50)

    for i, rho in enumerate(rho_values):
        seed_results = []
        for s in range(n_seeds):
            result = run_one(rho, bias_base=5.0, seed=42 + s * 17)
            results.append(result)
            seed_results.append(result)
        done = (i + 1) * n_seeds
        elapsed = time.time() - t0
        eta = (total - done) * elapsed / done if done > 0 else 0

        c_m = np.mean([r['conc_ratio_phi'] for r in seed_results])
        c_s = np.std([r['conc_ratio_phi'] for r in seed_results])
        B_m = np.mean([r['block_var_ant'] for r in seed_results])
        Be_m = np.mean([r['B_excess'] for r in seed_results])

        print(f"{rho:>7.4f} {c_m:>6.2f} {c_s:>5.2f} {B_m:>7.3f} {Be_m:>9.4f} "
              f"{eta/60:>5.0f}m", flush=True)

    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'linear_response.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {outpath}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

    # Analysis
    by_rho = {}
    for r in results:
        by_rho.setdefault(r['ant_density'], []).append(r)

    rhos = np.array(sorted(by_rho.keys()))
    Be = np.array([np.mean([r['B_excess'] for r in by_rho[rho]]) for rho in rhos])

    # Fit B_excess ~ (ρ - ρ_c)^β for ρ > ρ_c
    best = (0, 999, 0)
    for rho_c in np.arange(0.005, 0.10, 0.005):
        above = [(r, b) for r, b in zip(rhos, Be) if r > rho_c and b > 0.01]
        if len(above) < 4: continue
        x = np.log([r - rho_c for r, _ in above])
        y = np.log([b for _, b in above])
        slope, intercept = np.polyfit(x, y, 1)
        rmse = np.sqrt(np.mean((slope*x + intercept - y)**2))
        if rmse < best[1]:
            best = (rho_c, rmse, slope)
    rc, rmse, beta = best

    print(f"\n{'='*72}\nANALYSIS\n{'='*72}")
    print(f"Best fit: ρ_c = {rc:.4f}, β = {beta:.3f}, RMSE = {rmse:.4f}")
    print(f"\nCompare to:")
    print(f"  Mean-field (AC+): β = 1/2 = 0.500")
    print(f"  Error: {abs(beta - 0.5)/0.5*100:.0f}%")
    if abs(beta - 0.5) < 0.1:
        print("  *** CONSISTENT WITH AC+ MEAN-FIELD PREDICTION ***")
