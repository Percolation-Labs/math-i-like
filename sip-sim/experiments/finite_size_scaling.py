"""
Finite-size scaling test for the erosion-driven transition.

AC+ prediction: the ant-pheromone system is Keller-Segel class
with d_c = 2. In d = 2, mean-field is exact, so:
  - order parameter exponent β = 1/2
  - correlation length exponent ν = 1/2
  - transition width Δλ_c ~ L^{-1/ν} = L^{-2}

Test: run the EROSION transition (all ants follow, vary evaporation λ)
at grid sizes L = 40, 60, 80, 100. Measure block variance B vs λ.
The transition width should scale as L^{-2} (AC+ / mean-field)
vs L^{-1.36} (DP class) vs L^{-1} (Ising class).

This is the erosion transition, NOT the recruitment transition.
All ants follow pheromone (p_follow = 1.0 effectively via low
susceptibility threshold s = 0.1). Evaporation λ controls whether
trails are frozen (low λ) or dissolved (high λ).
"""

import numpy as np
import json
import time
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def make_spec(grid_size: int, evaporation: float, seed: int = 42) -> dict:
    """Create an ant pheromone spec for the erosion regime.

    All ants follow pheromone (susceptibility threshold = 0.1, very low).
    Evaporation λ is the control parameter.
    """
    return {
        "name": f"FSS L={grid_size} λ={evaporation:.3f}",
        "topology": {
            "kind": "grid", "width": grid_size, "height": grid_size,
            "wrap": True, "neighbourhood": "moore"
        },
        "agent_kinds": [{
            "name": "ant", "color": "#e74c3c",
            "initial_fraction": 0.04, "mobile": True,
            "state_schema": {"dir_r": "float", "dir_c": "float"},
            "initial_state": {"dir_r": 0, "dir_c": 0},
        }],
        "env_fields": [
            {"name": "pheromone", "initial_value": 0.0},
        ],
        "field_operations": [
            {"field": "pheromone", "kind": "decay", "rate": evaporation},
            {"field": "pheromone", "kind": "diffuse", "rate": 0.002},
        ],
        "rules": [
            {
                "name": "deposit",
                "condition": {"self_kind": "ant"},
                "action": {"kind": "modify_state",
                          "state_update": {"pheromone": "+5.0"}},
                "priority": 1,
            },
            {
                "name": "move",
                "condition": {"self_kind": "ant"},
                "action": {
                    "kind": "move",
                    "params": {
                        "bias_field": "pheromone",
                        "bias_exponent": 2.5,
                        "bias_base": 0.1,
                        "momentum": 2.0,
                        "heading_alpha": 0.4,
                        # Low noise = all ants follow (erosion regime)
                        "noise": 0.02,
                    },
                },
                "priority": 0,
            },
        ],
        "observables": [
            {"name": "ants", "kind": "count", "params": {"kind": "ant"}},
            {"name": "pheromone_mean", "kind": "env_mean",
             "params": {"field": "pheromone"}},
            {"name": "trail_linearity", "kind": "trail_linearity",
             "params": {"field": "pheromone", "threshold_frac": 0.3}},
            {"name": "trail_turnover", "kind": "field_autocorrelation",
             "params": {"field": "pheromone", "lag": 100}},
        ],
        "max_steps": 800,
        "snapshot_interval": 200,
        "seed": seed,
    }


def measure_block_variance(engine, block_size=10):
    """Compute block variance of the pheromone field."""
    phi = engine.state.env_grids.get('pheromone')
    if phi is None:
        return 0.0
    H, W = phi.shape
    bh = H // block_size
    bw = W // block_size
    if bh == 0 or bw == 0:
        return 0.0

    # Reshape into blocks and compute block means
    trimmed = phi[:bh*block_size, :bw*block_size]
    blocks = trimmed.reshape(bh, block_size, bw, block_size).mean(axis=(1, 3))

    global_mean = phi.mean()
    if global_mean < 1e-10:
        return 0.0

    return float(np.var(blocks) / global_mean**2)


def run_single(grid_size, evaporation, seed=42):
    """Run one simulation and return observables."""
    spec_dict = make_spec(grid_size, evaporation, seed)
    spec = SimulationSpec(**spec_dict)
    engine = build_engine(spec)

    # Run and collect block variance in the last 200 steps
    B_values = []
    A_values = []

    trace = engine.run()

    # Compute block variance from final state
    block_size = max(5, grid_size // 10)
    B_final = measure_block_variance(engine, block_size)

    # Get autocorrelation from trace
    A_final = 0.0
    L_final = 0.0
    for frame in trace.frames[-50:]:
        if 'trail_turnover' in frame.observables:
            A_final = frame.observables['trail_turnover']
        if 'trail_linearity' in frame.observables:
            L_final = frame.observables['trail_linearity']

    return {
        'grid_size': grid_size,
        'evaporation': evaporation,
        'block_variance': B_final,
        'autocorrelation': A_final,
        'linearity': L_final,
        'seed': seed,
    }


if __name__ == '__main__':
    grid_sizes = [40, 60, 80, 100]
    # Fine grid of evaporation rates around the expected transition
    evap_rates = np.arange(0.05, 0.50, 0.025)
    n_seeds = 3  # repeat each for statistics

    total = len(grid_sizes) * len(evap_rates) * n_seeds
    print(f"Finite-size scaling sweep: {len(grid_sizes)} sizes × "
          f"{len(evap_rates)} evap rates × {n_seeds} seeds = {total} runs")

    results = []
    done = 0
    t0 = time.time()

    for L in grid_sizes:
        for evap in evap_rates:
            for seed in range(n_seeds):
                result = run_single(L, evap, seed=42 + seed)
                results.append(result)
                done += 1

                if done % 10 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate
                    print(f"  [{done}/{total}] L={L} λ={evap:.3f} "
                          f"B={result['block_variance']:.4f} "
                          f"A={result['autocorrelation']:.3f} "
                          f"({eta:.0f}s remaining)")

    # Save results
    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'finite_size_scaling.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {outpath}")

    # Quick analysis
    print(f"\n{'='*60}")
    print(f"QUICK ANALYSIS")
    print(f"{'='*60}")

    for L in grid_sizes:
        L_data = [r for r in results if r['grid_size'] == L]

        # Composite C = B × (1-A) — same as paper
        for r in L_data:
            r['composite'] = r['block_variance'] * max(0, 1 - r['autocorrelation'])

        # Find peak composite
        by_evap = {}
        for r in L_data:
            e = r['evaporation']
            if e not in by_evap:
                by_evap[e] = []
            by_evap[e].append(r['composite'])

        mean_C = {e: np.mean(vals) for e, vals in by_evap.items()}
        peak_evap = max(mean_C, key=mean_C.get)
        peak_C = mean_C[peak_evap]

        # Transition width: range of λ where C > peak_C / 2
        above_half = [e for e, c in mean_C.items() if c > peak_C / 2]
        if above_half:
            width = max(above_half) - min(above_half)
        else:
            width = 0

        print(f"L={L:>4d}: peak at λ={peak_evap:.3f}, C={peak_C:.4f}, width={width:.3f}")

    # Finite-size scaling of the width
    print(f"\nFinite-size scaling of transition width Δλ:")
    widths = []
    for L in grid_sizes:
        L_data = [r for r in results if r['grid_size'] == L]
        by_evap = {}
        for r in L_data:
            e = r['evaporation']
            c = r['block_variance'] * max(0, 1 - r['autocorrelation'])
            if e not in by_evap:
                by_evap[e] = []
            by_evap[e].append(c)

        mean_C = {e: np.mean(vals) for e, vals in by_evap.items()}
        peak_C = max(mean_C.values())
        above_half = [e for e, c in mean_C.items() if c > peak_C / 2]
        width = max(above_half) - min(above_half) if above_half else 0.001
        widths.append((L, width))
        print(f"  L={L}: Δλ = {width:.4f}")

    if len(widths) >= 3:
        Ls = np.array([w[0] for w in widths])
        Ws = np.array([w[1] for w in widths])

        if all(w > 0 for w in Ws):
            slope, intercept = np.polyfit(np.log(Ls), np.log(Ws), 1)
            nu_measured = -1.0 / slope

            print(f"\n  Fit: Δλ ~ L^{slope:.3f}")
            print(f"  => 1/ν = {-slope:.3f}")
            print(f"  => ν = {nu_measured:.3f}")
            print(f"\n  Predictions:")
            print(f"    AC+ (mean-field, d=d_c): ν = 0.500, Δλ ~ L^{-2.0}")
            print(f"    DP class:                ν = 0.734, Δλ ~ L^{-1.36}")
            print(f"    Ising class:             ν = 1.000, Δλ ~ L^{-1.0}")
            print(f"    Measured:                ν = {nu_measured:.3f}, Δλ ~ L^{slope:.2f}")
