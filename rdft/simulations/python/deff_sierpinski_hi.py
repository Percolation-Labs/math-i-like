#!/usr/bin/env python3
"""Run higher-barrier parameter sets with large t_max."""
import os, sys, json, time
sys.path.insert(0, '/Users/sirsh/code/math/rdft')
from simulations.python.deff_sierpinski import (
    build_sierpinski_gasket, find_central_site, flatten_adj,
    quasi_potential_barrier, run_one_gillespie,
)
import numpy as np
from multiprocessing import Pool


def main():
    coords, adj = build_sierpinski_gasket(5)
    N = len(adj)
    center = find_central_site(coords)
    adj_flat, adj_off = flatten_adj(adj)
    print(f"Gasket L=5, N={N}, center={center}", flush=True)

    CASES = [
        (2.50, 5000.0),
        (2.75, 20000.0),
        (3.00, 80000.0),
    ]
    V_effs = [6.0, 8.0, 10.0]
    N_TRIALS = 80

    results = []
    t_start = time.time()
    for delta, t_max in CASES:
        qp = quasi_potential_barrier(0.05, 4.0, 0.7, delta, 1.0)
        print(f"\n=== delta={delta} (S1={qp['S1']:.3f} Δf={qp['Delta_f']:.3f}) ===", flush=True)
        per_V = []
        for V_eff in V_effs:
            actual_t_max = t_max * np.exp((V_eff - 6) * qp['S1'])
            actual_t_max = min(actual_t_max, 2e6)
            args_list = [
                (0.05, 4.0, 0.7, delta, 1.0,
                 V_eff, adj_flat, adj_off, N, center,
                 qp['x_low'], qp['x_high'], actual_t_max,
                 5_000_000, int(seed + V_eff * 1000 + delta * 10000))
                for seed in range(N_TRIALS)
            ]
            t0 = time.time()
            with Pool(6) as pool:
                out = pool.map(run_one_gillespie, args_list)
            dt = time.time() - t0
            taus = np.array([r['tau'] for r in out])
            flipped = np.array([r['flipped'] for r in out])
            ff = flipped.mean()
            taus_use = taus[flipped] if flipped.sum() >= 5 else taus
            print(f"    V_eff={V_eff}  t_max={actual_t_max:.0f}  flips={ff:.2f}  "
                  f"<τ>={taus_use.mean():.3e}  med τ={np.median(taus_use):.3e}  "
                  f"wall={dt:.1f}s",
                  flush=True)
            per_V.append({
                'V_eff': float(V_eff),
                'flip_frac': float(ff),
                't_max': float(actual_t_max),
                'tau_mean': float(taus_use.mean()),
                'tau_median': float(np.median(taus_use)),
                'n_flipped': int(flipped.sum()),
                'taus_all': taus.tolist(),
                'flipped_all': [bool(x) for x in flipped.tolist()],
            })
        results.append({
            'delta': delta,
            'S1': qp['S1'],
            'Delta_f': qp['Delta_f'],
            'x_low': qp['x_low'], 'x_high': qp['x_high'],
            'per_V': per_V,
        })

    print(f"\nTotal wall: {time.time()-t_start:.1f}s", flush=True)

    os.makedirs('/Users/sirsh/code/math/rdft/simulations/results', exist_ok=True)
    with open('/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski_hi.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Wrote results/deff_sierpinski_hi.json", flush=True)


if __name__ == '__main__':
    main()
