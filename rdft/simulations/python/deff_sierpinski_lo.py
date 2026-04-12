#!/usr/bin/env python3
"""Low-barrier parameter sweep: delta in {1.80, 1.95, 2.05, 2.15, 2.25}.
With these choices, all trials flip at V_eff in {6,8,10} with modest t_max.
"""
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

    DELTAS = [1.80, 1.95, 2.05, 2.15, 2.25]
    V_EFFS = [6.0, 8.0, 10.0]
    N_TRIALS = 120
    T_MAX_BASE = 300.0

    results = []
    t_start = time.time()
    for delta in DELTAS:
        qp = quasi_potential_barrier(0.05, 4.0, 0.7, delta, 1.0)
        print(f"\n=== delta={delta}  S1={qp['S1']:.4f}  Δf={qp['Delta_f']:.4f} ===", flush=True)
        per_V = []
        for V_eff in V_EFFS:
            t_max = T_MAX_BASE * np.exp(max(0.0, (V_eff - 6) * qp['S1']))
            t_max = min(t_max, 1e5)
            args_list = [
                (0.05, 4.0, 0.7, delta, 1.0,
                 V_eff, adj_flat, adj_off, N, center,
                 qp['x_low'], qp['x_high'], t_max,
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
            print(f"    V_eff={V_eff}  t_max={t_max:.1f}  flips={ff:.2f}  "
                  f"<τ>={taus_use.mean():.3e}  med τ={np.median(taus_use):.3e}  "
                  f"wall={dt:.1f}s",
                  flush=True)
            per_V.append({
                'V_eff': float(V_eff),
                'flip_frac': float(ff),
                't_max': float(t_max),
                'tau_mean': float(taus_use.mean()),
                'tau_median': float(np.median(taus_use)),
                'n_flipped': int(flipped.sum()),
                'taus_all': taus.tolist(),
                'flipped_all': [bool(x) for x in flipped.tolist()],
            })
        V_arr = np.array([r['V_eff'] for r in per_V])
        mask = np.array([r['flip_frac'] > 0.3 and r['n_flipped'] >= 5 for r in per_V])
        if mask.sum() >= 2:
            lt_med = np.log(np.array([r['tau_median'] for r in per_V]))
            lt_mean = np.log(np.array([r['tau_mean'] for r in per_V]))
            slope_med = np.polyfit(V_arr[mask], lt_med[mask], 1)[0]
            slope_mean = np.polyfit(V_arr[mask], lt_mean[mask], 1)[0]
        else:
            slope_med = slope_mean = float('nan')
        print(f"  B(median)={slope_med:.3f}   B(mean)={slope_mean:.3f}", flush=True)
        results.append({
            'delta': delta,
            'S1': qp['S1'],
            'Delta_f': qp['Delta_f'],
            'x_low': qp['x_low'], 'x_high': qp['x_high'], 'x_sad': qp['x_sad'],
            'per_V': per_V,
            'B_median': slope_med,
            'B_mean': slope_mean,
        })

    print(f"\nTotal wall: {time.time()-t_start:.1f}s", flush=True)

    Df = np.array([r['Delta_f'] for r in results])
    B = np.array([r['B_median'] for r in results])
    Bm = np.array([r['B_mean'] for r in results])
    valid = (B > 0) & np.isfinite(B)
    if valid.sum() >= 3:
        x = np.log(Df[valid])
        y = np.log(B[valid])
        ym = np.log(Bm[valid])
        slope, interc = np.polyfit(x, y, 1)
        slope_m, _ = np.polyfit(x, ym, 1)
        yhat = slope * x + interc
        resid = y - yhat
        n = len(x)
        s2 = (resid**2).sum() / max(n - 2, 1)
        sx2 = ((x - x.mean())**2).sum()
        se = float(np.sqrt(s2 / sx2)) if sx2 > 0 else float('nan')
        print(f"\n=== FIT ===")
        print(f"log(B_median) vs log(Δf): slope = {slope:.3f} ± {se:.3f}")
        print(f"  -> d_eff (median) = {1 - slope:.3f}")
        print(f"log(B_mean)   vs log(Δf): slope = {slope_m:.3f}")
        print(f"  -> d_eff (mean)   = {1 - slope_m:.3f}")
        print(f"Reference: d=1 → slope 0 ; d_s=1.365 → slope -0.365 ; d=2 → slope -1")

    os.makedirs('/Users/sirsh/code/math/rdft/simulations/results', exist_ok=True)
    with open('/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski_lo.json', 'w') as f:
        json.dump({
            'geometry': {'kind': 'Sierpinski gasket', 'level': 5, 'N': N,
                         'd_s_analytic': float(2*np.log(3)/np.log(5))},
            'params': {'a': 0.05, 'lam': 4.0, 'alpha': 0.7, 'd3': 1.0,
                       'knob': 'delta', 'deltas': DELTAS,
                       'V_effs': V_EFFS, 'trials': N_TRIALS},
            'results': results,
        }, f, indent=2)
    print("Wrote results/deff_sierpinski_lo.json", flush=True)


if __name__ == '__main__':
    main()
