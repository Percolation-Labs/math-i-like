#!/usr/bin/env python3
"""Analyze deff_sierpinski results: load, filter, fit, report."""
import json, sys, os
import numpy as np

def load(path):
    with open(path) as f:
        return json.load(f)


def fit_scaling(df_arr, B_arr, label=""):
    mask = (B_arr > 0) & np.isfinite(B_arr) & np.isfinite(df_arr)
    if mask.sum() < 3:
        print(f"  [{label}] not enough valid points ({mask.sum()})")
        return
    x = np.log(df_arr[mask])
    y = np.log(B_arr[mask])
    slope, interc = np.polyfit(x, y, 1)
    yhat = slope * x + interc
    resid = y - yhat
    n = len(x)
    s2 = (resid**2).sum() / max(n - 2, 1)
    sx2 = ((x - x.mean())**2).sum()
    se = float(np.sqrt(s2 / sx2)) if sx2 > 0 else float('nan')
    d_eff = 1.0 - slope
    d_eff_se = se
    print(f"  [{label}]  slope = {slope:.3f} ± {se:.3f}   "
          f"d_eff = {d_eff:.3f} ± {d_eff_se:.3f}   "
          f"N_points = {mask.sum()}")
    return slope, se, d_eff


def analyze_lo():
    path = '/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski_lo.json'
    if not os.path.exists(path):
        print(f"LO file not found: {path}")
        return None
    data = load(path)
    res = data['results']
    print("\n== LO (low-barrier sweep) ==")
    print(f"  Geometry: {data['geometry']}")
    print(f"  {'delta':>6s}  {'S1':>8s}  {'Δf':>8s}  {'B_med':>7s}  {'B_mean':>7s}  "
          f"{'flips[6,8,10]':>15s}")
    for r in res:
        flips = [r['per_V'][i]['flip_frac'] for i in range(len(r['per_V']))]
        print(f"  {r['delta']:6.3f}  {r['S1']:8.4f}  {r['Delta_f']:8.4f}  "
              f"{r['B_median']:7.3f}  {r['B_mean']:7.3f}  "
              f"[{flips[0]:.2f},{flips[1]:.2f},{flips[2]:.2f}]")
    Df = np.array([r['Delta_f'] for r in res])
    Bmed = np.array([r['B_median'] for r in res])
    Bmean = np.array([r['B_mean'] for r in res])
    print("\n  Fitting log(B) = const + α·log(Δf),  d_eff = 1 - α")
    fit_scaling(Df, Bmed, "median")
    fit_scaling(Df, Bmean, "mean")
    return data


def analyze_hi():
    path = '/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski_hi.json'
    if not os.path.exists(path):
        print(f"HI file not found: {path}")
        return None
    data = load(path)
    print("\n== HI (high-barrier extension) ==")
    # Data is flat list of {delta, S1, Δf, per_V}
    res = data if isinstance(data, list) else data.get('results', [])
    out = []
    for r in res:
        # Fit B from per_V tau_median vs V_eff (only using flipped>50%)
        per_V = r['per_V']
        V = np.array([x['V_eff'] for x in per_V])
        lt_med = np.log(np.array([x['tau_median'] for x in per_V]))
        lt_mean = np.log(np.array([x['tau_mean'] for x in per_V]))
        mask = np.array([x['flip_frac'] >= 0.3 for x in per_V])
        if mask.sum() >= 2:
            B_med = np.polyfit(V[mask], lt_med[mask], 1)[0]
            B_mean = np.polyfit(V[mask], lt_mean[mask], 1)[0]
        else:
            B_med = B_mean = float('nan')
        flips = [x['flip_frac'] for x in per_V]
        print(f"  delta={r['delta']}  S1={r['S1']:.3f}  Δf={r['Delta_f']:.3f}  "
              f"B_med={B_med:.3f}  B_mean={B_mean:.3f}  "
              f"flips={flips}")
        out.append((r['delta'], r['S1'], r['Delta_f'], B_med, B_mean, flips))
    return data, out


def combined_fit():
    """Combine LO and HI data to increase Δf range."""
    try:
        lo = load('/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski_lo.json')
    except FileNotFoundError:
        lo = None
    try:
        hi = load('/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski_hi.json')
    except FileNotFoundError:
        hi = None
    points = []  # (Δf, B_median, B_mean, flip_frac_min)
    if lo:
        for r in lo['results']:
            flips = [x['flip_frac'] for x in r['per_V']]
            points.append((r['Delta_f'], r['B_median'], r['B_mean'], min(flips)))
    if hi:
        res = hi if isinstance(hi, list) else hi.get('results', [])
        for r in res:
            per_V = r['per_V']
            V = np.array([x['V_eff'] for x in per_V])
            lt_med = np.log(np.array([x['tau_median'] for x in per_V]))
            lt_mean = np.log(np.array([x['tau_mean'] for x in per_V]))
            mask = np.array([x['flip_frac'] >= 0.3 for x in per_V])
            if mask.sum() >= 2:
                B_med = np.polyfit(V[mask], lt_med[mask], 1)[0]
                B_mean = np.polyfit(V[mask], lt_mean[mask], 1)[0]
                flips = [x['flip_frac'] for x in per_V]
                points.append((r['Delta_f'], B_med, B_mean, min(flips)))
    print("\n== COMBINED fit ==")
    print(f"  {'Δf':>8s}  {'B_med':>7s}  {'B_mean':>7s}  {'flip_min':>8s}")
    for p in sorted(points, key=lambda q: -q[0]):
        print(f"  {p[0]:8.3f}  {p[1]:7.3f}  {p[2]:7.3f}  {p[3]:8.2f}")
    arr = np.array(points)
    if len(arr) >= 3:
        Df = arr[:, 0]
        Bmed = arr[:, 1]
        Bmean = arr[:, 2]
        print("  (all points)")
        fit_scaling(Df, Bmed, "median/all")
        fit_scaling(Df, Bmean, "mean/all")
        # Filter to points with flip_min >= 0.3 to avoid heavy censoring
        good = arr[:, 3] >= 0.3
        if good.sum() >= 3:
            print(f"  (only flip_min>=0.3, N={good.sum()})")
            fit_scaling(Df[good], Bmed[good], "median/good")
            fit_scaling(Df[good], Bmean[good], "mean/good")


if __name__ == '__main__':
    analyze_lo()
    analyze_hi()
    combined_fit()
