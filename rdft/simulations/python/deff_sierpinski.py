#!/usr/bin/env python3
"""
Directly measure the effective dimension d_eff of nucleation on the
Sierpinski gasket via CNT scaling of the barrier vs driving force.

Model:  single-site birth-death with mean-field-across-neighbours coupling

  w+(x_i) = a + λ [ (1-α) x_i^2 + α/|N(i)| * Σ_{j∈N(i)} x_j^2 ]
  w-(x_i) = δ x_i + d x_i^3

Classical nucleation theory with effective dimension d_eff predicts

  Φ_c ∝ σ^{d_eff} / Δf^{d_eff - 1}

so that at fixed σ

  log Φ_c = const - (d_eff - 1) log Δf.

We vary the driving force Δf (by changing "a"), measure the Arrhenius slope
B of log τ vs V_eff for each parameter set, then fit B ~ Δf^{-(d_eff-1)}.

    d_eff  = 1  -> slope 0       (1D backbone)
    d_eff  = 1.365 (spectral)    slope -0.365
    d_eff  = 2                   slope -1
"""

import numpy as np
import json, time, os, sys, argparse
from multiprocessing import Pool, cpu_count

# Make module importable for multiprocessing workers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from numba import njit


# -------------------------------------------------------------------------
# Geometry: Sierpinski gasket
# -------------------------------------------------------------------------

def build_sierpinski_gasket(level: int):
    """Build Sierpinski gasket by recursive subdivision.

    We use coordinates (i, j) in a triangular lattice of side 2**level,
    restricted to the "allowed" sites of the fractal -- those whose
    coordinates' bitwise-AND is zero (Pascal-triangle mod 2).

    Returns: (coords, adj)
       coords : (N,2) float array of (x,y) positions
       adj    : list of lists, adj[i] = neighbours of site i
    """
    n = 1 << level  # side length = 2^level, so n+1 nodes along each edge
    # Enumerate (i,j) with 0 <= j <= i <= n, Pascal-triangle mod 2 allowed:
    # site allowed iff (i AND j) == 0 AND ((n-i) AND (i-j)) ... actually
    # simpler: use explicit recursive construction with three corner sets.

    # Use a deterministic recursive build:  the set S(level) of sites
    # placed on a unit triangle (vertices (0,0), (1,0), (0.5, sqrt3/2)).
    sqrt3 = np.sqrt(3.0) / 2.0

    def triangle(level, a, b, c, sites, edges):
        """a,b,c are the three corner coordinates of the current triangle."""
        if level == 0:
            # add the three corners as sites, and the three edges a-b, b-c, c-a
            ids = []
            for p in (a, b, c):
                key = (round(p[0], 9), round(p[1], 9))
                if key not in sites:
                    sites[key] = len(sites)
                ids.append(sites[key])
            i, j, k = ids
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))
            return
        ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        bc = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
        ca = ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
        # three sub-triangles
        triangle(level - 1, a, ab, ca, sites, edges)
        triangle(level - 1, ab, b, bc, sites, edges)
        triangle(level - 1, ca, bc, c, sites, edges)

    sites = {}
    edges = set()
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, sqrt3)
    triangle(level, A, B, C, sites, edges)

    N = len(sites)
    coords = np.zeros((N, 2))
    for key, idx in sites.items():
        coords[idx] = key
    adj = [[] for _ in range(N)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return coords, adj


def find_central_site(coords):
    """Return index of the site closest to the centroid of the gasket."""
    c = coords.mean(axis=0)
    d = np.sum((coords - c) ** 2, axis=1)
    return int(np.argmin(d))


# -------------------------------------------------------------------------
# Deterministic mean-field / quasi-potential
# -------------------------------------------------------------------------

def w_plus(x, a, lam, alpha, xbar):
    return a + lam * ((1 - alpha) * x * x + alpha * xbar * xbar)


def w_minus(x, delta, d3):
    return delta * x + d3 * x * x * x


def fixed_points_mean_field(a, lam, alpha, delta, d3):
    """Find fixed points of the uniform mean-field equation
        dx/dt = w+(x; xbar=x) - w-(x)
             =  a + lam*x^2 - delta*x - d3*x^3
    (alpha drops out when xbar=x).
    Returns sorted real roots in [0, x_max].
    """
    # d3 * x^3 - lam * x^2 + delta * x - a = 0
    coeffs = [d3, -lam, delta, -a]
    roots = np.roots(coeffs)
    real_roots = sorted(
        float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > -1e-9
    )
    return real_roots


def quasi_potential_barrier(a, lam, alpha, delta, d3):
    """Single-site "S_1": quasi-potential integral between low and high
    fixed points.

    For a 1-variable birth-death process the WKB/quasi-potential is
        V(x) = ∫ log( w-(y) / w+(y) ) dy
    The barrier S_1 is V(x_saddle) - V(x_low), where x_saddle is the
    unstable middle fixed point.  Integrate numerically.

    Also returns Δf := ∫_{x_low}^{x_high} [w-(y) - w+(y)] dy,
    which is the deterministic (drift-integrated) free-energy drop.
    """
    fps = fixed_points_mean_field(a, lam, alpha, delta, d3)
    # Need three fixed points for bistability
    if len(fps) < 3:
        return None
    x_low, x_sad, x_high = fps[0], fps[1], fps[-1]

    def integrand_V(y):
        wp = w_plus(y, a, lam, alpha, y)
        wm = w_minus(y, delta, d3)
        # Protect against division/log of zero
        if wp <= 0 or wm <= 0:
            return 0.0
        return np.log(wm / wp)

    # Use Simpson on dense grid
    from scipy.integrate import quad

    V_sad, _ = quad(integrand_V, x_low, x_sad, limit=200)
    V_high, _ = quad(integrand_V, x_low, x_high, limit=200)
    # Bulk driving force: ∫ (w- - w+)dx between low and high
    df_integrand = lambda y: w_minus(y, delta, d3) - w_plus(y, a, lam, alpha, y)
    Df, _ = quad(df_integrand, x_low, x_high, limit=200)
    return {
        "x_low": x_low,
        "x_sad": x_sad,
        "x_high": x_high,
        "S1": float(V_sad),           # single-site WKB barrier
        "V_high_minus_low": float(V_high),
        "Delta_f": float(-Df),         # >0 when high state is favoured
    }


# -------------------------------------------------------------------------
# Gillespie simulation on the gasket
# -------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _compute_rates_at_nb(site, counts, V_eff, a, lam, alpha, delta, d3,
                          adj_flat, adj_offsets):
    x = counts[site] / V_eff
    o0 = adj_offsets[site]
    o1 = adj_offsets[site + 1]
    k = o1 - o0
    s = 0.0
    if k > 0:
        for o in range(o0, o1):
            y = counts[adj_flat[o]] / V_eff
            s += y * y
        xbar_sq = s / k
    else:
        xbar_sq = 0.0
    rp = a + lam * ((1.0 - alpha) * x * x + alpha * xbar_sq)
    rm = delta * x + d3 * x * x * x
    if rp < 0.0:
        rp = 0.0
    if rm < 0.0:
        rm = 0.0
    return V_eff * rp, V_eff * rm


@njit(cache=True, fastmath=True)
def _gillespie_core(counts, V_eff, a, lam, alpha, delta, d3,
                     adj_flat, adj_offsets, N, center_idx,
                     threshold_count, t_max, max_events, seed):
    np.random.seed(seed)
    rates_plus = np.zeros(N)
    rates_minus = np.zeros(N)
    for i in range(N):
        rp, rm = _compute_rates_at_nb(i, counts, V_eff, a, lam, alpha, delta, d3,
                                       adj_flat, adj_offsets)
        rates_plus[i] = rp
        rates_minus[i] = rm
    total_p = rates_plus.sum()
    total_m = rates_minus.sum()

    t = 0.0
    events = 0
    flipped = False
    while t < t_max and events < max_events:
        total = total_p + total_m
        if total <= 0.0:
            break
        # Exponential
        u1 = np.random.random()
        if u1 <= 0.0:
            u1 = 1e-16
        dt = -np.log(u1) / total
        t += dt
        if t >= t_max:
            break
        u = np.random.random() * total
        if u < total_p:
            # find + site by linear scan
            cum = 0.0
            site = N - 1
            for i in range(N):
                cum += rates_plus[i]
                if u < cum:
                    site = i
                    break
            counts[site] += 1
        else:
            u2 = u - total_p
            cum = 0.0
            site = N - 1
            for i in range(N):
                cum += rates_minus[i]
                if u2 < cum:
                    site = i
                    break
            if counts[site] > 0:
                counts[site] -= 1
        # Update rates at site and all neighbours
        # site itself:
        rp_new, rm_new = _compute_rates_at_nb(site, counts, V_eff, a, lam, alpha, delta, d3,
                                               adj_flat, adj_offsets)
        total_p += rp_new - rates_plus[site]
        total_m += rm_new - rates_minus[site]
        rates_plus[site] = rp_new
        rates_minus[site] = rm_new
        # Neighbours:
        o0 = adj_offsets[site]; o1 = adj_offsets[site + 1]
        for o in range(o0, o1):
            j = adj_flat[o]
            rp_new, rm_new = _compute_rates_at_nb(j, counts, V_eff, a, lam, alpha, delta, d3,
                                                   adj_flat, adj_offsets)
            total_p += rp_new - rates_plus[j]
            total_m += rm_new - rates_minus[j]
            rates_plus[j] = rp_new
            rates_minus[j] = rm_new
        events += 1
        if counts[center_idx] >= threshold_count:
            flipped = True
            break
    return t, flipped, events


def run_one_gillespie(args):
    """Gillespie simulation; thin wrapper around numba core."""
    (a, lam, alpha, delta, d3,
     V_eff, adj_flat, adj_offsets, N, center_idx,
     x_low, x_high, t_max, max_events, seed) = args
    rng = np.random.default_rng(seed)
    counts = rng.poisson(V_eff * x_low, size=N).astype(np.int64)
    threshold_count = int(0.5 * (x_low + x_high) * V_eff)
    t, flipped, events = _gillespie_core(
        counts, float(V_eff), float(a), float(lam), float(alpha),
        float(delta), float(d3),
        adj_flat.astype(np.int64), adj_offsets.astype(np.int64),
        int(N), int(center_idx),
        int(threshold_count), float(t_max), int(max_events), int(seed) & 0x7fffffff,
    )
    return {"tau": float(t), "flipped": bool(flipped), "events": int(events)}


def flatten_adj(adj):
    offsets = np.zeros(len(adj) + 1, dtype=np.int64)
    for i, nb in enumerate(adj):
        offsets[i + 1] = offsets[i] + len(nb)
    flat = np.zeros(offsets[-1], dtype=np.int64)
    for i, nb in enumerate(adj):
        flat[offsets[i]:offsets[i + 1]] = np.array(nb, dtype=np.int64)
    return flat, offsets


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def measure_barrier(a, lam, alpha, delta, d3,
                    coords, adj, V_effs, n_trials,
                    t_max_factor=5e3, workers=None):
    """For a given parameter set, run Gillespie at multiple V_eff values
    and fit B = d log τ / d V_eff.
    """
    center = find_central_site(coords)
    qp = quasi_potential_barrier(a, lam, alpha, delta, d3)
    if qp is None:
        return None
    x_low, x_high = qp["x_low"], qp["x_high"]
    N = len(adj)
    adj_flat, adj_off = flatten_adj(adj)
    log_tau_median = []
    log_tau_mean = []
    flip_frac = []

    for V_eff in V_effs:
        # τ ~ exp(V_eff * B)  with B <~ 0.5; so τ at most ~ e^5 ~ 150.
        # Use a safety factor of 20x.  Also cap events per trial.
        t_max = t_max_factor
        max_events = 5_000_000
        args_list = [
            (a, lam, alpha, delta, d3,
             float(V_eff), adj_flat, adj_off, N, center,
             x_low, x_high, t_max, max_events,
             seed + int(V_eff * 100000))
            for seed in range(n_trials)
        ]
        # multiprocessing + numba cache is flaky on macOS spawn: force serial.
        out = []
        for k, a_ in enumerate(args_list):
            out.append(run_one_gillespie(a_))
            if (k + 1) % 25 == 0:
                print(f"      trial {k+1}/{len(args_list)}", flush=True)
        taus = np.array([r["tau"] for r in out])
        flipped = np.array([r["flipped"] for r in out])
        ff = flipped.mean()
        # Use only the trials that actually flipped for timing statistics
        if flipped.sum() >= max(5, n_trials // 4):
            taus_use = taus[flipped]
        else:
            taus_use = taus  # fall back; will be censored (underestimate)
        log_tau_median.append(float(np.log(np.median(taus_use))))
        log_tau_mean.append(float(np.log(np.mean(taus_use))))
        flip_frac.append(float(ff))
        print(f"    V_eff={V_eff:5.1f}  flips={ff:.2f}  "
              f"<τ>={np.mean(taus_use):.3e}  median τ={np.median(taus_use):.3e}")

    # Linear fit log τ vs V_eff -> slope = B
    V_arr = np.array(V_effs, dtype=float)
    slope_med, c_med = np.polyfit(V_arr, np.array(log_tau_median), 1)
    slope_mean, c_mean = np.polyfit(V_arr, np.array(log_tau_mean), 1)
    return {
        "a": a, "delta": delta,
        "S1": qp["S1"],
        "Delta_f": qp["Delta_f"],
        "x_low": qp["x_low"], "x_high": qp["x_high"], "x_sad": qp["x_sad"],
        "V_effs": list(V_effs),
        "log_tau_median": log_tau_median,
        "log_tau_mean": log_tau_mean,
        "flip_frac": flip_frac,
        "B_median_fit": float(slope_med),
        "B_mean_fit":   float(slope_mean),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--veffs", type=str, default="6,8,10")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lam", type=float, default=4.0)
    parser.add_argument("--d3", type=float, default=1.0)
    parser.add_argument("--knob", type=str, default="delta", choices=("a", "delta"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tmax-factor", type=float, default=3e3,
                        help="Maximum simulation time per trial (each unit of "
                             "V_eff adds ~e to τ).")
    parser.add_argument("--out", type=str,
                        default="/Users/sirsh/code/math/rdft/simulations/results/deff_sierpinski.json")
    args = parser.parse_args()

    print(f"Building Sierpinski gasket level {args.level} ...")
    coords, adj = build_sierpinski_gasket(args.level)
    N = len(adj)
    expected = (3 ** (args.level + 1) + 3) // 2
    print(f"  vertices = {N}  (expected (3^(L+1)+3)/2 = {expected})")
    assert N == expected, "Sierpinski gasket vertex count mismatch"
    degs = np.array([len(a_) for a_ in adj])
    print(f"  degrees min/mean/max = {degs.min()}/{degs.mean():.2f}/{degs.max()}")

    V_effs = [float(x) for x in args.veffs.split(",")]
    if args.knob == "a":
        a_values = [0.05, 0.10, 0.20, 0.30, 0.50]
        delta_values = [4.0] * 5
    else:
        # delta-knob at fixed a=0.05, lam=4 (wide Δf range ~ 2.4..8.8)
        a_values = [0.05] * 5
        delta_values = [2.0, 2.25, 2.5, 2.75, 3.0]

    nw = args.workers if args.workers > 0 else min(cpu_count(), args.trials)
    print(f"Using {nw} worker(s)")

    results = []
    t0 = time.time()
    for a_, d_ in zip(a_values, delta_values):
        print(f"\n=== a={a_:.3f}  delta={d_:.3f} ===")
        r = measure_barrier(
            a_, args.lam, args.alpha, d_, args.d3,
            coords, adj, V_effs, args.trials,
            t_max_factor=args.tmax_factor, workers=nw,
        )
        if r is None:
            print("  NO bistability, skipping.")
            continue
        print(f"  S1={r['S1']:.4f}  Δf={r['Delta_f']:.4f}  "
              f"B(median)={r['B_median_fit']:.3f}  B(mean)={r['B_mean_fit']:.3f}")
        results.append(r)

    # Fit log B vs log Δf  ->  slope α, d_eff = 1 - α
    Df_arr = np.array([r["Delta_f"] for r in results])
    B_med_arr = np.array([r["B_median_fit"] for r in results])
    B_mean_arr = np.array([r["B_mean_fit"] for r in results])

    # Clean: require positive B and Δf
    mask = (B_med_arr > 0) & (Df_arr > 0)
    if mask.sum() >= 3:
        x = np.log(Df_arr[mask])
        y_med = np.log(B_med_arr[mask])
        y_mean = np.log(B_mean_arr[mask])
        slope_m, c_m = np.polyfit(x, y_med, 1)
        slope_u, c_u = np.polyfit(x, y_mean, 1)
        # Standard error of slope
        def se(x, y, slope, inter):
            yhat = slope * x + inter
            resid = y - yhat
            n = len(x)
            if n <= 2:
                return float("nan")
            s2 = (resid ** 2).sum() / (n - 2)
            sx2 = ((x - x.mean()) ** 2).sum()
            return float(np.sqrt(s2 / sx2))
        se_m = se(x, y_med, slope_m, c_m)
        se_u = se(x, y_mean, slope_u, c_u)
        d_eff_med = 1.0 - slope_m
        d_eff_mean = 1.0 - slope_u
    else:
        slope_m = slope_u = float("nan")
        d_eff_med = d_eff_mean = float("nan")
        se_m = se_u = float("nan")

    # d_s of the gasket = log 3 / log (5/3)  (actually d_s = 2 log 3 / log 5)
    d_s_analytic = 2 * np.log(3) / np.log(5)

    summary = {
        "geometry": {
            "kind": "Sierpinski gasket",
            "level": args.level,
            "N": N,
            "d_s_analytic": float(d_s_analytic),
        },
        "params": {
            "alpha": args.alpha, "lam": args.lam, "d3": args.d3,
            "knob": args.knob, "V_effs": V_effs, "trials": args.trials,
        },
        "per_parameter": results,
        "fit_log_B_vs_log_Df": {
            "n_points": int(mask.sum()) if 'mask' in dir() else 0,
            "slope_median": float(slope_m),
            "slope_mean":   float(slope_u),
            "slope_median_se": float(se_m),
            "slope_mean_se":   float(se_u),
            "d_eff_median":  float(d_eff_med),
            "d_eff_mean":    float(d_eff_mean),
        },
        "wall_time_s": time.time() - t0,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"Sierpinski gasket L={args.level}, N={N}, d_s_analytic={d_s_analytic:.4f}")
    print(f"Slope of log B vs log Δf (median): {slope_m:.3f} ± {se_m:.3f}")
    print(f"  -> d_eff (median) = {d_eff_med:.3f}")
    print(f"Slope of log B vs log Δf (mean):   {slope_u:.3f} ± {se_u:.3f}")
    print(f"  -> d_eff (mean)   = {d_eff_mean:.3f}")
    print(f"Reference: d=1 -> 0,  d_s=1.365 -> -0.365,  d=2 -> -1")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
