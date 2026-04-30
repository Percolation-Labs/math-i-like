"""
Test: does CFAC branch-gap nucleation formula extend to a fractal substrate
via spectral substitution d -> d_s?

Build Sierpinski gasket at level L=4, run coupled bistable Gillespie dynamics
with different V_eff and alpha, measure MFPT to first-flip event, and compare
the scaling of log(tau) to CNT predictions with d=1, d_s=1.365, d=2.
"""

import numpy as np
import time
import math

rng = np.random.default_rng(20260411)

# ---------------------------------------------------------------------------
# 1) Build Sierpinski gasket by recursive identification of corners
# ---------------------------------------------------------------------------
def build_sierpinski(L):
    if L == 0:
        adj = [set() for _ in range(3)]
        for (a, b) in [(0, 1), (0, 2), (1, 2)]:
            adj[a].add(b); adj[b].add(a)
        return 3, adj, (0, 1, 2)

    nA, adjA, cA = build_sierpinski(L - 1)
    nB, adjB, cB = build_sierpinski(L - 1)
    nC, adjC, cC = build_sierpinski(L - 1)

    global_id = {}
    for i in range(nA):
        global_id[('A', i)] = i
    next_id = nA
    for i in range(nB):
        if i == cB[0]:
            global_id[('B', i)] = global_id[('A', cA[1])]
        else:
            global_id[('B', i)] = next_id
            next_id += 1
    for i in range(nC):
        if i == cC[0]:
            global_id[('C', i)] = global_id[('A', cA[2])]
        elif i == cC[1]:
            global_id[('C', i)] = global_id[('B', cB[2])]
        else:
            global_id[('C', i)] = next_id
            next_id += 1

    n_total = next_id
    adj_new = [set() for _ in range(n_total)]
    for (copy, local_adj) in [('A', adjA), ('B', adjB), ('C', adjC)]:
        for i, nbrs in enumerate(local_adj):
            gi = global_id[(copy, i)]
            for j in nbrs:
                gj = global_id[(copy, j)]
                adj_new[gi].add(gj)
                adj_new[gj].add(gi)

    corners_new = (
        global_id[('A', cA[0])],
        global_id[('B', cB[1])],
        global_id[('C', cC[2])],
    )
    return n_total, adj_new, corners_new


L = 4
N, adj, corners = build_sierpinski(L)
expected = (3**(L + 1) + 3) // 2
print(f"Sierpinski level {L}: N = {N} (expected {expected}) -- match: {N == expected}")
assert N == expected

adj_list = [sorted(list(s)) for s in adj]
deg = np.array([len(a) for a in adj_list])
print(f"  degrees: min={deg.min()}, max={deg.max()}, mean={deg.mean():.3f}")
print(f"  corners: {corners}")

# ---------------------------------------------------------------------------
# 2) Bistable birth-death process, S_1 computation
# ---------------------------------------------------------------------------
a_param = 0.1
lam     = 5.0
delta_p = 4.0
d_cube  = 1.0

roots = np.roots([1.0, -lam, delta_p, -a_param])
real_roots = sorted([float(r.real) for r in roots if abs(r.imag) < 1e-8])
print(f"\nSingle-site fixed points: {real_roots}")
x_low, x_mid, x_high = real_roots[0], real_roots[1], real_roots[-1]
print(f"  x_low={x_low:.4f}, x_mid={x_mid:.4f}, x_high={x_high:.4f}")

def integrand(x):
    wp = a_param + lam*x*x
    wm = delta_p*x + d_cube*x*x*x
    if wp <= 0 or wm <= 0:
        return 0.0
    return math.log(wm / wp)

xs_grid = np.linspace(x_low, x_mid, 4000)
ys_grid = np.array([integrand(x) for x in xs_grid])
S1 = np.trapz(ys_grid, xs_grid)
print(f"Single-site barrier S_1 = {S1:.4f} (expected ~0.456)")

# ---------------------------------------------------------------------------
# 3) Gillespie simulation, coupled
# ---------------------------------------------------------------------------
def simulate_once(V_eff, alpha, N_sites, adj_list, x_mid,
                  t_max=1e8, max_steps=3_000_000, seed=None):
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    n = np.full(N_sites, int(round(x_low * V_eff)), dtype=np.int64)

    n_nb = np.array([len(a) for a in adj_list], dtype=np.int64)
    n_sq = (n * n).astype(np.float64)
    sum_nbr_nsq = np.zeros(N_sites, dtype=np.float64)
    for i in range(N_sites):
        sum_nbr_nsq[i] = sum(n_sq[j] for j in adj_list[i])

    Vsq = float(V_eff) * V_eff

    def compute_rate(i):
        xsq_self = (n[i] * n[i]) / Vsq
        mean_nbr_xsq = sum_nbr_nsq[i] / n_nb[i] / Vsq
        x_self = n[i] / V_eff
        wp = a_param + lam * ((1.0 - alpha) * xsq_self + alpha * mean_nbr_xsq)
        wm = delta_p * x_self + d_cube * x_self ** 3
        return max(V_eff * wp, 0.0), max(V_eff * wm, 0.0)

    aplus = np.zeros(N_sites)
    aminus = np.zeros(N_sites)
    for i in range(N_sites):
        aplus[i], aminus[i] = compute_rate(i)

    t = 0.0
    step = 0
    thr = x_mid
    while step < max_steps and t < t_max:
        if (n / V_eff >= thr).any():
            return t, True
        total = aplus.sum() + aminus.sum()
        if total <= 0:
            return t, False
        dt = local_rng.exponential(1.0 / total)
        t += dt
        step += 1
        r = local_rng.random() * total
        cum_plus = np.cumsum(aplus)
        if r < cum_plus[-1]:
            idx = int(np.searchsorted(cum_plus, r))
            idx = min(idx, N_sites - 1)
            n[idx] += 1
            ds = 2 * n[idx] - 1
            n_sq[idx] = n[idx] * n[idx]
            for j in adj_list[idx]:
                sum_nbr_nsq[j] += ds
            for i in [idx] + adj_list[idx]:
                aplus[i], aminus[i] = compute_rate(i)
        else:
            r2 = r - cum_plus[-1]
            cum_minus = np.cumsum(aminus)
            idx = int(np.searchsorted(cum_minus, r2))
            idx = min(idx, N_sites - 1)
            if n[idx] == 0:
                continue
            n[idx] -= 1
            ds = -(2 * n[idx] + 1)
            n_sq[idx] = n[idx] * n[idx]
            for j in adj_list[idx]:
                sum_nbr_nsq[j] += ds
            for i in [idx] + adj_list[idx]:
                aplus[i], aminus[i] = compute_rate(i)

    if (n / V_eff >= thr).any():
        return t, True
    return t, False


# ---------------------------------------------------------------------------
# 4) Sweep
# ---------------------------------------------------------------------------
V_effs  = [3, 4, 5]
alphas  = [0.0, 0.3, 0.7]
N_TRIALS = {3: 150, 4: 100, 5: 60}
MAX_STEPS = 2_000_000
T_MAX = {3: 1e5, 4: 1e6, 5: 1e7}

results = {}
print("\n" + "="*70)
print("Running Gillespie MFPT sweep on Sierpinski gasket L=4 (N=123)")
print("="*70)

start_all = time.time()
for V_eff in V_effs:
    for alpha in alphas:
        n_tr = N_TRIALS[V_eff]
        ts = []
        n_success = 0
        t0 = time.time()
        for k in range(n_tr):
            t_hit, ok = simulate_once(
                V_eff, alpha, N, adj_list, x_mid,
                t_max=T_MAX[V_eff], max_steps=MAX_STEPS
            )
            ts.append((t_hit, ok))
            if ok:
                n_success += 1
        elapsed = time.time() - t0
        results[(V_eff, alpha)] = ts
        succ_times = [t for (t, ok) in ts if ok]
        if succ_times:
            mean_t = np.mean(succ_times)
            log_mean = np.log(mean_t)
            median_t = np.median(succ_times)
            print(f"  V={V_eff}, alpha={alpha:.2f}: {n_success}/{n_tr} succ, "
                  f"<t>={mean_t:.3e}, log<t>={log_mean:.3f}, "
                  f"med={median_t:.3e}, wall={elapsed:.1f}s")
        else:
            print(f"  V={V_eff}, alpha={alpha:.2f}: NO SUCCESSES in {n_tr}, wall={elapsed:.1f}s")

print(f"\nTotal wall time: {time.time()-start_all:.1f}s")

# ---------------------------------------------------------------------------
# 5) Analysis
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("Analysis")
print("="*70)

log_tau = {}
for (V, al), runs in results.items():
    succ = [t for (t, ok) in runs if ok]
    if succ:
        log_tau[(V, al)] = np.log(np.mean(succ))

print("\n-- log(tau) table --")
print(f"{'V_eff':>6}  " + "  ".join(f"alpha={al:.2f}".rjust(14) for al in alphas))
for V in V_effs:
    row = [f"{V:>6}"]
    for al in alphas:
        row.append(f"{log_tau[(V, al)]:.4f}".rjust(14) if (V, al) in log_tau else "   N/A".rjust(14))
    print("  ".join(row))

d_s = 2 * math.log(3) / math.log(5)
print(f"\nSierpinski spectral dimension d_s = {d_s:.4f}")

print("\n-- Linear fit  log(tau) = A + B * V_eff  (slope B = effective barrier per V) --")
slopes = {}
intercepts = {}
for al in alphas:
    Vs = np.array([V for V in V_effs if (V, al) in log_tau], dtype=float)
    ys = np.array([log_tau[(V, al)] for V in V_effs if (V, al) in log_tau])
    if len(Vs) >= 2:
        B, A = np.polyfit(Vs, ys, 1)
        slopes[al] = B
        intercepts[al] = A
        print(f"  alpha={al:.2f}:  B = {B:.4f}, A = {A:.4f}")

print()
print("-- CNT predictions --")
print(f"  Uncoupled (alpha=0): expect slope B ~ S_1 = {S1:.4f}")
print(f"                        intercept A ~ -log(N) = {-np.log(N):.4f} (any-site flips)")
print()
print("  Coupled (alpha>0): barrier becomes Phi(N_c) = sigma^d / Delta_f^(d-1)")
print("                      with N_c ~ (sigma/Delta_f)^d")
print("                      Phi/V_eff slope depends on d through the droplet shape.")
print()
print("-- Slope ratio r(alpha) = B(alpha) / B(0) --")
print("   (How much the effective barrier grows with coupling)")
if 0.0 in slopes and slopes[0.0] != 0:
    for al in alphas:
        if al in slopes:
            print(f"   alpha={al:.2f}:  r = {slopes[al]/slopes[0.0]:.3f}")

print()
print("-- Implied critical droplet size under each dimension --")
print("   If Phi(N_c) = S_1 * N_c^(1-1/d), then slope_coupled/S_1 = N_c^(1-1/d)")
print("   so N_c = (slope_coupled / S_1)^(d/(d-1)) for d != 1.")
for al in alphas:
    if al == 0.0 or al not in slopes:
        continue
    r_S1 = slopes[al] / S1
    print(f"\n   alpha={al:.2f}:  slope/S_1 = {r_S1:.3f}")
    for d_label, d_val in [("d=1    ", 1.0), ("d_s=1.365", d_s), ("d=2    ", 2.0)]:
        if d_val == 1.0:
            print(f"     {d_label}: d=1 gives N_c ambiguous (exponent 0)")
        else:
            expo = d_val / (d_val - 1.0)
            if r_S1 > 0:
                N_c = r_S1 ** expo
                print(f"     {d_label}: N_c = (slope/S_1)^(d/(d-1)) = {r_S1:.3f}^{expo:.3f} = {N_c:.2f}")

print("\nDone.")
