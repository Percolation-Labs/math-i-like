"""Pure Gillespie runs on 3-site ring, alpha in {0.0, 0.1, 0.3, 0.5, 0.7},
V in {4, 6, 8, 10}, 500+ trials each."""
import sys, time
sys.path.insert(0, '/Users/sirsh/code/math/rdft/paper/wip')
from complex_branch_gap import gillespie_3ring
import numpy as np

ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7]
V_LIST = [4, 6, 8, 10]
N_TRIALS = 500

results = {}
for alpha in ALPHAS:
    print(f"\n=== alpha = {alpha} ===", flush=True)
    logT = []
    Vs_valid = []
    for k, V in enumerate(V_LIST):
        t0 = time.time()
        times, succ, tot = gillespie_3ring(alpha, V, n_trials=N_TRIALS,
                                            t_max=1e11,
                                            rng_seed=42000 + int(alpha * 1000) + k)
        if succ >= 20:
            m = np.mean(times)
            std_m = np.std(times, ddof=1) / np.sqrt(succ) / m  # rel SE of mean
            lt = np.log(m)
            logT.append(lt)
            Vs_valid.append(V)
            print(f"  V={V}: {succ}/{tot} succ, <t>={m:.3e}, "
                  f"log<t>={lt:.4f} (se/mean={std_m:.3f}, wall={time.time()-t0:.1f}s)",
                  flush=True)
        else:
            print(f"  V={V}: only {succ}/{tot} succ (skipped, wall={time.time()-t0:.1f}s)",
                  flush=True)
    if len(Vs_valid) >= 2:
        slope, inter = np.polyfit(Vs_valid, logT, 1)
        print(f"  => slope S = {slope:.4f}, intercept = {inter:.4f}", flush=True)
        results[alpha] = (slope, inter, Vs_valid, logT)
    else:
        results[alpha] = (None, None, Vs_valid, logT)

print("\n\n=== SUMMARY ===", flush=True)
for alpha in ALPHAS:
    s, i, _, _ = results[alpha]
    if s is not None:
        print(f"  alpha={alpha:.2f}: S = {s:.4f}  (intercept={i:.4f})", flush=True)
    else:
        print(f"  alpha={alpha:.2f}: (insufficient successes)", flush=True)
