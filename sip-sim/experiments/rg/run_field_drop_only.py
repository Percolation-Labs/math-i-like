"""Run just the field_drop mode (the others completed in run_rg_tying.py)."""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from run_rg_tying import *

def run_field_drop():
    print("=" * 70)
    print("FIELD DROP ONLY (completing the tying experiment)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    V = 16
    D = 48
    H = 4
    N_EX = 8
    N_LAYER = 4
    STEPS = 5000
    BS = 64
    EVAP = 0.10
    seeds = [42, 137, 256]
    n_test_list = [N_EX, N_EX * 2, N_EX * 3]

    mode = "field_drop"
    cfg = MODE_CONFIGS[mode]

    print(f"\n  {mode}  (params: {cfg})")

    seed_results = {n: [] for n in n_test_list}
    field_infos = []
    t0 = time.perf_counter()

    for seed in seeds:
        results, fi, n_params = train_and_eval(
            cfg, V, N_EX, seed,
            d_model=D, n_head=H, n_layer=N_LAYER,
            train_steps=STEPS, batch_size=BS, evap_rate=EVAP,
            n_test_list=n_test_list)
        for n in n_test_list:
            seed_results[n].append(results[n])
        field_infos.append(fi)
        elapsed = time.perf_counter() - t0
        accs = [f"{results[n]:.3f}" for n in n_test_list]
        print(f"    seed={seed}  [{', '.join(accs)}]  "
              f"params={n_params}  ({elapsed:.0f}s)")

    accs_str = "  ".join(
        f"T={2*(n+1)}:{np.mean(seed_results[n]):.3f}+/-{np.std(seed_results[n]):.3f}"
        for n in n_test_list)
    print(f"    => {accs_str}")

    fi = field_infos[-1]
    infs = [fi.get(f"L{l}_influence", 0) for l in range(N_LAYER)]
    rets = [fi.get(f"L{l}_retain", 0) for l in range(N_LAYER)]
    print(f"    influence: {['%.4f' % v for v in infs]}")
    print(f"    retention: {['%.4f' % v for v in rets]}")
    mu = np.mean(infs)
    cv = np.std(infs) / (mu + 1e-8) if mu > 0 else 0
    print(f"    CV={cv:.3f}  mean_I={mu:.4f}")

    summary = {
        "label": mode,
        "mode": mode,
        "n_layer": N_LAYER,
        "n_params": n_params,
        "results": {},
        "field_info": field_infos,
    }
    for n in n_test_list:
        vals = seed_results[n]
        summary["results"][str(n)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "vals": vals,
        }

    out_path = OUT_DIR / "exp_field_drop_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    run_field_drop()
