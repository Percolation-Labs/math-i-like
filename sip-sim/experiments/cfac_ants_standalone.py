"""
Standalone lattice simulation for CFAC validation.

Implements EXACTLY the microscopic rules that match the mean-field theory,
bypassing sip-sim's step-function limitation on random_threshold.

Model:
  Species A (active), B (passive), on an L×L grid.
  Field φ(x,t) continuous.

  Per time step (small dt=1):
    For each B site:  B→A with probability k₊(1 + χ·φ_local)·dt
    For each A site:  A→B with probability k₋·dt
                      φ_local += δ      (deposit)
    For φ field:      φ *= (1 - λ)      (decay)
                      φ += σ·(∇²φ)      (diffusion)

The rate law k₊(1+χφ) is LINEAR in φ — exactly what mean-field theory assumes.

Mean-field steady state (spatially uniform):
  f_A = k₊(1+χφ)/(k₊(1+χφ) + k₋),   φ = δ·n·f_A/λ

Let g ≡ χδn/λ, α ≡ k₊/k₋.  Steady-state f_A solves:
  f_A · (1 + α(1 + g·f_A)) = α(1 + g·f_A)
  ⟹  α·g·f_A² - (α·g - α - 1)·f_A + (-α) = 0  [wait, redo]

Simpler: f_A = α(1 + g·f_A) / (1 + α(1 + g·f_A))
        f_A·(1 + α + α·g·f_A) = α + α·g·f_A
        f_A + α·f_A - α·g·f_A + α·g·f_A² = α
        α·g·f_A² + (1 + α - α·g)·f_A - α = 0

Quadratic in f_A:  f_A = [-(1+α-αg) + √((1+α-αg)² + 4α²g)] / (2αg)

This has a single physical root in [0,1] for α, g > 0. No bistability
in this quadratic — to get bistability we'd need a different nonlinearity.
The test is quantitative agreement: MF should match lattice within ~few
percent at tree level, with a 1/d correction at finite density.
"""

import numpy as np
import time
import csv
import os
from math import sqrt


def mean_field_f_A(n, k_plus, k_minus, chi, delta, lam):
    """Analytical MF steady-state f_A for linear-rate model."""
    alpha = k_plus / k_minus
    g = chi * delta * n / lam
    # α·g·f² + (1+α-αg)·f - α = 0
    a = alpha * g
    b = 1 + alpha - alpha * g
    c = -alpha
    if a < 1e-12:
        # g → 0: trivial f_A = α/(1+α)
        return alpha / (1 + alpha)
    disc = b * b - 4 * a * c
    f_A = (-b + sqrt(disc)) / (2 * a)
    return float(np.clip(f_A, 0.0, 1.0))


def run_lattice(n, L=80, k_plus=0.01, k_minus=0.05, chi=1.0,
                delta=2.0, lam=0.10, sigma=0.005,
                n_steps=3000, burn_in=1500, seed=42):
    """
    Lattice simulation with LINEAR rate k₊(1+χφ).

    Returns mean and std of f_A over the averaging window.
    """
    rng = np.random.default_rng(seed)

    # Initialize: Bernoulli occupancy at density n, 50/50 A/B split
    occ = rng.random((L, L)) < n          # which cells have an agent
    is_A = rng.random((L, L)) < 0.5       # of those, which are A
    is_A = is_A & occ
    is_B = occ & ~is_A
    phi = np.zeros((L, L), dtype=np.float64)

    # Precompute diffusion stencil (5-point Laplacian, periodic)
    def laplacian(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    f_A_trace = []
    phi_A_trace = []   # ⟨φ|A⟩
    phi_B_trace = []   # ⟨φ|B⟩
    phi_E_trace = []   # ⟨φ|empty⟩

    for t in range(n_steps):
        # B → A: per-site probability k₊(1+χφ)
        rate_BA = k_plus * (1.0 + chi * phi)   # per-site linear rate
        # Clip to [0,1] (should be far below 1 for reasonable params)
        p_BA = np.clip(rate_BA, 0.0, 1.0)
        convert_BA = is_B & (rng.random((L, L)) < p_BA)

        # A → B: per-site probability k₋
        p_AB = k_minus
        convert_AB = is_A & (rng.random((L, L)) < p_AB)

        # Apply transitions (simultaneously so no double-counting)
        new_is_A = (is_A & ~convert_AB) | convert_BA
        new_is_B = (is_B & ~convert_BA) | convert_AB
        is_A = new_is_A
        is_B = new_is_B

        # Deposit: each A site adds δ to its φ
        phi = phi + delta * is_A.astype(np.float64)

        # Decay
        phi = phi * (1.0 - lam)

        # Diffuse (explicit Euler, σ small enough to be stable)
        phi = phi + sigma * laplacian(phi)

        if t >= burn_in:
            n_A = int(is_A.sum())
            n_B = int(is_B.sum())
            if n_A + n_B > 0:
                f_A_trace.append(n_A / (n_A + n_B))
            if n_A > 0:
                phi_A_trace.append(float(phi[is_A].mean()))
            if n_B > 0:
                phi_B_trace.append(float(phi[is_B].mean()))
            empty = ~(is_A | is_B)
            if empty.any():
                phi_E_trace.append(float(phi[empty].mean()))

    f_A_arr = np.array(f_A_trace)
    return {
        'n': n,
        'f_A_mean': float(f_A_arr.mean()) if len(f_A_arr) else 0.0,
        'f_A_std': float(f_A_arr.std()) if len(f_A_arr) else 0.0,
        'phi_mean': float(phi.mean()),
        'phi_A': float(np.mean(phi_A_trace)) if phi_A_trace else 0.0,
        'phi_B': float(np.mean(phi_B_trace)) if phi_B_trace else 0.0,
        'phi_E': float(np.mean(phi_E_trace)) if phi_E_trace else 0.0,
        'n_A_final': int(is_A.sum()),
        'n_B_final': int(is_B.sum()),
    }


def main():
    # Parameters chosen so k₊(1+χφ) stays comfortably < 1 per step
    k_plus = 0.01
    k_minus = 0.05
    chi = 1.0
    delta = 2.0
    lam = 0.10
    sigma = 0.005
    L = 80

    print("=" * 72)
    print("CFAC STANDALONE LATTICE: linear rate k₊(1+χφ)")
    print(f"  k₊={k_plus}, k₋={k_minus}, χ={chi}, δ={delta}, λ={lam}, σ={sigma}")
    print(f"  Grid {L}×{L}, 3000 steps, burn-in 1500")
    print("=" * 72, flush=True)

    # Sanity: at what density does the linear-rate stay < 1?
    # At steady state, φ ≈ δ·n·f_A/λ ≈ 20·n·f_A. With f_A~0.5, n=0.2: φ≈2.
    # Then rate = 0.01·(1+2) = 0.03. Comfortably small. Good.

    densities = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]

    print(f"\n{'n':>6s} {'MF_naiv':>8s} {'lat_f':>7s} {'±':>6s} "
          f"{'⟨φ⟩':>6s} {'φ|A':>6s} {'φ|B':>6s} "
          f"{'LocalMF':>8s} {'%err_L':>7s}")
    print("-" * 75)

    results = []
    for n in densities:
        mf_naive = mean_field_f_A(n, k_plus, k_minus, chi, delta, lam)
        seed_runs = []
        t0 = time.time()
        for s in range(3):
            r = run_lattice(n, L=L, k_plus=k_plus, k_minus=k_minus,
                            chi=chi, delta=delta, lam=lam, sigma=sigma,
                            seed=42 + s * 13)
            seed_runs.append(r)
        f_m = np.mean([r['f_A_mean'] for r in seed_runs])
        f_s = np.std([r['f_A_mean'] for r in seed_runs])
        phi_m = np.mean([r['phi_mean'] for r in seed_runs])
        phiA = np.mean([r['phi_A'] for r in seed_runs])
        phiB = np.mean([r['phi_B'] for r in seed_runs])

        # CORRECTED mean field: use ⟨φ|B⟩ not ⟨φ⟩_global
        # At steady state: rate_B→A · (1-f_A) = k₋ · f_A
        # with rate_B→A = k₊(1 + χ·⟨φ|B⟩)
        rate_BA_local = k_plus * (1 + chi * phiB)
        mf_local = rate_BA_local / (rate_BA_local + k_minus)
        err_local = abs(f_m - mf_local)
        pct_local = 100 * err_local / mf_local if mf_local > 0 else 0
        dt = time.time() - t0

        print(f"{n:>6.3f} {mf_naive:>8.4f} {f_m:>7.4f} {f_s:>6.4f} "
              f"{phi_m:>6.3f} {phiA:>6.3f} {phiB:>6.3f} "
              f"{mf_local:>8.4f} {pct_local:>6.1f}%  ({dt:.0f}s)",
              flush=True)
        results.append({
            'n': n, 'mf_naive': mf_naive, 'lat_mean': f_m, 'lat_std': f_s,
            'phi_mean': phi_m, 'phi_A': phiA, 'phi_B': phiB,
            'mf_local': mf_local, 'pct_err_local': pct_local,
        })

    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'ac_plus_standalone.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved → {outpath}")

    mean_pct = np.mean([r['pct_err_local'] for r in results])
    max_pct = np.max([r['pct_err_local'] for r in results])
    print(f"\nSUMMARY (Local MF using ⟨φ|B⟩): "
          f"mean %err = {mean_pct:.1f}%, max %err = {max_pct:.1f}%")
    if mean_pct < 5:
        print("  ✓ CFAC TREE-LEVEL AGREEMENT (<5%) with ⟨φ|B⟩ closure")
    elif mean_pct < 15:
        print("  ~ Moderate agreement — 1-loop correction needed")
    else:
        print("  ✗ Still poor agreement — something deeper is wrong")

    # Naive MF vs local MF: demonstrate that the correlation ⟨φ|B⟩≠⟨φ⟩
    # is exactly the CFAC normal-ordering correction
    print("\nCONTRAST:")
    naive_err = np.mean([100 * abs(r['lat_mean'] - r['mf_naive']) /
                          r['mf_naive'] if r['mf_naive'] > 0 else 0
                          for r in results])
    print(f"  Naive MF (⟨φ⟩_global):   mean err = {naive_err:.1f}%")
    print(f"  Local MF (⟨φ|B⟩ closure): mean err = {mean_pct:.1f}%")


if __name__ == '__main__':
    main()
