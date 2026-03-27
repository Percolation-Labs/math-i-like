#!/usr/bin/env python3
"""
Prion propagation simulator on lattices.
Multi-species Gillespie tau-leaping for H + M → 2M, ∅ → H, H → ∅, M → ∅.

This is the first spatial stochastic simulation of prion propagation
analyzed via field theory. The goal is to measure the density decay
exponent at criticality and compare with directed percolation (DP).

Usage:
    python simulations/python/prion_sim.py [--dim 1] [--size 200] [-n 5000] [--tmax 2000]
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import time
import argparse
from multiprocessing import Pool, cpu_count


@dataclass
class PrionParams:
    """Prion propagation rate parameters."""
    beta: float     # H + M → 2M (template conversion)
    lam: float      # ∅ → H (healthy protein production)
    mu_H: float     # H → ∅ (healthy degradation)
    mu_M: float     # M → ∅ (misfolded clearance)
    D_H: float = 1.0   # H diffusion rate
    D_M: float = 1.0   # M diffusion rate

    @property
    def R0(self) -> float:
        """Basic reproduction number."""
        H0 = self.lam / self.mu_H
        return self.beta * H0 / self.mu_M

    @property
    def H0(self) -> float:
        """Disease-free steady-state H concentration."""
        return self.lam / self.mu_H

    def at_criticality(self) -> 'PrionParams':
        """Adjust beta so R0 = 1 exactly."""
        beta_c = self.mu_H * self.mu_M / self.lam
        return PrionParams(beta_c, self.lam, self.mu_H, self.mu_M, self.D_H, self.D_M)

    def __repr__(self):
        return (f"PrionParams(β={self.beta:.4f}, λ={self.lam}, "
                f"μ_H={self.mu_H}, μ_M={self.mu_M}, R₀={self.R0:.4f})")


def build_1d_neighbors(L: int) -> List[List[int]]:
    """Periodic 1D lattice neighbors."""
    return [[(i - 1) % L, (i + 1) % L] for i in range(L)]


def build_2d_neighbors(L: int) -> List[List[int]]:
    """Periodic 2D lattice neighbors."""
    N = L * L
    adj = [[] for _ in range(N)]
    for y in range(L):
        for x in range(L):
            i = y * L + x
            adj[i] = [
                y * L + (x + 1) % L,
                y * L + (x - 1) % L,
                ((y + 1) % L) * L + x,
                ((y - 1) % L) * L + x,
            ]
    return adj


def build_3d_neighbors(L: int) -> List[List[int]]:
    """Periodic 3D lattice neighbors."""
    N = L * L * L
    adj = [[] for _ in range(N)]
    for z in range(L):
        for y in range(L):
            for x in range(L):
                i = z * L * L + y * L + x
                adj[i] = [
                    z * L * L + y * L + (x + 1) % L,
                    z * L * L + y * L + (x - 1) % L,
                    z * L * L + ((y + 1) % L) * L + x,
                    z * L * L + ((y - 1) % L) * L + x,
                    ((z + 1) % L) * L * L + y * L + x,
                    ((z - 1) % L) * L * L + y * L + x,
                ]
    return adj


def run_single_realization(args) -> Optional[dict]:
    """
    Run one realization of prion propagation.
    Tau-leaping with dt = 1.0 (same as Rust simulator).
    """
    params, adj, N, t_max, record_times, seed, initial_M_sites = args
    rng = np.random.default_rng(seed)

    # State: H[i], M[i] = particle counts at site i
    H = np.zeros(N, dtype=np.int32)
    M = np.zeros(N, dtype=np.int32)

    # Initialize: H at steady state, seed M at center
    H[:] = rng.poisson(params.H0, size=N).astype(np.int32)
    for site in initial_M_sites:
        M[site] = 1

    n_neighbors = np.array([len(adj[i]) for i in range(N)])
    dt = 1.0

    # Recording
    record_idx = 0
    times_out = []
    M_total_out = []
    H_total_out = []
    M_sites_out = []  # number of sites with M > 0

    t = 0.0
    while t < t_max:
        # Record if needed
        while record_idx < len(record_times) and t >= record_times[record_idx]:
            M_tot = int(M.sum())
            H_tot = int(H.sum())
            M_occupied = int((M > 0).sum())
            times_out.append(record_times[record_idx])
            M_total_out.append(M_tot)
            H_total_out.append(H_tot)
            M_sites_out.append(M_occupied)
            record_idx += 1

        # Check extinction
        if M.sum() == 0:
            # Fill remaining record times with zeros
            while record_idx < len(record_times):
                times_out.append(record_times[record_idx])
                M_total_out.append(0)
                H_total_out.append(int(H.sum()))
                M_sites_out.append(0)
                record_idx += 1
            break

        # --- REACTIONS (tau-leaping, dt=1) ---

        # 1. ∅ → H: Poisson production at each site
        H_produced = rng.poisson(params.lam * dt, size=N).astype(np.int32)
        H += H_produced

        # 2. H → ∅: binomial degradation
        H_degraded = rng.binomial(H, min(params.mu_H * dt, 1.0)).astype(np.int32)
        H -= H_degraded

        # 3. M → ∅: binomial clearance
        M_degraded = rng.binomial(M, min(params.mu_M * dt, 1.0)).astype(np.int32)
        M -= M_degraded

        # 4. H + M → 2M: conversion
        # At each site, number of H-M pairs = H[i] * M[i]
        # Each pair converts with probability beta * dt
        # Net effect: some H become M
        pairs = H * M
        mask = pairs > 0
        if mask.any():
            # Expected conversions at each site
            mean_conversions = params.beta * dt * pairs[mask]
            # Cap at available H
            conversions = np.minimum(
                rng.poisson(mean_conversions).astype(np.int32),
                H[mask]
            )
            H[mask] -= conversions
            M[mask] += conversions

        # Clamp to non-negative (safety)
        np.maximum(H, 0, out=H)
        np.maximum(M, 0, out=M)

        # --- DIFFUSION ---
        # H diffusion
        if params.D_H > 0:
            _diffuse(H, adj, n_neighbors, params.D_H, dt, rng, N)

        # M diffusion
        if params.D_M > 0:
            _diffuse(M, adj, n_neighbors, params.D_M, dt, rng, N)

        t += dt

        # Safety: cap total particles to prevent blowup
        total = H.sum() + M.sum()
        if total > 500_000:
            scale = 250_000 / total
            H = rng.binomial(H, scale).astype(np.int32)
            M = rng.binomial(M, scale).astype(np.int32)

    survived = M.sum() > 0

    return {
        'times': times_out,
        'M_total': M_total_out,
        'H_total': H_total_out,
        'M_sites': M_sites_out,
        'survived': survived,
    }


def _diffuse(state, adj, n_neighbors, D, dt, rng, N):
    """Diffuse particles: each hops to a random neighbor with prob D*dt."""
    hop_prob = min(D * dt, 1.0)
    hoppers = rng.binomial(state, hop_prob).astype(np.int32)
    state -= hoppers

    # Distribute hoppers to neighbors
    for i in range(N):
        if hoppers[i] > 0:
            nbrs = adj[i]
            k = len(nbrs)
            if k > 0:
                # Multinomial: distribute hoppers[i] among k neighbors
                dest = rng.choice(k, size=hoppers[i])
                for d in dest:
                    state[nbrs[d]] += 1


def run_ensemble(params: PrionParams, dim: int, L: int,
                 n_realizations: int, t_max: float,
                 n_workers: int = None) -> dict:
    """Run ensemble of prion simulations."""

    # Build graph
    if dim == 1:
        N = L
        adj = build_1d_neighbors(L)
        d_s = 1.0
    elif dim == 2:
        N = L * L
        adj = build_2d_neighbors(L)
        d_s = 2.0
    elif dim == 3:
        N = L * L * L
        adj = build_3d_neighbors(L)
        d_s = 3.0
    else:
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

    # Record times (log-spaced)
    record_times = np.unique(np.logspace(0, np.log10(t_max), 80).astype(int)).astype(float).tolist()

    # Initial M seeds: center site(s)
    center = N // 2
    initial_M = [center]

    print(f"Prion simulation: {dim}D lattice (L={L}, N={N})")
    print(f"Parameters: {params}")
    print(f"R₀ = {params.R0:.6f}")
    print(f"Realizations: {n_realizations}, t_max: {t_max}")

    if n_workers is None:
        n_workers = min(cpu_count(), n_realizations)

    # Build args for each realization
    args_list = [
        (params, adj, N, t_max, record_times, seed, initial_M)
        for seed in range(n_realizations)
    ]

    t0 = time.time()

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results_list = pool.map(run_single_realization, args_list)
    else:
        results_list = [run_single_realization(a) for a in args_list]

    wall_time = time.time() - t0
    print(f"Completed in {wall_time:.1f}s ({wall_time/60:.1f}min)")

    # Aggregate
    n_survived = sum(1 for r in results_list if r['survived'])
    print(f"Survived: {n_survived}/{n_realizations} = {n_survived/n_realizations:.3f}")

    # Compute moments of M_total
    n_times = len(record_times)
    M_all = np.zeros((n_realizations, n_times))
    for i, r in enumerate(results_list):
        n = min(len(r['M_total']), n_times)
        M_all[i, :n] = r['M_total'][:n]

    # Conditional (surviving) and unconditional moments
    survived_mask = np.array([r['survived'] for r in results_list])

    moments = {}
    for p in [1, 2]:
        # Unconditional: average over all realizations
        moments[f'M^{p}_uncond'] = np.mean(M_all**p, axis=0).tolist()
        # Conditional: average over surviving only
        if survived_mask.sum() > 10:
            moments[f'M^{p}_cond'] = np.mean(M_all[survived_mask]**p, axis=0).tolist()

    # Also track M density (per site)
    M_density = np.mean(M_all / N, axis=0)

    # Fit power-law exponent to M density decay
    # For DP at criticality: ρ(t) ~ t^{-δ} with δ ≈ 0.159 (d=1)
    # Use late-time window
    t_arr = np.array(record_times)
    rho = M_density

    # Find fitting window: where rho > 0 and t > 10
    mask_fit = (rho > 0) & (t_arr > 10) & (t_arr < t_max * 0.8)
    if mask_fit.sum() > 5:
        log_t = np.log(t_arr[mask_fit])
        log_rho = np.log(rho[mask_fit])
        # Linear regression in log-log
        coeffs = np.polyfit(log_t, log_rho, 1)
        delta_fit = -coeffs[0]  # ρ ~ t^{-δ}
        print(f"\nFitted density decay: ρ(t) ~ t^{{-{delta_fit:.3f}}}")
        print(f"DP prediction (d={dim}): δ = {_dp_delta(dim):.3f}")
    else:
        delta_fit = None
        print("\nInsufficient data for power-law fit")

    output = {
        'dim': dim,
        'L': L,
        'N': N,
        'd_s': d_s,
        'params': {
            'beta': params.beta,
            'lambda': params.lam,
            'mu_H': params.mu_H,
            'mu_M': params.mu_M,
            'R0': params.R0,
        },
        'n_realizations': n_realizations,
        'n_survived': n_survived,
        'survival_fraction': n_survived / n_realizations,
        'times': record_times,
        'M_density': M_density.tolist(),
        'moments': moments,
        'fitted_delta': delta_fit,
        'dp_delta': _dp_delta(dim),
        'wall_time_secs': wall_time,
    }

    return output


def _dp_delta(d):
    """DP density decay exponent δ in d dimensions (numerical values)."""
    # ρ(t) ~ t^{-δ} at criticality
    # From Hinrichsen (2000) Table 1
    dp = {1: 0.159464, 2: 0.4505, 3: 0.732}
    if d >= 4:
        return 1.0  # mean-field
    return dp.get(d, None)


def scan_R0(dim: int, L: int, n_realizations: int, t_max: float,
            base_params: PrionParams, R0_values: list) -> list:
    """Scan across R0 to find the critical point and measure exponents."""
    results = []
    for R0_target in R0_values:
        # Adjust beta to hit target R0
        beta = R0_target * base_params.mu_H * base_params.mu_M / base_params.lam
        p = PrionParams(beta, base_params.lam, base_params.mu_H, base_params.mu_M)
        print(f"\n--- R₀ = {R0_target:.3f} (β = {beta:.4f}) ---")
        r = run_ensemble(p, dim, L, n_realizations, t_max)
        r['R0_target'] = R0_target
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(description='Prion propagation simulator')
    parser.add_argument('--dim', type=int, default=1, help='Lattice dimension (1, 2, 3)')
    parser.add_argument('--size', type=int, default=200, help='Lattice size per dimension')
    parser.add_argument('-n', '--realizations', type=int, default=5000, help='Number of realizations')
    parser.add_argument('--tmax', type=float, default=2000, help='Max simulation time')
    parser.add_argument('--mu-H', type=float, default=0.5, help='H degradation rate')
    parser.add_argument('--mu-M', type=float, default=0.1, help='M clearance rate')
    parser.add_argument('--lam', type=float, default=1.0, help='H production rate')
    parser.add_argument('--scan', action='store_true', help='Scan R0 around criticality')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Parallel workers')
    args = parser.parse_args()

    base = PrionParams(
        beta=1.0,  # will be adjusted
        lam=args.lam,
        mu_H=args.mu_H,
        mu_M=args.mu_M,
    )

    if args.scan:
        # Scan R0 around criticality
        R0_values = [0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2]
        results = scan_R0(args.dim, args.size, args.realizations,
                         args.tmax, base, R0_values)
        output = {'scan': results, 'dim': args.dim, 'L': args.size}
    else:
        # Single run at criticality
        params = base.at_criticality()
        output = run_ensemble(params, args.dim, args.size,
                            args.realizations, args.tmax, args.workers)

    # Save
    out_file = args.output or f'prion_{args.dim}d_L{args.size}_n{args.realizations}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
