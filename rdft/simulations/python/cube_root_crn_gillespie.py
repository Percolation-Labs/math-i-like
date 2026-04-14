"""
Gillespie simulation of the cube-root CRN and a detuned control.

  Reactions (A is the single species):
      A -> 3A    rate r1
      2A -> A    rate r2
      2A -> 3A   rate r3
  plus a spontaneous decay A -> 0 with rate q, added to make extinction possible.

  Tuned    : (r1, r2, r3) = (3, 2, 1)   -> phi coefficients (b, c) = (3, 1) ON C_3
  Detuned  : (r1, r2, r3) = (3, 2, 1.5) -> (b, c) = (4, -0.5)  off C_3

We tune q so that each CRN is approximately critical at low density, then
record for many trajectories:
  - total number of reaction events |T|
  - extinction time tau

and compare the tail exponents.  The AC+ prediction from the cube-root
branch point is tail exponent |T|^{-4/3} for the tuned CRN vs |T|^{-3/2}
for the detuned one (DP class).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import time


@dataclass
class CRNParams:
    r1: float   # A -> 3A
    r2: float   # 2A -> A
    r3: float   # 2A -> 3A
    q:  float   # A -> 0 (spontaneous decay)

    def name(self):
        return f'r1={self.r1}, r2={self.r2}, r3={self.r3}, q={self.q}'


def gillespie(params: CRNParams, N0: int = 1, max_events: int = 10_000_000,
              rng: np.random.Generator = None) -> Tuple[int, float, int]:
    """Simulate a single trajectory until extinction (N=0) or max_events.

    Returns (n_events, extinction_time, peak_population).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = N0
    t = 0.0
    n_events = 0
    peak = N

    while N > 0 and n_events < max_events:
        # propensities
        a1 = params.r1 * N                      # A -> 3A
        aq = params.q  * N                      # A -> 0
        if N >= 2:
            pairs = N * (N - 1)  # 2 * C(N,2) — we absorb factor of 2 into rates
            a2 = params.r2 * pairs / 2
            a3 = params.r3 * pairs / 2
        else:
            a2 = 0.0
            a3 = 0.0
        atot = a1 + aq + a2 + a3
        if atot <= 0:
            break
        # time step
        t += rng.exponential(1.0 / atot)
        # choose reaction
        u = rng.uniform() * atot
        if u < a1:
            N += 2  # A -> 3A, net +2
        elif u < a1 + aq:
            N -= 1  # A -> 0
        elif u < a1 + aq + a2:
            N -= 1  # 2A -> A, net -1
        else:
            N += 1  # 2A -> 3A, net +1
        n_events += 1
        if N > peak:
            peak = N

    return n_events, t, peak


def run_ensemble(params: CRNParams, n_traj: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sizes = np.empty(n_traj, dtype=np.int64)
    times = np.empty(n_traj)
    peaks = np.empty(n_traj, dtype=np.int64)
    truncated = 0
    for i in range(n_traj):
        n, t, p = gillespie(params, N0=1, max_events=2_000_000, rng=rng)
        sizes[i] = n
        times[i] = t
        peaks[i] = p
        if n >= 2_000_000:
            truncated += 1
    return dict(sizes=sizes, times=times, peaks=peaks, truncated=truncated,
                params=params)


def fit_tail_exponent(sizes: np.ndarray, lo_frac: float = 0.5, hi_frac: float = 0.99) -> float:
    """Fit P(|T|>=n) ~ n^{-(tau-1)} via the complementary CDF (cleaner than PMF)."""
    sizes = np.asarray(sizes)
    sizes = sizes[sizes > 0]
    sizes_sorted = np.sort(sizes)
    n = len(sizes_sorted)
    # empirical CCDF: F_bar(x) = fraction with size >= x
    # plot log(F_bar) vs log(size), fit slope in the middle range
    x = sizes_sorted
    y = 1 - np.arange(n) / n  # F_bar at x=x[i]
    mask = (x > np.quantile(x, lo_frac)) & (x < np.quantile(x, hi_frac))
    if mask.sum() < 10:
        return np.nan
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    # slope = -(tau - 1)  =>  tau = 1 - slope
    return 1 - slope


def main():
    # Calibrate q for approximate criticality.
    # mean-field drift: dN/dt = 2 r1 N + (r3 - r2) N(N-1)/2 - q N
    # linear part zero when q = 2 r1.  At this q, drift = (r3-r2) N(N-1)/2 < 0 for r3 < r2.
    # For r3 > r2 the system is super-critical.  Our cube-root has r3 < r2 (1 < 2), good.
    # Detuned case has r3 = 1.5 < r2 = 2, also subcritical at quadratic order.
    q = 6.0  # = 2 * r1 for r1 = 3, puts us at the critical "linearization" point

    tuned   = CRNParams(r1=3.0, r2=2.0, r3=1.0, q=q)
    detuned = CRNParams(r1=3.0, r2=2.0, r3=1.5, q=q)

    n_traj = 100_000
    print(f'Running {n_traj} trajectories per CRN, q = {q} ...')
    t0 = time.time()
    tuned_res   = run_ensemble(tuned,   n_traj, seed=1)
    detuned_res = run_ensemble(detuned, n_traj, seed=2)
    print(f'  elapsed: {time.time() - t0:.1f} s')

    for name, res in [('TUNED  (on C_3)', tuned_res), ('DETUNED', detuned_res)]:
        sizes = res['sizes']
        times = res['times']
        peaks = res['peaks']
        print(f'\n=== {name}:  {res["params"].name()} ===')
        print(f'  mean |T|   = {sizes.mean():.2f}, max |T| = {sizes.max()}, truncated: {res["truncated"]}')
        print(f'  mean tau   = {times.mean():.3f}, max tau = {times.max():.3f}')
        print(f'  mean peak  = {peaks.mean():.2f}, max peak = {peaks.max()}')
        tau = fit_tail_exponent(sizes, lo_frac=0.7, hi_frac=0.99)
        print(f'  fitted size-tail exponent tau = {tau:.4f}')
        print(f'    AC+ cube-root prediction:  tau = 4/3 = 1.3333')
        print(f'    DP square-root prediction: tau = 3/2 = 1.5000')

    # Save raw data for the figure script
    out = {
        'tuned_sizes':   tuned_res['sizes'],
        'tuned_times':   tuned_res['times'],
        'detuned_sizes': detuned_res['sizes'],
        'detuned_times': detuned_res['times'],
        'q': q,
    }
    from pathlib import Path
    outpath = Path(__file__).parent.parent.parent / 'paper' / 'cfac' / 'figures' / 'cube_root_gillespie_data.npz'
    np.savez(outpath, **out)
    print(f'\nSaved raw trajectory data to {outpath}')


if __name__ == '__main__':
    main()
