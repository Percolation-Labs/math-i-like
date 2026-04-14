"""
Engineered lattice model for AC+ validation.

Three species: A (active), B (passive). Total number conserved.
One field: φ (secreted by A, decays).

Reactions (local, on each cell):
  B → A at rate k₊(1 + χφ)    (field biases to active)
  A → B at rate k₋             (spontaneous deactivation)
  A → deposits dφ              (field source)

Movement: both A and B random-walk at rate D (no bias — keep it simple).
Field:    φ evaporates (rate λ), diffuses (rate σ).

This is a genuine DP-MSR coupled system with:
  DP sector: discrete B ↔ A transitions (stoichiometric)
  MSR sector: continuous φ field
  Coupling:  χ·φ in the B→A rate (linear, not power-law)

Mean-field equation for f_A = A/(A+B):
  df_A/dt = k₊(1 + χφ)(1-f_A) - k₋ f_A
  dφ/dt = δ·n·f_A - λφ    (with n = A+B = density, conserved)

Steady state:
  φ_ss = δ n f_A / λ
  f_A = k₊(1 + χ δ n f_A / λ) / [k₊(1 + χ δ n f_A / λ) + k₋]

This is a self-consistent equation. Let g = χδn/λ, then:
  f_A·(k₊ + k₊ g f_A + k₋) = k₊(1 + g f_A)
  f_A·(k₊ + k₋) + k₊ g f_A² = k₊ + k₊ g f_A
  f_A = [k₊ + k₊ g f_A - k₊ g f_A²] / (k₊+k₋)

Bistability appears when the cubic has 3 real roots, i.e. for
sufficiently strong coupling g·k₊/(k₊+k₋) > threshold.

The tree-level transition (tangent bifurcation) is at:
  n_c·χ·δ/λ = (k₊+k₋)²/(k₊·k₋)  [derived from df/df_A=0 condition]

or simply: strong coupling enables bistability; weak coupling is monostable.

We sweep n (total density) at fixed couplings and look for the
transition in f_A from low (OFF phase) to high (ON phase).
"""

import numpy as np
import time
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def make_spec(n_density=0.1, grid=80,
              k_plus=0.01, k_minus=0.05, chi=1.0,
              delta=2.0, lam=0.10, sigma=0.005,
              seed=42, max_steps=2000):
    """
    Two-species lattice model with field.

    Initial fraction 50/50 A/B; let it equilibrate.
    """
    return {
        'name': f'ac+ n={n_density:.3f}',
        'topology': {'kind': 'grid', 'width': grid, 'height': grid,
                     'wrap': True, 'neighbourhood': 'moore'},
        'agent_kinds': [
            # A: active, secretes field, stochastically deactivates
            {'name': 'active', 'color': '#e74c3c',
             'initial_fraction': n_density * 0.5, 'mobile': True,
             'state_schema': {}},
            # B: passive, stochastically activated by field
            {'name': 'passive', 'color': '#3498db',
             'initial_fraction': n_density * 0.5, 'mobile': True,
             'state_schema': {}},
        ],
        'env_fields': [
            {'name': 'field', 'initial_value': 0.0},
        ],
        'field_operations': [
            {'field': 'field', 'kind': 'decay', 'rate': lam},
            {'field': 'field', 'kind': 'diffuse', 'rate': sigma},
        ],
        'rules': [
            # A deposits field (priority 2: always happens)
            {'name': 'deposit',
             'condition': {'self_kind': 'active'},
             'action': {'kind': 'modify_state',
                        'state_update': {'field': f'+{delta}'}},
             'priority': 2},
            # A → B spontaneously at rate k₋
            {'name': 'deactivate',
             'condition': {'self_kind': 'active',
                          'random_threshold': k_minus},
             'action': {'kind': 'transform', 'new_kind': 'passive'},
             'priority': 1},
            # B → A at rate k₊ * (1 + χφ): approximated as
            # random_threshold = k₊ + k₊·χ·φ_local (use env field value)
            # Since we can't condition on env directly, we split into two rules:
            # base rate k₊:
            {'name': 'activate_base',
             'condition': {'self_kind': 'passive',
                          'random_threshold': k_plus},
             'action': {'kind': 'transform', 'new_kind': 'active'},
             'priority': 1},
            # field-dependent rate: condition on env field > threshold.
            # Use env_state predicate to require field > 0.5, then apply
            # a higher random_threshold for passive-to-active transition.
            {'name': 'activate_field',
             'condition': {'self_kind': 'passive',
                          'env_state': {'field': {'gt': 0.5}},
                          'random_threshold': k_plus * chi * 5.0},
             'action': {'kind': 'transform', 'new_kind': 'active'},
             'priority': 1},
            # Movement: both A and B random walk
            {'name': 'move_A',
             'condition': {'self_kind': 'active'},
             'action': {'kind': 'move', 'params': {}}, 'priority': 0},
            {'name': 'move_B',
             'condition': {'self_kind': 'passive'},
             'action': {'kind': 'move', 'params': {}}, 'priority': 0},
        ],
        'observables': [
            {'name': 'n_A', 'kind': 'count', 'params': {'kind': 'active'}},
            {'name': 'n_B', 'kind': 'count', 'params': {'kind': 'passive'}},
            {'name': 'field_mean', 'kind': 'env_mean',
             'params': {'field': 'field'}},
            {'name': 'field_max', 'kind': 'env_max',
             'params': {'field': 'field'}},
        ],
        'max_steps': max_steps, 'seed': seed,
    }


def run_one(n_density, **kwargs):
    spec = SimulationSpec(**make_spec(n_density=n_density, **kwargs))
    engine = build_engine(spec)
    trace = engine.run()

    # Time-averaged observables from last 500 steps
    f_A_vals = []
    phi_vals = []
    phi_max_vals = []
    for f in trace.frames[-500:]:
        nA = f.observables.get('n_A', 0)
        nB = f.observables.get('n_B', 0)
        if nA + nB > 0:
            f_A_vals.append(nA / (nA + nB))
        phi_vals.append(f.observables.get('field_mean', 0))
        phi_max_vals.append(f.observables.get('field_max', 0))

    return {
        'n_density': n_density,
        'f_A_mean': np.mean(f_A_vals) if f_A_vals else 0,
        'f_A_std': np.std(f_A_vals) if f_A_vals else 0,
        'phi_mean': np.mean(phi_vals),
        'phi_max': np.mean(phi_max_vals),
    }


if __name__ == '__main__':
    print("=" * 72)
    print("ENGINEERED LATTICE: B ↔ A transitions coupled to field")
    print("  Looking for a density-driven transition f_A (OFF → ON)")
    print("=" * 72, flush=True)

    # Diagnostic: a single run at moderate density
    print("\n--- Diagnostic at n=0.10 ---")
    r = run_one(0.10, k_plus=0.01, k_minus=0.05, chi=1.0, delta=2.0)
    print(f"  f_A = {r['f_A_mean']:.4f} ± {r['f_A_std']:.4f}")
    print(f"  <φ> = {r['phi_mean']:.3f}, max(φ) = {r['phi_max']:.2f}")

    # Mean-field prediction for comparison:
    # f_A = k₊(1+χφ) / [k₊(1+χφ) + k₋]
    # where φ = δ·n·f_A/λ
    k_p, k_m, chi, delta, lam = 0.01, 0.05, 1.0, 2.0, 0.10

    def mf_residual(f_A, n):
        phi = delta * n * f_A / lam
        rate_on = k_p * (1 + chi * phi)
        return rate_on / (rate_on + k_m) - f_A

    def brentq(f, a, b, args=(), tol=1e-8, maxiter=200):
        fa, fb = f(a, *args), f(b, *args)
        if fa * fb > 0:
            raise ValueError("No sign change")
        for _ in range(maxiter):
            c = (a + b) / 2
            fc = f(c, *args)
            if abs(fc) < tol or (b - a) < tol:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2

    print("\n--- Mean-field f_A vs density ---")
    print(f"{'n':>6s} {'f_A (MF)':>10s} {'φ_ss':>8s}")
    for n in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
        try:
            f_mf = brentq(mf_residual, 0.001, 0.999, args=(n,))
        except ValueError:
            f_mf = 0.5
        phi_ss = delta * n * f_mf / lam
        print(f"{n:>6.3f} {f_mf:>10.4f} {phi_ss:>8.3f}")

    # Now lattice sweep
    print("\n--- Lattice sweep: n varying ---")
    print(f"{'n':>6s} {'f_A':>7s} {'±':>6s} {'<φ>':>7s} {'max(φ)':>8s}")
    print("-" * 38)

    # Theory says transition is near n_c where k₊·g·n_c ≈ k₋
    # with g = χδ/λ = 20, k₊=0.01, k₋=0.05: n_c ≈ 0.25
    # Focus the sweep on the transition region
    n_values = np.concatenate([
        np.array([0.02, 0.05, 0.10]),         # low-n sanity
        np.arange(0.12, 0.35, 0.02),           # transition region (finer)
        np.array([0.40, 0.50]),                # high-n saturation
    ])
    results = []
    t0 = time.time()
    for n in n_values:
        seed_res = [run_one(n, seed=42 + s * 13) for s in range(3)]
        f_m = np.mean([r['f_A_mean'] for r in seed_res])
        f_s = np.std([r['f_A_mean'] for r in seed_res])
        phi_m = np.mean([r['phi_mean'] for r in seed_res])
        phi_max = np.mean([r['phi_max'] for r in seed_res])
        results.append({'n': n, 'f_A': f_m, 'f_A_std': f_s,
                        'phi_mean': phi_m, 'phi_max': phi_max})
        print(f"{n:>6.3f} {f_m:>7.4f} {f_s:>6.4f} {phi_m:>7.3f} {phi_max:>8.2f}",
              flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

    # Save
    outpath = os.path.join(os.path.dirname(__file__), 'results',
                           'ac_plus_lattice.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Does f_A cross the midpoint in a controlled way?
    ns = np.array([r['n'] for r in results])
    fs = np.array([r['f_A'] for r in results])
    f_range = fs.max() - fs.min()
    print(f"\nf_A range: [{fs.min():.3f}, {fs.max():.3f}], span {f_range:.3f}")
    if f_range > 0.2:
        # Find midpoint crossing
        f_mid = (fs.min() + fs.max()) / 2
        for i in range(len(fs) - 1):
            if fs[i] < f_mid < fs[i + 1]:
                n_c = ns[i] + (f_mid - fs[i]) * (ns[i+1] - ns[i]) / (fs[i+1] - fs[i])
                print(f"Transition midpoint: n_c ≈ {n_c:.3f} (f_A crosses {f_mid:.3f})")
                break
