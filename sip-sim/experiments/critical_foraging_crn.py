"""
Critical foraging CRN with hoarding dynamics.

Species: U (unladen ant), L (laden ant), F (food)
Reactions:
    U + F → L + F    (pick up: ant near food becomes laden. Food stays.)
    L + F → U + F    (hoard/drop: laden ant near food drops load. Food stays.)
    L → U            (random drop: small rate, creates new food clusters)

Field: φ (pheromone)
    Deposited by L only (laden ants mark successful return paths)
    Evaporates at rate λ
    Diffuses at rate σ
    U follows ∇φ (searching ants follow trails)

The hoarding rule: laden ants drop food NEAR existing food with high
probability (k₂ >> k₃). This concentrates food into clusters.
The absorbing state: all food in one cluster, frozen trail network.
The critical regime: trails compete — some reinforced, some die.

The critical parameter: evaporation λ.
    Low λ: all trails persist (frozen, no competition)
    High λ: only trails to nearest food survive (pruned network)
    Critical λ: trails form and dissolve dynamically (edge of chaos)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def make_spec(
    grid_size=80,
    ant_density=0.04,
    food_density=0.01,
    evaporation=0.15,
    diffusion=0.005,
    deposit_rate=5.0,
    bias_exponent=1.5,
    bias_base=0.5,
    momentum=1.0,
    noise=0.10,
    pickup_prob=0.3,     # prob of U→L when adjacent to F
    hoard_prob=0.5,      # prob of L→U when adjacent to F (drop near food)
    random_drop=0.02,    # prob of L→U anywhere (create new food cluster)
    seed=42,
    max_steps=2000,
):
    """
    The key insight: food is not consumed. It's a permanent landmark.
    Laden ants carry "virtual food" and drop it near existing food
    (hoarding). The random_drop creates new food clusters occasionally.

    We model food as a static agent that ants interact with by proximity.
    Ants don't actually move food — the "hoarding" is modelled by the
    laden→unladen transition being more likely near food.
    """
    return {
        "name": f"Foraging CRN λ={evaporation:.3f} food={food_density:.3f}",
        "topology": {
            "kind": "grid", "width": grid_size, "height": grid_size,
            "wrap": True, "neighbourhood": "moore",
        },
        "agent_kinds": [
            {
                "name": "unladen", "color": "#3498db",
                "initial_fraction": ant_density, "mobile": True,
                "state_schema": {"dir_r": "float", "dir_c": "float"},
                "initial_state": {"dir_r": 0, "dir_c": 0},
            },
            {
                "name": "laden", "color": "#e74c3c",
                "initial_fraction": 0.0, "mobile": True,
                "state_schema": {"dir_r": "float", "dir_c": "float"},
                "initial_state": {"dir_r": 0, "dir_c": 0},
            },
            {
                "name": "food", "color": "#27ae60",
                "initial_fraction": food_density, "mobile": False,
            },
        ],
        "env_fields": [
            {"name": "pheromone", "initial_value": 0.0},
        ],
        "field_operations": [
            {"field": "pheromone", "kind": "decay", "rate": evaporation},
            {"field": "pheromone", "kind": "diffuse", "rate": diffusion},
        ],
        "rules": [
            # ── U + F → L + F (pick up near food) ──
            {
                "name": "pick_up",
                "condition": {
                    "self_kind": "unladen",
                    "neighbour_count": {"kind": "food", "gte": 1},
                    "random_threshold": pickup_prob,
                },
                "action": {"kind": "transform", "new_kind": "laden"},
                "priority": 4,
            },
            # ── L + F → U + F (hoard: drop near existing food) ──
            {
                "name": "hoard_drop",
                "condition": {
                    "self_kind": "laden",
                    "neighbour_count": {"kind": "food", "gte": 1},
                    "random_threshold": hoard_prob,
                },
                "action": {"kind": "transform", "new_kind": "unladen"},
                "priority": 3,
            },
            # ── L → U (random drop, low rate) ──
            {
                "name": "random_drop",
                "condition": {
                    "self_kind": "laden",
                    "random_threshold": random_drop,
                },
                "action": {"kind": "transform", "new_kind": "unladen"},
                "priority": 2,
            },
            # ── L deposits pheromone (reward signal) ──
            {
                "name": "laden_deposits",
                "condition": {"self_kind": "laden"},
                "action": {
                    "kind": "modify_state",
                    "state_update": {"pheromone": f"+{deposit_rate}"},
                },
                "priority": 1,
            },
            # ── U follows pheromone (searching) ──
            {
                "name": "unladen_moves",
                "condition": {"self_kind": "unladen"},
                "action": {
                    "kind": "move",
                    "params": {
                        "bias_field": "pheromone",
                        "bias_exponent": bias_exponent,
                        "bias_base": bias_base,
                        "momentum": momentum,
                        "heading_alpha": 0.4,
                        "noise": noise,
                    },
                },
                "priority": 0,
            },
            # ── L follows pheromone (returning) ──
            {
                "name": "laden_moves",
                "condition": {"self_kind": "laden"},
                "action": {
                    "kind": "move",
                    "params": {
                        "bias_field": "pheromone",
                        "bias_exponent": bias_exponent,
                        "bias_base": bias_base,
                        "momentum": momentum,
                        "heading_alpha": 0.4,
                        "noise": noise * 0.3,
                    },
                },
                "priority": 0,
            },
        ],
        "observables": [
            {"name": "unladen", "kind": "count", "params": {"kind": "unladen"}},
            {"name": "laden", "kind": "count", "params": {"kind": "laden"}},
            {"name": "food", "kind": "count", "params": {"kind": "food"}},
            {"name": "pheromone_mean", "kind": "env_mean",
             "params": {"field": "pheromone"}},
            {"name": "pheromone_max", "kind": "env_max",
             "params": {"field": "pheromone"}},
            {"name": "trail_linearity", "kind": "trail_linearity",
             "params": {"field": "pheromone", "threshold_frac": 0.3}},
            {"name": "trail_turnover", "kind": "field_autocorrelation",
             "params": {"field": "pheromone", "lag": 200}},
        ],
        "max_steps": max_steps,
        "snapshot_interval": 200,
        "seed": seed,
    }


def run_and_measure(spec_dict):
    spec = SimulationSpec(**spec_dict)
    engine = build_engine(spec)
    trace = engine.run()

    phi = engine.state.env_grids['pheromone']
    mean_phi = phi.mean()
    max_phi = phi.max()
    conc = max_phi / mean_phi if mean_phi > 0.01 else 1.0

    g = spec_dict['topology']['width']
    bs = max(5, g // 10)
    bh, bw = g // bs, g // bs
    tr = phi[:bh*bs, :bw*bs]
    blocks = tr.reshape(bh, bs, bw, bs).mean(axis=(1, 3))
    B = float(np.var(blocks) / mean_phi**2) if mean_phi > 0.01 else 0

    last = trace.frames[-1].observables
    n_U = last.get('unladen', 0)
    n_L = last.get('laden', 0)
    n_F = last.get('food', 0)

    # Time series of laden fraction
    f_laden_series = []
    for f in trace.frames:
        nU = f.observables.get('unladen', 0)
        nL = f.observables.get('laden', 0)
        if nU + nL > 0:
            f_laden_series.append(nL / (nU + nL))

    L_vals = [f.observables.get('trail_linearity', 0)
              for f in trace.frames[-200:] if 'trail_linearity' in f.observables]
    A_vals = [f.observables.get('trail_turnover', 0)
              for f in trace.frames[-100:] if 'trail_turnover' in f.observables]

    return {
        'n_U': n_U, 'n_L': n_L, 'n_F': n_F,
        'f_laden': n_L / max(1, n_U + n_L),
        'f_laden_mean': np.mean(f_laden_series[-200:]) if f_laden_series else 0,
        'f_laden_std': np.std(f_laden_series[-200:]) if f_laden_series else 0,
        'conc_ratio': conc,
        'block_var': B,
        'linearity': np.mean(L_vals) if L_vals else 0,
        'autocorr': np.mean(A_vals) if A_vals else 0,
        'mean_phi': mean_phi,
    }


if __name__ == '__main__':
    print("="*72)
    print("CRITICAL FORAGING CRN (with hoarding)")
    print("  U + F → L + F (pick up)")
    print("  L + F → U + F (hoard/drop near food)")
    print("  L → U         (random drop, low rate)")
    print("  Only L deposits pheromone. U follows pheromone.")
    print("  Food is PERMANENT (not consumed).")
    print("="*72)

    # Sweep evaporation
    print(f"\nSweep: evaporation λ (food=0.01, ants=0.04)")
    print(f"{'λ':>6s} {'n_U':>5s} {'n_L':>5s} {'n_F':>5s} {'<f_L>':>6s} "
          f"{'σ(f_L)':>6s} {'C_rat':>6s} {'B':>6s} {'L':>6s} {'A':>6s}")
    print("-"*66)

    for lam in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        r = run_and_measure(make_spec(evaporation=lam))
        print(f"{lam:>6.2f} {r['n_U']:>5.0f} {r['n_L']:>5.0f} {r['n_F']:>5.0f} "
              f"{r['f_laden_mean']:>6.3f} {r['f_laden_std']:>6.3f} "
              f"{r['conc_ratio']:>6.1f} {r['block_var']:>6.3f} "
              f"{r['linearity']:>6.3f} {r['autocorr']:>6.3f}", flush=True)

    # Sweep food density
    print(f"\nSweep: food density (λ=0.15, ants=0.04)")
    print(f"{'ρ_F':>7s} {'n_U':>5s} {'n_L':>5s} {'n_F':>5s} {'<f_L>':>6s} "
          f"{'σ(f_L)':>6s} {'C_rat':>6s} {'B':>6s} {'L':>6s} {'A':>6s}")
    print("-"*68)

    for food in [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
        r = run_and_measure(make_spec(food_density=food, evaporation=0.15))
        print(f"{food:>7.3f} {r['n_U']:>5.0f} {r['n_L']:>5.0f} {r['n_F']:>5.0f} "
              f"{r['f_laden_mean']:>6.3f} {r['f_laden_std']:>6.3f} "
              f"{r['conc_ratio']:>6.1f} {r['block_var']:>6.3f} "
              f"{r['linearity']:>6.3f} {r['autocorr']:>6.3f}", flush=True)

    print(f"\nLook for: f_laden transition (0 → nonzero) as food density increases.")
    print(f"That's the critical point for trail formation.")
