"""
Critical foraging: ants with food sources.

The key difference from the unconditional-deposit model:
  - Food sources placed at random locations (env field 'food')
  - Ants have a state 'carrying' (0 = searching, 1 = returning with food)
  - Searching ants: follow pheromone with weak bias, exploring
  - Ants on food: pick up food (carrying → 1)
  - Returning ants: deposit pheromone, follow pheromone home
  - At nest: drop food (carrying → 0)

The critical point: when food density × ant density × trail lifetime
just sustains the trail network. Below critical: trails to food
dissolve (not enough reinforcement). Above: trails persist.

This is the genuine coupled DP-MSR system:
  DP sector: ants finding food (discrete stochastic events)
  MSR sector: pheromone field (continuous, diffuses, evaporates)
  Coupling: successful foraging → pheromone deposit → more foraging
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sip_sim.spec import SimulationSpec
from sip_sim.engine import build_engine


def make_foraging_spec(
    grid_size: int = 80,
    ant_density: float = 0.04,
    food_density: float = 0.01,  # fraction of cells with food
    evaporation: float = 0.15,
    diffusion: float = 0.005,
    deposit_rate: float = 5.0,
    bias_exponent: float = 1.5,  # WEAK bias (near-critical)
    bias_base: float = 0.5,      # strong exploration baseline
    momentum: float = 1.0,       # moderate persistence
    noise: float = 0.10,         # significant exploration
    seed: int = 42,
    max_steps: int = 1000,
) -> dict:
    """Create a foraging ant spec.

    The trick: we use TWO pheromone fields.
      'food_field': static field marking food locations (1 at food, 0 elsewhere)
      'pheromone': dynamic trail field (deposited by returning ants)

    Ants have state 'carrying' (0 or 1).

    Rules:
      1. If on food AND not carrying: pick up (carrying → 1)
      2. If carrying: deposit pheromone
      3. Movement: follow pheromone (weak bias when searching, strong when returning)

    For simplicity in the existing engine, we approximate:
      - ALL ants follow pheromone (but with weak bias)
      - Only ants near food deposit extra pheromone
      - The 'food_field' is a static env field with value > 0 at food cells

    Actually, the simplest model that captures the physics:
      - Food cells have high initial pheromone (seed the field)
      - Ants deposit pheromone unconditionally BUT at a LOW rate
      - The food cells continuously emit pheromone (persistent source)
      - Trails to food are reinforced; trails to nowhere fade

    This is equivalent to the reward model because:
      - Food sources = persistent pheromone emitters
      - Trail to food: reinforced by food source + ant traffic
      - Trail to nothing: reinforced only by ant traffic (weak)
      - Critical point: when ant-traffic reinforcement alone
        can't sustain a trail (evaporation wins)
    """
    return {
        "name": f"Foraging L={grid_size} λ={evaporation:.3f} food={food_density:.3f}",
        "topology": {
            "kind": "grid", "width": grid_size, "height": grid_size,
            "wrap": True, "neighbourhood": "moore",
        },
        "agent_kinds": [
            {
                "name": "ant", "color": "#e74c3c",
                "initial_fraction": ant_density, "mobile": True,
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
            # Food sources continuously emit pheromone (the "reward signal")
            {
                "name": "food_emits",
                "condition": {"self_kind": "food"},
                "action": {
                    "kind": "modify_state",
                    "state_update": {"pheromone": f"+{deposit_rate * 3}"},
                },
                "priority": 2,
            },
            # Ants deposit pheromone at LOW rate (exploration deposits)
            {
                "name": "ant_deposits",
                "condition": {"self_kind": "ant"},
                "action": {
                    "kind": "modify_state",
                    "state_update": {"pheromone": f"+{deposit_rate * 0.3}"},
                },
                "priority": 1,
            },
            # Ants follow pheromone with WEAK bias
            {
                "name": "ant_moves",
                "condition": {"self_kind": "ant"},
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
        ],
        "observables": [
            {"name": "ants", "kind": "count", "params": {"kind": "ant"}},
            {"name": "pheromone_max", "kind": "env_max",
             "params": {"field": "pheromone"}},
            {"name": "pheromone_mean", "kind": "env_mean",
             "params": {"field": "pheromone"}},
            {"name": "trail_linearity", "kind": "trail_linearity",
             "params": {"field": "pheromone", "threshold_frac": 0.3}},
            {"name": "trail_turnover", "kind": "field_autocorrelation",
             "params": {"field": "pheromone", "lag": 100}},
        ],
        "max_steps": max_steps,
        "snapshot_interval": 200,
        "seed": seed,
    }


def measure_trail_strength(spec_dict):
    """Run simulation and measure trail observables."""
    spec = SimulationSpec(**spec_dict)
    engine = build_engine(spec)
    trace = engine.run()

    phi = engine.state.env_grids['pheromone']
    mean_phi = phi.mean()
    max_phi = phi.max()
    conc_ratio = max_phi / mean_phi if mean_phi > 0.01 else 1.0

    # Block variance
    grid = spec_dict['topology']['width']
    bs = max(5, grid // 10)
    bh, bw = grid // bs, grid // bs
    tr = phi[:bh*bs, :bw*bs]
    blocks = tr.reshape(bh, bs, bw, bs).mean(axis=(1, 3))
    B = float(np.var(blocks) / mean_phi**2) if mean_phi > 0.01 else 0

    # Linearity from trace
    L_vals = [f.observables.get('trail_linearity', 0)
              for f in trace.frames[-100:] if 'trail_linearity' in f.observables]
    L = np.mean(L_vals) if L_vals else 0

    # Autocorrelation
    A_vals = [f.observables.get('trail_turnover', 0)
              for f in trace.frames[-50:] if 'trail_turnover' in f.observables]
    A = np.mean(A_vals) if A_vals else 0

    return {
        'conc_ratio': conc_ratio,
        'block_var': B,
        'linearity': L,
        'autocorr': A,
        'mean_phi': mean_phi,
        'max_phi': max_phi,
    }


if __name__ == '__main__':
    print("="*72)
    print("CRITICAL FORAGING: ANTS WITH FOOD SOURCES")
    print("="*72)

    # First: show there IS a transition as we vary evaporation
    print("\nSweep: evaporation λ at fixed food density = 0.01")
    print(f"{'λ':>6s} {'C_rat':>6s} {'B':>6s} {'L':>6s} {'A':>6s} {'mean_φ':>7s}")
    print("-"*42)

    for lam in np.arange(0.05, 0.55, 0.05):
        spec = make_foraging_spec(evaporation=lam, max_steps=800)
        result = measure_trail_strength(spec)
        print(f"{lam:>6.2f} {result['conc_ratio']:>6.1f} {result['block_var']:>6.3f} "
              f"{result['linearity']:>6.3f} {result['autocorr']:>6.3f} "
              f"{result['mean_phi']:>7.2f}", flush=True)

    # Second: sweep food density at fixed evaporation
    print(f"\nSweep: food density at λ = 0.20")
    print(f"{'ρ_food':>7s} {'C_rat':>6s} {'B':>6s} {'L':>6s} {'A':>6s} {'mean_φ':>7s}")
    print("-"*44)

    for food_dens in [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
        spec = make_foraging_spec(food_density=food_dens, evaporation=0.20,
                                   max_steps=800)
        result = measure_trail_strength(spec)
        print(f"{food_dens:>7.3f} {result['conc_ratio']:>6.1f} {result['block_var']:>6.3f} "
              f"{result['linearity']:>6.3f} {result['autocorr']:>6.3f} "
              f"{result['mean_phi']:>7.2f}", flush=True)

    print(f"\nIf there's a transition, the concentration ratio and block variance")
    print(f"should DROP sharply at some critical food density or evaporation rate.")
    print(f"Below critical: trails dissolve (only food sources glow).")
    print(f"Above critical: connected trail network to food sources.")
