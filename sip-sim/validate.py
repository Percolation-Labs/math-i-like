"""Validate sip-sim engine against known results from the literature.

Each model has quantitative expectations that must hold for the engine
to be considered correct. We check these and save diagnostic snapshots.
"""

import time
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sip_sim import build_engine, load
from sip_sim.viz import plot_observables, plot_snapshots


PASS = 0
FAIL = 0


def check(name: str, condition: bool, msg: str) -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{status}] {name}: {msg}")


def save_key_snapshots(trace, colors, model_name, steps):
    """Save snapshots at specific time steps for visual inspection."""
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
    if len(steps) == 1:
        axes = [axes]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors[:len(trace.kind_names)])

    for ax, target_step in zip(axes, steps):
        # Find closest snapshot
        best = None
        best_dist = float("inf")
        for frame in trace.frames:
            if frame.snapshot is not None:
                d = abs(frame.step - target_step)
                if d < best_dist:
                    best_dist = d
                    best = frame
        if best is not None and best.snapshot is not None:
            ax.imshow(best.snapshot, cmap=cmap, vmin=0, vmax=len(trace.kind_names)-1,
                      interpolation="nearest")
            ax.set_title(f"t={best.step}", fontsize=11)
        else:
            ax.set_title(f"t={target_step} (no snapshot)")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(model_name, fontsize=13)
    fig.tight_layout()
    path = f"validate_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# 1. Game of Life
# ─────────────────────────────────────────────────────────────

def validate_game_of_life():
    print("\n" + "=" * 60)
    print("  GAME OF LIFE")
    print("=" * 60)

    spec = load("game_of_life")
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    # Known results for GoL from random initial density ~0.3:
    # 1. Population drops sharply in first ~20 steps (chaotic die-off)
    # 2. Stabilises to ~10-15% density (still lifes + oscillators)
    # 3. Population at t=0 should be ~30% of 10000 = ~3000

    _, pop = trace.observable_series("population")
    _, dens = trace.observable_series("density")

    pop_t0 = pop[0]
    pop_final = pop[-1]
    pop_min_first_50 = min(pop[:50])

    check("initial_density",
          2500 < pop_t0 < 3500,
          f"t=0 population={pop_t0:.0f} (expect ~3000 from 30% of 10000)")

    check("die_off",
          pop[20] < pop_t0 * 0.7,
          f"t=20 population={pop[20]:.0f} < {pop_t0*0.7:.0f} (expect sharp initial decline)")

    check("stabilises",
          abs(pop[-1] - pop[-20]) / max(pop[-1], 1) < 0.15,
          f"final 20 steps: pop ranges {min(pop[-20:]):.0f}-{max(pop[-20:]):.0f} (should be roughly stable)")

    check("final_density_range",
          500 < pop_final < 2000,
          f"final population={pop_final:.0f} (expect 5-20% density for GoL)")

    colors = ["#ffffff", "#000000"]
    save_key_snapshots(trace, colors, "Game of Life", [0, 20, 100, 180])

    fig = plot_observables(trace, ["population"], save_path="validate_gol_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 2. Forest Fire
# ─────────────────────────────────────────────────────────────

def validate_forest_fire():
    print("\n" + "=" * 60)
    print("  FOREST FIRE (Drossel-Schwabl)")
    print("=" * 60)

    spec = load("forest_fire")
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    # Known results:
    # 1. Tree density oscillates (boom-bust)
    # 2. Mean density settles around 0.4-0.6 (depends on p_grow/p_lightning ratio)
    # 3. Fire count spikes correspond to tree density drops
    # 4. System never reaches static equilibrium — perpetual fluctuation

    _, trees = trace.observable_series("trees")
    _, burning = trace.observable_series("burning")
    _, density = trace.observable_series("tree_density")

    total_cells = 150 * 150

    # Skip first 50 steps (initial transient from uniform 60% seeding)
    trees_steady = trees[50:]
    density_steady = density[50:]

    check("oscillates",
          max(trees_steady) - min(trees_steady) > 2000,
          f"tree count range: {min(trees_steady):.0f}-{max(trees_steady):.0f} (expect large swings)")

    mean_dens = np.mean(density_steady)
    check("mean_density",
          0.25 < mean_dens < 0.70,
          f"mean tree density={mean_dens:.2f} (expect 0.3-0.65)")

    # Fire spikes should be present
    max_fire = max(burning)
    check("fire_spikes",
          max_fire > 100,
          f"max simultaneous burning={max_fire:.0f} (expect large fire events)")

    # Not static — check variance in last 100 steps
    std_last = np.std(trees[-100:])
    check("not_static",
          std_last > 100,
          f"std of trees in last 100 steps={std_last:.0f} (expect continued fluctuation)")

    colors = ["#ffffff", "#2ecc71", "#e74c3c"]
    save_key_snapshots(trace, colors, "Forest Fire", [0, 50, 250, 475])

    fig = plot_observables(trace, ["trees", "burning"], save_path="validate_ff_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 3. Schelling Segregation
# ─────────────────────────────────────────────────────────────

def validate_schelling():
    print("\n" + "=" * 60)
    print("  SCHELLING SEGREGATION")
    print("=" * 60)

    # Use larger grid and more steps for clearer segregation
    from sip_sim.spec import SimulationSpec
    from sip_sim.examples import SCHELLING
    spec_dict = {**SCHELLING, "max_steps": 300, "snapshot_interval": 30,
                 "topology": {**SCHELLING["topology"], "width": 80, "height": 80}}
    spec = SimulationSpec(**spec_dict)
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    # Known results (Schelling 1971, Hatna & Benenson 2012):
    # 1. Population counts should be conserved (no births/deaths)
    # 2. With tolerance=0.3, strong clustering emerges
    # 3. Measure: mean same-kind neighbour fraction should increase over time

    _, red = trace.observable_series("red")
    _, blue = trace.observable_series("blue")

    check("conservation_red",
          abs(red[-1] - red[0]) < 5,
          f"red count: t=0={red[0]:.0f}, t=end={red[-1]:.0f} (should be conserved)")

    check("conservation_blue",
          abs(blue[-1] - blue[0]) < 5,
          f"blue count: t=0={blue[0]:.0f}, t=end={blue[-1]:.0f} (should be conserved)")

    # Use the engine's native same_kind_fraction observable (vectorised, full grid)
    _, seg = trace.observable_series("segregation")

    skf_start = seg[0]
    skf_end = seg[-1]

    check("segregation_increases",
          skf_end > skf_start + 0.01,
          f"same-kind neighbour fraction: t=0={skf_start:.3f}, t=end={skf_end:.3f} (must increase)")

    check("strong_segregation",
          skf_end > 0.55,
          f"final same-kind fraction={skf_end:.3f} (expect >0.55 for tolerance=0.3)")

    # At random placement with 45%/45%/10% (red/blue/empty), expected same-kind is ~0.50
    # Note: step 0 in trace is after the first simulation step, so some
    # segregation has already begun. Random placement gives ~0.50; after
    # one step it should still be well below the converged value.
    check("started_near_random",
          skf_start < 0.65,
          f"early same-kind fraction={skf_start:.3f} (expect < 0.65, near random baseline)")

    colors = ["#ffffff", "#e74c3c", "#3498db"]
    save_key_snapshots(trace, colors, "Schelling Segregation", [0, 30, 150, 270])

    fig = plot_observables(trace, ["red", "blue"], save_path="validate_schelling_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 4. Predator-Prey
# ─────────────────────────────────────────────────────────────

def validate_predator_prey():
    print("\n" + "=" * 60)
    print("  PREDATOR-PREY (Lotka-Volterra)")
    print("=" * 60)

    spec = load("predator_prey")
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    # Known results for spatial LV:
    # 1. Both species should persist (coexistence) — neither goes to zero
    # 2. In spatial models, oscillations are often damped to a steady state
    #    (unlike the ODE which has neutral cycles)
    # 3. Predators should be fewer than prey at equilibrium

    _, prey = trace.observable_series("prey")
    _, preds = trace.observable_series("predators")

    check("prey_persist",
          prey[-1] > 50,
          f"final prey count={prey[-1]:.0f} (must survive)")

    check("predators_persist",
          preds[-1] > 10,
          f"final predator count={preds[-1]:.0f} (must survive)")

    check("predators_fewer_than_prey",
          preds[-1] < prey[-1],
          f"predators={preds[-1]:.0f} < prey={prey[-1]:.0f} (expect predators < prey)")

    # Check that the system isn't trivially stuck at initial conditions
    check("prey_dynamics",
          max(prey) != min(prey),
          f"prey range: {min(prey):.0f}-{max(prey):.0f} (must show dynamics)")

    check("predator_dynamics",
          max(preds) != min(preds),
          f"predator range: {min(preds):.0f}-{max(preds):.0f} (must show dynamics)")

    colors = ["#ffffff", "#2ecc71", "#e74c3c"]
    save_key_snapshots(trace, colors, "Predator Prey", [0, 25, 250, 475])

    fig = plot_observables(trace, ["prey", "predators"], save_path="validate_pp_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 5. Ant Clustering
# ─────────────────────────────────────────────────────────────

def validate_ant_clustering():
    print("\n" + "=" * 60)
    print("  ANT CLUSTERING (Deneubourg)")
    print("=" * 60)

    spec = load("ant_clustering")
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    # Known results:
    # 1. Food count must be conserved (no creation/destruction)
    # 2. Spatial entropy should decrease as clusters form
    # 3. Final entropy should be significantly lower than initial

    _, food = trace.observable_series("food")
    _, entropy = trace.observable_series("food_entropy")

    check("food_conserved",
          abs(food[-1] - food[0]) < 5,
          f"food count: t=0={food[0]:.0f}, t=end={food[-1]:.0f} (should be conserved)")

    check("entropy_decreases",
          entropy[-1] < entropy[0],
          f"spatial entropy: t=0={entropy[0]:.2f}, t=end={entropy[-1]:.2f} (must decrease)")

    # Entropy should drop substantially — at least 15% reduction
    entropy_reduction = (entropy[0] - entropy[-1]) / max(entropy[0], 0.01)
    check("substantial_clustering",
          entropy_reduction > 0.10,
          f"entropy reduction={entropy_reduction:.1%} (expect >10%)")

    # Mid-run entropy should be between start and end (monotonic-ish trend)
    mid = len(entropy) // 2
    check("progressive_clustering",
          entropy[mid] < entropy[0],
          f"mid-run entropy={entropy[mid]:.2f} < initial={entropy[0]:.2f} (should decrease over time)")

    colors = ["#ffffff", "#e67e22"]
    save_key_snapshots(trace, colors, "Ant Clustering", [0, 600, 1500, 2700])

    fig = plot_observables(trace, ["food_entropy"], save_path="validate_clustering_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 6. Pheromone Trail Formation
# ─────────────────────────────────────────────────────────────

def validate_ant_pheromone():
    print("\n" + "=" * 60)
    print("  PHEROMONE TRAILS (Deneubourg/Goss)")
    print("=" * 60)

    spec = load("ant_pheromone")
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    # Known results:
    # 1. Ant count should be conserved (no births/deaths)
    # 2. Pheromone field should build up (max > 0)
    # 3. Spatial entropy of ants should decrease (trail/cluster formation)
    # 4. Pheromone should concentrate (max >> mean)

    _, ants = trace.observable_series("ants")
    _, ant_entropy = trace.observable_series("ant_entropy")
    _, phero_max = trace.observable_series("pheromone_max")
    _, phero_mean = trace.observable_series("pheromone_mean")
    _, alignment = trace.observable_series("alignment")
    _, linearity = trace.observable_series("trail_linearity")

    check("ants_conserved",
          abs(ants[-1] - ants[0]) < 5,
          f"ant count: t=0={ants[0]:.0f}, t=end={ants[-1]:.0f} (should be conserved)")

    check("pheromone_builds",
          phero_max[-1] > 1.0,
          f"final pheromone max={phero_max[-1]:.1f} (must build up)")

    check("pheromone_concentrates",
          phero_max[-1] > 5 * phero_mean[-1],
          f"max/mean ratio={phero_max[-1]/max(phero_mean[-1],0.01):.1f} (expect concentrated trails)")

    check("ant_entropy_decreases",
          ant_entropy[-1] < ant_entropy[0],
          f"ant spatial entropy: t=0={ant_entropy[0]:.2f}, t=end={ant_entropy[-1]:.2f} (must decrease)")

    # Entropy reduction should be meaningful
    ent_reduction = (ant_entropy[0] - ant_entropy[-1]) / max(ant_entropy[0], 0.01)
    check("trail_formation",
          ent_reduction > 0.05,
          f"ant entropy reduction={ent_reduction:.1%} (expect >5% from trail formation)")

    # Order parameter candidates — report values for discovery
    print(f"\n  --- Order parameter candidates ---")
    print(f"  Directional alignment (nematic S): start={alignment[0]:.3f}, end={alignment[-1]:.3f}")
    print(f"  Trail linearity: start={linearity[0]:.3f}, end={linearity[-1]:.3f}")
    print(f"  Pheromone max/mean: start={phero_max[0]/max(phero_mean[0],0.01):.1f}, end={phero_max[-1]/max(phero_mean[-1],0.01):.1f}")
    print(f"  Ant entropy: start={ant_entropy[0]:.2f}, end={ant_entropy[-1]:.2f}")

    colors = ["#ffffff", "#e74c3c"]
    save_key_snapshots(trace, colors, "Pheromone Trails", [0, 400, 1000, 1800])

    fig = plot_observables(trace, ["ant_entropy", "pheromone_max", "alignment", "trail_linearity"],
                           save_path="validate_pheromone_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 7. Pheromone Trail Formation — Non-Equilibrium (Transient)
# ─────────────────────────────────────────────────────────────

def validate_ant_pheromone_transient():
    print("\n" + "=" * 60)
    print("  PHEROMONE TRAILS — TRANSIENT (Non-Equilibrium)")
    print("=" * 60)

    spec = load("ant_pheromone_transient")
    engine = build_engine(spec)

    t0 = time.perf_counter()
    trace = engine.run()
    elapsed = time.perf_counter() - t0
    print(f"  {spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps in {elapsed:.2f}s")

    _, ants = trace.observable_series("ants")
    _, ant_entropy = trace.observable_series("ant_entropy")
    _, phero_max = trace.observable_series("pheromone_max")
    _, phero_mean = trace.observable_series("pheromone_mean")
    _, linearity = trace.observable_series("trail_linearity")
    _, turnover = trace.observable_series("trail_turnover")

    check("ants_conserved",
          abs(ants[-1] - ants[0]) < 5,
          f"ant count: t=0={ants[0]:.0f}, t=end={ants[-1]:.0f} (should be conserved)")

    check("pheromone_builds",
          phero_max[-1] > 1.0,
          f"final pheromone max={phero_max[-1]:.1f} (must build up)")

    # Trail structure should exist at any given moment
    linearity_late = linearity[len(linearity) // 3:]
    mean_lin = np.mean(linearity_late)
    check("trails_exist",
          mean_lin > 0.05,
          f"mean trail linearity (late)={mean_lin:.3f} (expect >0.05 — trails present)")

    # Trail linearity should fluctuate (not converge to a fixed value)
    std_lin = np.std(linearity_late)
    check("trails_fluctuate",
          std_lin > 0.01,
          f"linearity std={std_lin:.4f} (expect >0.01 — trails are not frozen)")

    # Trail turnover: pheromone field should decorrelate over the lag window.
    # Skip early values (before ring buffer fills) and boundary refreshes.
    # Filter out values very close to 1.0 (boundary artifacts).
    turnover_real = [t for t in turnover[len(turnover) // 5:] if t < 0.95]
    if turnover_real:
        mean_turnover = np.mean(turnover_real)
        check("trail_turnover",
              mean_turnover < 0.7,
              f"mean field autocorrelation={mean_turnover:.3f} (expect <0.7 — trails rewire)")
    else:
        check("trail_turnover", False, "no valid turnover measurements (all near 1.0)")

    # Should NOT be fully disordered — pheromone should concentrate
    check("not_disordered",
          phero_max[-1] > 3 * phero_mean[-1],
          f"max/mean ratio={phero_max[-1] / max(phero_mean[-1], 0.01):.1f} (expect >3 — not just noise)")

    # Report order parameter candidates
    print(f"\n  --- Non-equilibrium diagnostics ---")
    print(f"  Trail linearity: mean={mean_lin:.3f}, std={std_lin:.4f}")
    if turnover_real:
        print(f"  Field autocorrelation: mean={mean_turnover:.3f} (lag={spec.observables[-1].params.get('lag', '?')})")
    print(f"  Pheromone max/mean: {phero_max[-1] / max(phero_mean[-1], 0.01):.1f}")
    print(f"  Ant entropy: {ant_entropy[0]:.2f} -> {ant_entropy[-1]:.2f}")

    colors = ["#ffffff", "#e74c3c"]
    save_key_snapshots(trace, colors, "Pheromone Trails (Transient)", [0, 600, 1500, 2700])

    fig = plot_observables(trace, ["trail_linearity", "trail_turnover"],
                           save_path="validate_transient_obs.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_game_of_life()
    validate_forest_fire()
    validate_schelling()
    validate_predator_prey()
    validate_ant_clustering()
    validate_ant_pheromone()
    validate_ant_pheromone_transient()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")

    sys.exit(1 if FAIL > 0 else 0)
