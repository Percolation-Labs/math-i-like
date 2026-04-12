"""Simulation engine — functional core with numpy-backed grid state.

Architecture:
  WorldState     = immutable-per-step snapshot (kind_grid + state_grids + env_grids)
  Neighbourhood  = pre-computed neighbour counts for one step
  StepContext    = shared config needed by effect functions

  Each simulation step is a four-phase pipeline:
    1. compute_neighbourhood()     — vectorised neighbour counts
    2. apply_state_effects()       — stackable modify_state rules (all fire, no claiming)
    3. apply_main_effects()        — exclusive rules (transform/die/move/spawn) with claiming
    4. apply_field_operations()    — decay/diffuse env fields

  State grids vs env grids:
    - state_grids: agent-bound fields — zeroed when an agent dies or moves away
    - env_grids:   environment fields — persist regardless of agent presence
                   (pheromone, food deposits, temperature, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sip_sim.spec import (
    Action,
    AgentKind,
    BehaviorStep,
    Condition,
    EnvField,
    FieldOperation,
    HeadingSpec,
    MoveWeight,
    NumberPredicate,
    ObservableSpec,
    Rule,
    SimulationSpec,
    Topology,
)

EMPTY = 0  # kind index 0 is always "empty"


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


@dataclass
class WorldState:
    """Complete grid state at one point in time."""

    kind_grid: np.ndarray  # (H, W) uint8  — agent kind per cell
    state_grids: dict[str, np.ndarray]  # agent-bound fields
    env_grids: dict[str, np.ndarray]  # persistent environment fields

    def copy(self) -> WorldState:
        return WorldState(
            self.kind_grid.copy(),
            {f: g.copy() for f, g in self.state_grids.items()},
            {f: g.copy() for f, g in self.env_grids.items()},
        )


@dataclass
class Neighbourhood:
    """Pre-computed neighbourhood information for one simulation step."""

    counts: dict[int, np.ndarray]  # kind_idx -> (H, W) neighbour count
    total: np.ndarray  # (H, W) total non-empty neighbours


@dataclass
class StepContext:
    """Immutable config passed to all effect functions."""

    kind_index: dict[str, int]
    offsets: list[tuple[int, int]]
    wrap: bool
    H: int
    W: int
    rng: np.random.Generator


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------


@dataclass
class TraceFrame:
    step: int
    observables: dict[str, float]
    snapshot: np.ndarray | None = None


@dataclass
class Trace:
    spec: SimulationSpec
    frames: list[TraceFrame] = field(default_factory=list)
    kind_names: list[str] = field(default_factory=list)

    def observable_series(self, name: str) -> tuple[list[int], list[float]]:
        steps, values = [], []
        for f in self.frames:
            if name in f.observables:
                steps.append(f.step)
                values.append(f.observables[name])
        return steps, values

    def snapshots(self) -> list[tuple[int, np.ndarray]]:
        return [(f.step, f.snapshot) for f in self.frames if f.snapshot is not None]


# ---------------------------------------------------------------------------
# Pure functions: neighbourhood
# ---------------------------------------------------------------------------


def build_offsets(neighbourhood: str, radius: int) -> list[tuple[int, int]]:
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if (dr, dc) == (0, 0):
                continue
            if neighbourhood == "von_neumann" and abs(dr) + abs(dc) > radius:
                continue
            offsets.append((dr, dc))
    return offsets


def compute_neighbourhood(
    kind_grid: np.ndarray, n_kinds: int, offsets: list[tuple[int, int]],
) -> Neighbourhood:
    counts: dict[int, np.ndarray] = {}
    for ki in range(n_kinds):
        mask = (kind_grid == ki).astype(np.int32)
        total = np.zeros_like(mask)
        for dr, dc in offsets:
            total += np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
        counts[ki] = total

    occupied = (kind_grid != EMPTY).astype(np.int32)
    total_occ = np.zeros_like(occupied)
    for dr, dc in offsets:
        total_occ += np.roll(np.roll(occupied, dr, axis=0), dc, axis=1)

    return Neighbourhood(counts=counts, total=total_occ)


# ---------------------------------------------------------------------------
# Pure functions: condition matching
# ---------------------------------------------------------------------------


def apply_predicate(mask: np.ndarray, grid: np.ndarray, pred: NumberPredicate) -> np.ndarray:
    if pred.eq is not None:
        mask = mask & (grid == pred.eq)
    if pred.lt is not None:
        mask = mask & (grid < pred.lt)
    if pred.lte is not None:
        mask = mask & (grid <= pred.lte)
    if pred.gt is not None:
        mask = mask & (grid > pred.gt)
    if pred.gte is not None:
        mask = mask & (grid >= pred.gte)
    return mask


def match_condition(
    cond: Condition,
    kind_grid: np.ndarray,
    state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    nbhd: Neighbourhood,
    kind_index: dict[str, int],
    signals: dict[str, float],
) -> np.ndarray:
    """Return (H, W) bool mask of cells where condition is satisfied."""
    H, W = kind_grid.shape
    mask = np.ones((H, W), dtype=bool)

    if cond.self_kind is not None:
        ki = kind_index.get(cond.self_kind, -1)
        mask &= kind_grid == ki

    if cond.self_state:
        for fname, preds in cond.self_state.items():
            if fname not in state_grids:
                return np.zeros((H, W), dtype=bool)
            mask = apply_predicate(mask, state_grids[fname], NumberPredicate(**preds))

    if cond.env_state:
        for fname, preds in cond.env_state.items():
            if fname not in env_grids:
                return np.zeros((H, W), dtype=bool)
            mask = apply_predicate(mask, env_grids[fname], NumberPredicate(**preds))

    if cond.neighbour_kind_count:
        for kind_name, preds in cond.neighbour_kind_count.items():
            ki = kind_index.get(kind_name, -1)
            if ki < 0 or ki not in nbhd.counts:
                return np.zeros((H, W), dtype=bool)
            mask = apply_predicate(mask, nbhd.counts[ki].astype(np.float64), NumberPredicate(**preds))

    if cond.neighbour_kind_fraction:
        for kind_name, preds in cond.neighbour_kind_fraction.items():
            ki = kind_index.get(kind_name, -1)
            if ki < 0 or ki not in nbhd.counts:
                return np.zeros((H, W), dtype=bool)
            nc = nbhd.counts[ki].astype(np.float64)
            tn = nbhd.total.astype(np.float64)
            frac = np.divide(nc, tn, out=np.zeros_like(nc), where=tn > 0)
            mask = apply_predicate(mask, frac, NumberPredicate(**preds))

    if cond.global_signal:
        for sig_name, preds in cond.global_signal.items():
            val = signals.get(sig_name, 0.0)
            if not NumberPredicate(**preds).matches(val):
                return np.zeros((H, W), dtype=bool)

    return mask


def apply_stochastic(mask: np.ndarray, rule: Rule, rng: np.random.Generator) -> np.ndarray:
    H, W = mask.shape
    if rule.rate < 1.0:
        mask = mask & (rng.random((H, W)) < rule.rate)
    if rule.condition.random_threshold is not None:
        mask = mask & (rng.random((H, W)) < rule.condition.random_threshold)
    return mask


# ---------------------------------------------------------------------------
# Pure functions: effects
# ---------------------------------------------------------------------------


def effect_transform(
    action: Action, mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    ctx: StepContext,
) -> None:
    new_ki = ctx.kind_index.get(action.new_kind or "", EMPTY)
    kind_grid[mask] = new_ki
    if action.state_update:
        _apply_updates(action.state_update, mask, state_grids, env_grids)


def effect_die(
    mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
) -> None:
    kind_grid[mask] = EMPTY
    for g in state_grids.values():
        g[mask] = 0.0
    # env_grids intentionally NOT touched — they persist


def effect_spawn(
    action: Action, mask: np.ndarray,
    kind_grid: np.ndarray,
    ctx: StepContext,
) -> None:
    target_ki = ctx.kind_index.get(
        action.new_kind or action.params.get("target_kind", ""), EMPTY
    )
    if target_ki == EMPTY:
        return
    candidates = np.zeros_like(kind_grid, dtype=bool)
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        for dr, dc in ctx.offsets:
            nr = (y + dr) % ctx.H if ctx.wrap else y + dr
            nc = (x + dc) % ctx.W if ctx.wrap else x + dc
            if 0 <= nr < ctx.H and 0 <= nc < ctx.W and kind_grid[nr, nc] == EMPTY:
                candidates[nr, nc] = True
    kind_grid[candidates] = target_ki


def _apply_updates(
    updates: dict[str, Any], mask: np.ndarray,
    state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
) -> None:
    """Apply state_update entries to whichever grid dict contains the field."""
    for fname, val in updates.items():
        target = state_grids if fname in state_grids else env_grids if fname in env_grids else None
        if target is None:
            continue
        sval = str(val)
        if sval.startswith("+"):
            target[fname][mask] += float(sval[1:])
        else:
            target[fname][mask] = float(sval)


def effect_modify_state(
    action: Action, mask: np.ndarray,
    state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
) -> None:
    if action.state_update:
        # In antisocial mode, only social ants deposit pheromone
        effective_mask = mask
        if "_social" in state_grids:
            social_grid = state_grids["_social"]
            effective_mask = mask & (social_grid > 0.5)
        _apply_updates(action.state_update, effective_mask, state_grids, env_grids)


def _move_agent(
    y: int, x: int, ny: int, nx: int,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
) -> None:
    """Move one agent from (y,x) to (ny,nx). Env grids are NOT touched."""
    kind_grid[ny, nx] = kind_grid[y, x]
    kind_grid[y, x] = EMPTY
    for g in state_grids.values():
        g[ny, nx] = g[y, x]
        g[y, x] = 0.0


def effect_move_local(
    mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    ctx: StepContext,
) -> None:
    ys, xs = np.where(mask)
    order = np.arange(len(ys))
    ctx.rng.shuffle(order)
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if kind_grid[y, x] == EMPTY:
            continue
        empties = []
        for dr, dc in ctx.offsets:
            nr = (y + dr) % ctx.H if ctx.wrap else y + dr
            nc = (x + dc) % ctx.W if ctx.wrap else x + dc
            if 0 <= nr < ctx.H and 0 <= nc < ctx.W and kind_grid[nr, nc] == EMPTY:
                empties.append((nr, nc))
        if empties:
            nr, nc = empties[ctx.rng.integers(len(empties))]
            _move_agent(y, x, nr, nc, kind_grid, state_grids)


def effect_move_global(
    mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    ctx: StepContext,
) -> None:
    ys, xs = np.where(mask)
    order = np.arange(len(ys))
    ctx.rng.shuffle(order)

    ey, ex = np.where(kind_grid == EMPTY)
    pool = list(zip(ey.tolist(), ex.tolist()))
    ctx.rng.shuffle(pool)
    pi = 0

    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if kind_grid[y, x] == EMPTY:
            continue
        while pi < len(pool) and kind_grid[pool[pi][0], pool[pi][1]] != EMPTY:
            pi += 1
        if pi >= len(pool):
            break
        ny, nx = pool[pi]
        _move_agent(y, x, ny, nx, kind_grid, state_grids)
        pi += 1
        pool.append((y, x))


def effect_move_biased(
    mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    ctx: StepContext,
    bias_field: str, bias_strength: float, bias_base: float,
    momentum: float = 0.0, heading_alpha: float = 0.3,
    noise: float = 0.0,
    p_follow: float | None = None,
    p_social: float | None = None,
    susceptibility: float | None = None,
) -> None:
    """Move agents toward cells with higher env field values, with optional momentum.

    Three modes:

    **Legacy mode** (when p_follow and p_social are None):
      w(j) = (bias_base + field[j])^bias_strength * momentum_factor(j)
      With probability `noise`, ignore pheromone and move randomly.

    **Simple mode** (when p_follow is set, 0-1):
      With probability p_follow: pick neighbour weighted linearly by pheromone.
      Otherwise: move randomly with momentum (directional persistence).
      Only 3 params matter: p_follow, evaporation rate, diffusion rate.
      bias_strength, bias_base, and noise are ignored in this mode.

    **Antisocial mode** (when p_social is set, 0-1):
      Like simple mode, but antisocial ants also stop depositing pheromone.
      With probability p_social: follow pheromone AND deposit (social).
      Otherwise: walk with momentum, NO deposit (antisocial).
      Sets _social state grid so deposit rule can check it.

    **Susceptibility mode** (when susceptibility is set):
      State-dependent following. All ants always deposit pheromone (they
      secrete it as they walk). But an ant only FOLLOWS pheromone when it
      detects concentration above the susceptibility threshold in its Moore
      neighbourhood. Below threshold, the ant walks with momentum, ignoring
      the social signal. The susceptibility parameter controls recruitment
      sensitivity — lower = more easily recruited into trail-following.
      Does NOT set _social grid (deposit is unconditional).
    """
    # Use capped pheromone field if saturation is active
    field = env_grids.get("_pheromone_capped") if "_pheromone_capped" in env_grids else env_grids.get(bias_field)
    if field is None:
        effect_move_local(mask, kind_grid, state_grids, ctx)
        return

    has_momentum = "dir_r" in state_grids and "dir_c" in state_grids
    dir_r = state_grids.get("dir_r")
    dir_c = state_grids.get("dir_c")

    ys, xs = np.where(mask)
    order = np.arange(len(ys))
    ctx.rng.shuffle(order)

    use_simple = p_follow is not None or p_social is not None or susceptibility is not None
    use_antisocial = p_social is not None
    use_susceptibility = susceptibility is not None
    social_grid = state_grids.get("_social") if use_antisocial else None
    follow_grid = state_grids.get("_follow") if use_susceptibility else None

    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if kind_grid[y, x] == EMPTY:
            continue

        # Current heading
        if has_momentum:
            hr, hc = dir_r[y, x], dir_c[y, x]
            h_mag = (hr * hr + hc * hc) ** 0.5
        else:
            hr, hc, h_mag = 0.0, 0.0, 0.0

        if use_susceptibility:
            # Susceptibility mode: use pre-computed follow flag from _step()
            following = follow_grid is not None and follow_grid[y, x] > 0.5
        elif use_antisocial:
            # Antisocial mode: use pre-computed social flag from _step()
            following = social_grid is not None and social_grid[y, x] > 0.5
        elif use_simple:
            # Simple mode: coin flip between follow-pheromone and random-with-momentum
            following = ctx.rng.random() < p_follow
        else:
            # Legacy mode: noise = probability of random exploration
            following = not (noise > 0 and ctx.rng.random() < noise)

        candidates = []
        weights = []

        for dr, dc in ctx.offsets:
            nr = (y + dr) % ctx.H if ctx.wrap else y + dr
            nc = (x + dc) % ctx.W if ctx.wrap else x + dc
            if 0 <= nr < ctx.H and 0 <= nc < ctx.W and kind_grid[nr, nc] == EMPTY:
                if use_simple:
                    if following:
                        # Linear pheromone weighting with momentum for directional persistence
                        # (ants walk ALONG trails, not just toward brightest spot)
                        w = field[nr, nc] + 0.01
                        if has_momentum and momentum > 0 and h_mag > 0.01:
                            d_mag = (dr * dr + dc * dc) ** 0.5
                            cos_a = (hr * dr + hc * dc) / (h_mag * d_mag) if d_mag > 0 else 0.0
                            w *= max(0.01, 1.0 + momentum * cos_a)
                    else:
                        # Random walk with momentum
                        w = 1.0
                        if has_momentum and h_mag > 0.01:
                            d_mag = (dr * dr + dc * dc) ** 0.5
                            cos_a = (hr * dr + hc * dc) / (h_mag * d_mag) if d_mag > 0 else 0.0
                            w = max(0.01, 1.0 + momentum * cos_a)
                else:
                    # Legacy mode
                    if not following:
                        w = 1.0  # uniform random (exploring)
                    else:
                        w = (bias_base + field[nr, nc]) ** bias_strength
                        if momentum > 0 and has_momentum and h_mag > 0.01:
                            d_mag = (dr * dr + dc * dc) ** 0.5
                            cos_a = (hr * dr + hc * dc) / (h_mag * d_mag) if d_mag > 0 else 0.0
                            w *= max(0.01, 1.0 + momentum * cos_a)

                candidates.append((nr, nc))
                weights.append(w)

        if candidates:
            wa = np.array(weights)
            wa /= wa.sum()
            choice = ctx.rng.choice(len(candidates), p=wa)
            nr, nc = candidates[choice]

            # Update heading before move (exponential moving average)
            if has_momentum:
                move_dr = nr - y
                move_dc = nc - x
                # Handle wrapping — use shortest path direction
                if ctx.wrap:
                    if abs(move_dr) > ctx.H // 2:
                        move_dr = -np.sign(move_dr) * (ctx.H - abs(move_dr))
                    if abs(move_dc) > ctx.W // 2:
                        move_dc = -np.sign(move_dc) * (ctx.W - abs(move_dc))
                dir_r[y, x] = (1 - heading_alpha) * hr + heading_alpha * move_dr
                dir_c[y, x] = (1 - heading_alpha) * hc + heading_alpha * move_dc

            _move_agent(y, x, nr, nc, kind_grid, state_grids)


# ---------------------------------------------------------------------------
# Declarative behavior pipeline
# ---------------------------------------------------------------------------


@dataclass
class AgentScratch:
    """Ephemeral per-cell values computed by pipeline steps within a single tick."""
    grids: dict[str, np.ndarray] = field(default_factory=dict)

    def get_or_create(self, name: str, H: int, W: int) -> np.ndarray:
        if name not in self.grids:
            self.grids[name] = np.zeros((H, W), dtype=np.float64)
        return self.grids[name]


def _pipe_sense(
    params: dict, mask: np.ndarray,
    env_grids: dict[str, np.ndarray],
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Read an env field in the neighbourhood, reduce to scalar per agent."""
    field_name = params["field"]
    method = params.get("method", "max_neighbour")
    output = params.get("output", field_name)
    fld = env_grids.get(field_name)
    if fld is None:
        return
    result = scratch.get_or_create(output, ctx.H, ctx.W)
    if method == "max_neighbour":
        max_nbr = np.copy(fld)
        for dr, dc in ctx.offsets:
            shifted = np.roll(np.roll(fld, -dr, axis=0), -dc, axis=1)
            np.maximum(max_nbr, shifted, out=max_nbr)
        result[mask] = max_nbr[mask]
    elif method == "sum_neighbour":
        total = np.zeros_like(fld)
        for dr, dc in ctx.offsets:
            total += np.roll(np.roll(fld, -dr, axis=0), -dc, axis=1)
        result[mask] = total[mask]
    elif method == "at_cell":
        result[mask] = fld[mask]


def _pipe_threshold(
    params: dict, mask: np.ndarray,
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Hard threshold: output=1 if input > value."""
    input_name = params.get("input", "pheromone")
    value = params["value"]
    output = params.get("output", "_follow")
    inp = scratch.grids.get(input_name)
    if inp is None:
        return
    result = scratch.get_or_create(output, ctx.H, ctx.W)
    result[mask] = (inp[mask] > value).astype(np.float64)


def _pipe_hill(
    params: dict, mask: np.ndarray,
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Soft sigmoid threshold: P(follow) = phi^n / (phi^n + s^n)."""
    input_name = params.get("input", "pheromone")
    s = params["half_max"]
    n = params["steepness"]
    output = params.get("output", "_follow")
    inp = scratch.grids.get(input_name)
    if inp is None:
        return
    result = scratch.get_or_create(output, ctx.H, ctx.W)
    phi = inp[mask]
    phi_n = np.power(phi, n)
    s_n = s ** n
    p_follow = phi_n / (phi_n + s_n)
    rolls = ctx.rng.random(size=p_follow.shape)
    result[mask] = (rolls < p_follow).astype(np.float64)


def _pipe_coin(
    params: dict, mask: np.ndarray,
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Simple probability gate: output=1 with probability p."""
    p = params["p"]
    output = params.get("output", "_follow")
    result = scratch.get_or_create(output, ctx.H, ctx.W)
    rolls = ctx.rng.random(size=(ctx.H, ctx.W))
    result[mask] = (rolls[mask] < p).astype(np.float64)


def _pipe_habituation(
    params: dict, mask: np.ndarray,
    state_grids: dict[str, np.ndarray],
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Temporal hysteresis on a boolean variable.

    Maintains persistent _hab and _hab_deaf counters in state_grids.
    Modifies the follow decision in scratch.
    """
    input_name = params.get("input", "_follow")
    tau_h = params["tau"]
    recovery_rate = params.get("recovery", 1.0)
    gate_deposit = params.get("gate_deposit", False)
    deposit_var = params.get("deposit_var", "_deposit")

    follow = scratch.grids.get(input_name)
    if follow is None:
        return

    ants = mask

    # Initialize persistent state if needed
    if "_hab" not in state_grids:
        hab_init = np.zeros((ctx.H, ctx.W), dtype=np.float64)
        hab_init[ants] = ctx.rng.uniform(0, tau_h, size=int(ants.sum()))
        state_grids["_hab"] = hab_init
    if "_hab_deaf" not in state_grids:
        deaf_init = np.zeros((ctx.H, ctx.W), dtype=np.float64)
        deaf_init[ants] = (state_grids["_hab"][ants] >= tau_h).astype(np.float64)
        state_grids["_hab_deaf"] = deaf_init

    hab = state_grids["_hab"]
    deaf = state_grids["_hab_deaf"]

    # Step 1: Apply deafness override
    is_deaf = deaf[ants] > 0.5
    follow_vals = follow[ants].copy()
    follow_vals[is_deaf] = 0.0
    follow[ants] = follow_vals

    # Step 2: Update hab counter
    actually_following = follow[ants] > 0.5
    hab_vals = hab[ants].copy()
    hab_vals[actually_following] += 1.0
    hab_vals[~actually_following] -= recovery_rate
    np.clip(hab_vals, 0.0, tau_h * 2, out=hab_vals)
    hab[ants] = hab_vals

    # Step 3: Update deafness state (hysteresis)
    deaf_vals = deaf[ants].copy()
    deaf_vals[hab[ants] >= tau_h] = 1.0
    deaf_vals[hab[ants] <= 0] = 0.0
    deaf[ants] = deaf_vals

    # Step 4: Gate deposition if requested
    if gate_deposit:
        dep = scratch.get_or_create(deposit_var, ctx.H, ctx.W)
        dep[:] = 1.0  # default: all deposit
        dep[deaf > 0.5] = 0.0  # deaf ants don't deposit


def _pipe_crowding(
    params: dict, mask: np.ndarray,
    kind_grid: np.ndarray,
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Override follow decision when too many agent neighbours."""
    max_nbrs = params["max_neighbours"]
    input_name = params.get("input", "_follow")
    output = params.get("output", "_follow")

    follow = scratch.grids.get(input_name)
    if follow is None:
        return

    ant_mask_float = (kind_grid != EMPTY).astype(np.float64)
    ant_count = np.zeros_like(kind_grid, dtype=np.float64)
    for dr, dc in ctx.offsets:
        ant_count += np.roll(np.roll(ant_mask_float, -dr, axis=0), -dc, axis=1)
    crowded = ant_count > max_nbrs

    if output != input_name:
        result = scratch.get_or_create(output, ctx.H, ctx.W)
        result[:] = follow
        result[mask & crowded] = 0.0
    else:
        follow[mask & crowded] = 0.0


def _pipe_saturation(
    params: dict, mask: np.ndarray,
    env_grids: dict[str, np.ndarray],
    scratch: AgentScratch, ctx: StepContext,
) -> None:
    """Peaked response: effective = field * exp(-field/peak)."""
    field_name = params["field"]
    peak = params["peak"]
    output_field = params.get("output_field", f"_effective_{field_name}")

    fld = env_grids.get(field_name)
    if fld is None:
        return
    effective = fld * np.exp(-fld / peak)
    # Store in env_grids so movement can use it as bias_field
    env_grids[output_field] = effective


def execute_pipeline(
    steps: list[BehaviorStep],
    mask: np.ndarray,
    kind_grid: np.ndarray,
    state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    scratch: AgentScratch,
    ctx: StepContext,
) -> None:
    """Execute a sequence of behavior steps, populating scratch variables."""
    for step in steps:
        match step.kind:
            case "sense":
                _pipe_sense(step.params, mask, env_grids, scratch, ctx)
            case "threshold":
                _pipe_threshold(step.params, mask, scratch, ctx)
            case "hill":
                _pipe_hill(step.params, mask, scratch, ctx)
            case "coin":
                _pipe_coin(step.params, mask, scratch, ctx)
            case "habituation":
                _pipe_habituation(step.params, mask, state_grids, scratch, ctx)
            case "crowding":
                _pipe_crowding(step.params, mask, kind_grid, scratch, ctx)
            case "saturation":
                _pipe_saturation(step.params, mask, env_grids, scratch, ctx)
            case "gate":
                pass  # gates are consumed by modify_state, not pipeline


def effect_move_weighted(
    mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    scratch: AgentScratch,
    ctx: StepContext,
    move_weights: list[MoveWeight],
    heading: HeadingSpec | None,
) -> None:
    """Move agents using declarative weight equation.

    P(agent moves to cell j) ∝ ∏_k term_k(j)

    Each MoveWeight contributes a multiplicative factor.  Terms gated
    by `requires` are only active when the named scratch variable
    is > 0.5 for the moving agent; otherwise they contribute 1.0.
    """
    # Pre-resolve heading state
    if heading is not None:
        dir_fields = heading.fields
        has_heading = all(f in state_grids for f in dir_fields)
    else:
        dir_fields = []
        has_heading = False
    if has_heading:
        dir_r_grid = state_grids[dir_fields[0]]
        dir_c_grid = state_grids[dir_fields[1]]

    # Pre-resolve env fields referenced by weight terms
    resolved_fields: dict[str, np.ndarray | None] = {}
    for wt in move_weights:
        if wt.kind == "field":
            fname = wt.params.get("field", "")
            if fname not in resolved_fields:
                resolved_fields[fname] = env_grids.get(fname)

    ys, xs = np.where(mask)
    order = np.arange(len(ys))
    ctx.rng.shuffle(order)

    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if kind_grid[y, x] == EMPTY:
            continue

        # Current heading
        if has_heading:
            hr, hc = float(dir_r_grid[y, x]), float(dir_c_grid[y, x])
            h_mag = (hr * hr + hc * hc) ** 0.5
        else:
            hr, hc, h_mag = 0.0, 0.0, 0.0

        # Pre-check which gated terms are active for this agent
        term_active = []
        for wt in move_weights:
            if wt.requires is not None:
                sg = scratch.grids.get(wt.requires)
                active = sg is not None and sg[y, x] > 0.5
            else:
                active = True
            term_active.append(active)

        candidates = []
        weights = []
        for dr, dc in ctx.offsets:
            nr = (y + dr) % ctx.H if ctx.wrap else y + dr
            nc = (x + dc) % ctx.W if ctx.wrap else x + dc
            if not (0 <= nr < ctx.H and 0 <= nc < ctx.W) or kind_grid[nr, nc] != EMPTY:
                continue

            w = 1.0
            for i, wt in enumerate(move_weights):
                if not term_active[i]:
                    continue
                match wt.kind:
                    case "field":
                        fld = resolved_fields.get(wt.params.get("field", ""))
                        if fld is not None:
                            floor = wt.params.get("floor", 0.01)
                            power = wt.params.get("power", 1.0)
                            val = floor + fld[nr, nc]
                            w *= val ** power if power != 1.0 else val
                    case "momentum":
                        strength = wt.params.get("strength", 1.0)
                        floor = wt.params.get("floor", 0.01)
                        if has_heading and h_mag > 0.01:
                            d_mag = (dr * dr + dc * dc) ** 0.5
                            cos_a = (hr * dr + hc * dc) / (h_mag * d_mag) if d_mag > 0 else 0.0
                            w *= max(floor, 1.0 + strength * cos_a)
                    case "avoid_kind":
                        target_kind = wt.params.get("kind", "")
                        threshold = wt.params.get("threshold", 4.0)
                        floor = wt.params.get("floor", 0.01)
                        ki = ctx.kind_index.get(target_kind, -1)
                        if ki >= 0:
                            count = 0
                            for dr2, dc2 in ctx.offsets:
                                cr = (nr + dr2) % ctx.H if ctx.wrap else nr + dr2
                                cc = (nc + dc2) % ctx.W if ctx.wrap else nc + dc2
                                if 0 <= cr < ctx.H and 0 <= cc < ctx.W and kind_grid[cr, cc] == ki:
                                    count += 1
                            w *= max(floor, 1.0 - count / threshold)

            candidates.append((nr, nc))
            weights.append(w)

        if candidates:
            wa = np.array(weights)
            wa /= wa.sum()
            choice = ctx.rng.choice(len(candidates), p=wa)
            nr, nc = candidates[choice]
            if has_heading and heading is not None:
                move_dr = nr - y
                move_dc = nc - x
                if ctx.wrap:
                    if abs(move_dr) > ctx.H // 2:
                        move_dr = -np.sign(move_dr) * (ctx.H - abs(move_dr))
                    if abs(move_dc) > ctx.W // 2:
                        move_dc = -np.sign(move_dc) * (ctx.W - abs(move_dc))
                a = heading.alpha
                dir_r_grid[y, x] = (1 - a) * hr + a * move_dr
                dir_c_grid[y, x] = (1 - a) * hc + a * move_dc
            _move_agent(y, x, nr, nc, kind_grid, state_grids)


def effect_move_pipeline(
    mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    scratch: AgentScratch,
    ctx: StepContext,
    params: dict[str, Any],
) -> None:
    """Legacy pipeline movement — used when move_weights is not set."""
    bias_field_name = params.get("bias_field")
    momentum = params.get("momentum", 0.0)
    heading_alpha = params.get("heading_alpha", 0.3)
    follow_var = params.get("follow_var", "_follow")
    effective_field_name = params.get("effective_field")

    if effective_field_name and effective_field_name in env_grids:
        field = env_grids[effective_field_name]
    elif bias_field_name:
        field = env_grids.get(bias_field_name)
    else:
        field = None

    if field is None:
        effect_move_local(mask, kind_grid, state_grids, ctx)
        return

    follow_grid = scratch.grids.get(follow_var)
    has_momentum = "dir_r" in state_grids and "dir_c" in state_grids
    dir_r = state_grids.get("dir_r")
    dir_c = state_grids.get("dir_c")

    ys, xs = np.where(mask)
    order = np.arange(len(ys))
    ctx.rng.shuffle(order)

    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if kind_grid[y, x] == EMPTY:
            continue

        if has_momentum:
            hr, hc = dir_r[y, x], dir_c[y, x]
            h_mag = (hr * hr + hc * hc) ** 0.5
        else:
            hr, hc, h_mag = 0.0, 0.0, 0.0

        following = follow_grid is not None and follow_grid[y, x] > 0.5

        candidates = []
        weights = []
        for dr, dc in ctx.offsets:
            nr = (y + dr) % ctx.H if ctx.wrap else y + dr
            nc = (x + dc) % ctx.W if ctx.wrap else x + dc
            if 0 <= nr < ctx.H and 0 <= nc < ctx.W and kind_grid[nr, nc] == EMPTY:
                if following:
                    w = field[nr, nc] + 0.01
                    if has_momentum and momentum > 0 and h_mag > 0.01:
                        d_mag = (dr * dr + dc * dc) ** 0.5
                        cos_a = (hr * dr + hc * dc) / (h_mag * d_mag) if d_mag > 0 else 0.0
                        w *= max(0.01, 1.0 + momentum * cos_a)
                else:
                    w = 1.0
                    if has_momentum and h_mag > 0.01:
                        d_mag = (dr * dr + dc * dc) ** 0.5
                        cos_a = (hr * dr + hc * dc) / (h_mag * d_mag) if d_mag > 0 else 0.0
                        w = max(0.01, 1.0 + momentum * cos_a)
                candidates.append((nr, nc))
                weights.append(w)

        if candidates:
            wa = np.array(weights)
            wa /= wa.sum()
            choice = ctx.rng.choice(len(candidates), p=wa)
            nr, nc = candidates[choice]
            if has_momentum:
                move_dr = nr - y
                move_dc = nc - x
                if ctx.wrap:
                    if abs(move_dr) > ctx.H // 2:
                        move_dr = -np.sign(move_dr) * (ctx.H - abs(move_dr))
                    if abs(move_dc) > ctx.W // 2:
                        move_dc = -np.sign(move_dc) * (ctx.W - abs(move_dc))
                dir_r[y, x] = (1 - heading_alpha) * hr + heading_alpha * move_dr
                dir_c[y, x] = (1 - heading_alpha) * hc + heading_alpha * move_dc
            _move_agent(y, x, nr, nc, kind_grid, state_grids)


def apply_effect(
    rule: Rule, mask: np.ndarray,
    kind_grid: np.ndarray, state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    ctx: StepContext,
    scratch: AgentScratch | None = None,
) -> None:
    action = rule.action
    match action.kind:
        case "transform":
            effect_transform(action, mask, kind_grid, state_grids, env_grids, ctx)
        case "die":
            effect_die(mask, kind_grid, state_grids)
        case "spawn":
            effect_spawn(action, mask, kind_grid, ctx)
        case "modify_state":
            # Check deposit_gate (new) or legacy _social grid
            if action.deposit_gate and scratch is not None:
                gate = scratch.grids.get(action.deposit_gate)
                if gate is not None:
                    mask = mask & (gate > 0.5)
            effect_modify_state(action, mask, state_grids, env_grids)
        case "move":
            if action.move_weights is not None and scratch is not None:
                # Declarative equation of motion
                effect_move_weighted(
                    mask, kind_grid, state_grids, env_grids,
                    scratch, ctx, action.move_weights, action.heading,
                )
            elif action.pipeline is not None and scratch is not None:
                # Pipeline mode without declarative weights (legacy pipeline)
                effect_move_pipeline(
                    mask, kind_grid, state_grids, env_grids,
                    scratch, ctx, action.params,
                )
            else:
                # Legacy mode: params-based dispatch
                scope = action.params.get("scope", "local")
                bias_field = action.params.get("bias_field")
                if bias_field:
                    effect_move_biased(
                        mask, kind_grid, state_grids, env_grids, ctx,
                        bias_field=bias_field,
                        bias_strength=action.params.get("bias_strength", 2.0),
                        bias_base=action.params.get("bias_base", 1.0),
                        momentum=action.params.get("momentum", 0.0),
                        heading_alpha=action.params.get("heading_alpha", 0.3),
                        noise=action.params.get("noise", 0.0),
                        p_follow=action.params.get("p_follow"),
                        p_social=action.params.get("p_social"),
                        susceptibility=action.params.get("susceptibility"),
                    )
                elif scope == "global":
                    effect_move_global(mask, kind_grid, state_grids, ctx)
                else:
                    effect_move_local(mask, kind_grid, state_grids, ctx)
        case "noop":
            pass


# ---------------------------------------------------------------------------
# Field operations (decay, diffuse)
# ---------------------------------------------------------------------------


def apply_field_operations(
    env_grids: dict[str, np.ndarray],
    operations: list[FieldOperation],
    offsets: list[tuple[int, int]],
) -> None:
    """Apply global field operations (evaporation, diffusion) to env grids."""
    for op in operations:
        if op.field not in env_grids:
            continue
        g = env_grids[op.field]
        if op.kind == "decay":
            g *= (1.0 - op.rate)
        elif op.kind == "diffuse":
            n = len(offsets)
            laplacian = -n * g.copy()
            for dr, dc in offsets:
                laplacian += np.roll(np.roll(g, dr, axis=0), dc, axis=1)
            g += op.rate * laplacian
            np.clip(g, 0.0, None, out=g)


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def compute_observables(
    kind_grid: np.ndarray,
    state_grids: dict[str, np.ndarray],
    env_grids: dict[str, np.ndarray],
    kind_index: dict[str, int],
    kind_names: list[str],
    specs: list[ObservableSpec],
    nbhd: Neighbourhood | None = None,
) -> dict[str, float]:
    obs: dict[str, float] = {}
    H, W = kind_grid.shape

    for o in specs:
        match o.kind:
            case "count":
                ki = kind_index.get(o.params.get("kind", ""), -1)
                obs[o.name] = int(np.sum(kind_grid == ki))

            case "density":
                ki = kind_index.get(o.params.get("kind", ""), -1)
                obs[o.name] = float(np.mean(kind_grid == ki))

            case "entropy":
                counts = np.bincount(kind_grid.ravel(), minlength=len(kind_names))[1:]
                total = counts.sum()
                if total > 0:
                    probs = counts[counts > 0] / total
                    obs[o.name] = float(-np.sum(probs * np.log2(probs)))
                else:
                    obs[o.name] = 0.0

            case "aggregate_state":
                fname = o.params.get("field", "")
                agg = o.params.get("agg", "mean")
                kind_name = o.params.get("kind")
                # Check both state and env grids
                grid = state_grids.get(fname, env_grids.get(fname))
                if grid is not None:
                    if kind_name:
                        vals = grid[kind_grid == kind_index.get(kind_name, -1)]
                    else:
                        vals = grid[kind_grid != EMPTY]
                    if len(vals) > 0:
                        fn = {"mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min}
                        obs[o.name] = float(fn.get(agg, np.mean)(vals))
                    else:
                        obs[o.name] = 0.0

            case "same_kind_fraction":
                if nbhd is None:
                    obs[o.name] = 0.0
                    continue
                occupied = kind_grid != EMPTY
                same = np.zeros((H, W), dtype=np.float64)
                for ki in range(1, len(kind_names)):
                    ki_mask = kind_grid == ki
                    same[ki_mask] = nbhd.counts[ki][ki_mask].astype(np.float64)
                tn = nbhd.total.astype(np.float64)
                frac = np.divide(same, tn, out=np.zeros_like(same), where=(tn > 0) & occupied)
                vals = frac[occupied]
                obs[o.name] = float(np.mean(vals)) if len(vals) > 0 else 0.0

            case "spatial_entropy":
                # Shannon entropy of agent distribution over grid blocks.
                # High = uniform spread, low = clustered.
                ki = kind_index.get(o.params.get("kind", ""), -1)
                block_size = o.params.get("block_size", 10)
                present = (kind_grid == ki).astype(np.float64)
                bh, bw = H // block_size, W // block_size
                if bh == 0 or bw == 0:
                    obs[o.name] = 0.0
                    continue
                # Reshape into blocks and sum
                trimmed = present[:bh * block_size, :bw * block_size]
                blocks = trimmed.reshape(bh, block_size, bw, block_size).sum(axis=(1, 3))
                block_counts = blocks.ravel()
                total = block_counts.sum()
                if total > 0:
                    probs = block_counts[block_counts > 0] / total
                    obs[o.name] = float(-np.sum(probs * np.log2(probs)))
                else:
                    obs[o.name] = 0.0

            case "env_max":
                fname = o.params.get("field", "")
                if fname in env_grids:
                    obs[o.name] = float(np.max(env_grids[fname]))
                else:
                    obs[o.name] = 0.0

            case "env_mean":
                fname = o.params.get("field", "")
                if fname in env_grids:
                    obs[o.name] = float(np.mean(env_grids[fname]))
                else:
                    obs[o.name] = 0.0

            case "directional_alignment":
                # Nematic order parameter S = <cos(2θ)> over agents with headings.
                # S=0 for random directions, S→1 for global alignment.
                # Uses dir_r, dir_c state grids as heading vectors.
                ki = kind_index.get(o.params.get("kind", ""), -1)
                dr_grid = state_grids.get("dir_r")
                dc_grid = state_grids.get("dir_c")
                if dr_grid is not None and dc_grid is not None and ki >= 0:
                    agent_mask = kind_grid == ki
                    dr_vals = dr_grid[agent_mask]
                    dc_vals = dc_grid[agent_mask]
                    mag = np.sqrt(dr_vals ** 2 + dc_vals ** 2)
                    moving = mag > 0.01
                    if moving.sum() > 1:
                        # Angles of moving agents
                        theta = np.arctan2(dr_vals[moving], dc_vals[moving])
                        # Nematic: cos(2θ) and sin(2θ) to get alignment
                        # regardless of which "direction" along the axis
                        c2 = np.mean(np.cos(2 * theta))
                        s2 = np.mean(np.sin(2 * theta))
                        obs[o.name] = float(np.sqrt(c2 ** 2 + s2 ** 2))
                    else:
                        obs[o.name] = 0.0
                else:
                    obs[o.name] = 0.0

            case "trail_linearity":
                # Fraction of "hot" pheromone cells that look like trail segments
                # (exactly 2 hot neighbours) vs blob centres (3+ hot neighbours).
                # High = linear trails, low = blobs/clusters.
                fname = o.params.get("field", "")
                threshold_frac = o.params.get("threshold_frac", 0.5)
                if fname in env_grids:
                    field = env_grids[fname]
                    fmax = float(np.max(field))
                    if fmax > 0.01:
                        threshold = fmax * threshold_frac
                        hot = (field > threshold).astype(np.int32)
                        n_hot = int(hot.sum())
                        if n_hot > 2:
                            # Count hot neighbours for each hot cell
                            hot_nbrs = np.zeros_like(hot)
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                hot_nbrs += np.roll(np.roll(hot, dr, axis=0), dc, axis=1)
                            # Trail cells: hot with exactly 2 hot neighbours
                            trail_cells = int(np.sum((hot == 1) & (hot_nbrs == 2)))
                            obs[o.name] = float(trail_cells / n_hot)
                        else:
                            obs[o.name] = 0.0
                    else:
                        obs[o.name] = 0.0
                else:
                    obs[o.name] = 0.0

            case "block_variance":
                # Variance of block-averaged field values, normalised by mean².
                # Measures spatial heterogeneity at a coarse scale.
                # High = pheromone concentrated in some regions (trail networks).
                # Low = pheromone spread uniformly (random deposition).
                # Unlike trail_linearity, this metric reliably distinguishes
                # collective trail structure from individual-momentum artifacts
                # because random deposition is uniform at the block scale even
                # when it creates cell-level texture.
                fname = o.params.get("field", "")
                block_size = int(o.params.get("block_size", 8))
                if fname in env_grids:
                    field = env_grids[fname]
                    fH, fW = field.shape
                    bh, bw = fH // block_size, fW // block_size
                    overall_mean = float(field.mean())
                    if bh > 0 and bw > 0 and overall_mean > 1e-10:
                        trimmed = field[:bh * block_size, :bw * block_size]
                        blocks = trimmed.reshape(bh, block_size, bw, block_size).mean(axis=(1, 3))
                        obs[o.name] = float(np.var(blocks)) / (overall_mean ** 2)
                    else:
                        obs[o.name] = 0.0
                else:
                    obs[o.name] = 0.0

            case "field_autocorrelation":
                # Pearson correlation between current field and a stored
                # previous snapshot (lag steps ago). Measures trail turnover.
                # 1.0 = frozen trails, 0.0 = fully decorrelated, between = transient.
                # The engine stores _prev_env snapshots — see GridEngine._step().
                fname = o.params.get("field", "")
                prev_key = f"_prev_{fname}"
                if fname in env_grids and prev_key in env_grids:
                    curr = env_grids[fname].ravel()
                    prev = env_grids[prev_key].ravel()
                    cm, pm = curr.mean(), prev.mean()
                    cd, pd = curr - cm, prev - pm
                    denom = (np.sqrt(np.sum(cd ** 2)) * np.sqrt(np.sum(pd ** 2)))
                    if denom > 1e-10:
                        obs[o.name] = float(np.sum(cd * pd) / denom)
                    else:
                        obs[o.name] = 1.0
                else:
                    obs[o.name] = 1.0

            case _:
                obs[o.name] = 0.0

    return obs


# ---------------------------------------------------------------------------
# Grid Engine
# ---------------------------------------------------------------------------


class GridEngine:
    """Simulation engine for regular 2D grids.

    Step pipeline:
      neighbourhood → state effects → main effects → field ops → observe
    """

    def __init__(self, spec: SimulationSpec):
        self.spec = spec
        topo = spec.topology
        assert topo.kind == "grid", "GridEngine only handles grid topologies"

        self.H = topo.height or 50
        self.W = topo.width or 50
        self.rng = np.random.default_rng(spec.seed)

        # Kind registry
        self.kind_names: list[str] = ["__empty__"]
        self.kind_index: dict[str, int] = {"__empty__": EMPTY}
        self.kind_specs: dict[int, AgentKind] = {}
        for i, ak in enumerate(spec.agent_kinds, start=1):
            self.kind_names.append(ak.name)
            self.kind_index[ak.name] = i
            self.kind_specs[i] = ak

        # Agent state fields
        seen: set[str] = set()
        self.state_fields: list[str] = []
        for ak in spec.agent_kinds:
            for fname in ak.state_schema:
                if fname not in seen:
                    self.state_fields.append(fname)
                    seen.add(fname)

        # Neighbourhood offsets
        self._offsets = build_offsets(topo.neighbourhood, topo.neighbourhood_radius)

        # Step context
        self._ctx = StepContext(
            kind_index=self.kind_index,
            offsets=self._offsets,
            wrap=topo.wrap,
            H=self.H, W=self.W,
            rng=self.rng,
        )

        # Signals
        self.signals: dict[str, float] = {
            gs.name: gs.initial_value for gs in spec.global_signals
        }

        # Pre-split rules
        self._state_rules = sorted(
            [r for r in spec.rules if r.action.kind == "modify_state"],
            key=lambda r: -r.priority,
        )
        self._main_rules = sorted(
            [r for r in spec.rules if r.action.kind != "modify_state"],
            key=lambda r: -r.priority,
        )

        # Check if any rule uses declarative pipelines
        self._has_pipelines = any(
            r.action.pipeline is not None for r in spec.rules
        )
        # Collect pipeline rules for Phase 0.5
        self._pipeline_rules = [
            r for r in spec.rules if r.action.pipeline is not None
        ]

        # Legacy: check if any move rule uses p_social, susceptibility, or habituation
        self._p_social = None
        self._susceptibility = None
        self._susceptibility_field = None
        self._habituation = None       # τ_h: steps before ant habituates
        self._hab_recovery = 1.0       # recovery rate (>1 = faster recovery)
        self._crowding_threshold = None # max ant neighbours before avoiding
        self._pheromone_saturation = None  # cap effective pheromone for movement
        self._hill_steepness = None        # Hill function steepness for soft threshold
        self._coupled_deposition = False   # if True, non-following ants don't deposit
        for r in self._main_rules:
            params = r.action.params or {}
            if params.get("p_social") is not None:
                self._p_social = params["p_social"]
            if params.get("susceptibility") is not None:
                self._susceptibility = params["susceptibility"]
                self._susceptibility_field = params.get("bias_field", "pheromone")
            if params.get("coupled_deposition"):
                self._coupled_deposition = True
            if params.get("habituation") is not None:
                self._habituation = params["habituation"]
                self._hab_recovery = params.get("hab_recovery", 1.0)
            if params.get("crowding_threshold") is not None:
                self._crowding_threshold = params["crowding_threshold"]
            if params.get("pheromone_saturation") is not None:
                self._pheromone_saturation = params["pheromone_saturation"]
            if params.get("hill_steepness") is not None:
                self._hill_steepness = params["hill_steepness"]
            if self._p_social is not None or self._susceptibility is not None:
                break

        # Initialise world state
        self.state = self._init_world()

    def _init_world(self) -> WorldState:
        kg = np.zeros((self.H, self.W), dtype=np.uint8)
        sg = {f: np.zeros((self.H, self.W), dtype=np.float64) for f in self.state_fields}
        eg = {ef.name: np.full((self.H, self.W), ef.initial_value, dtype=np.float64)
              for ef in self.spec.env_fields}

        total_cells = self.H * self.W
        all_positions = list(np.ndindex(self.H, self.W))
        self.rng.shuffle(all_positions)
        pos_idx = 0

        for kind_idx, ak in self.kind_specs.items():
            if ak.initial_count is not None:
                count = ak.initial_count
            elif ak.initial_fraction is not None:
                count = int(total_cells * ak.initial_fraction)
            else:
                continue

            if ak.initial_placement == "left":
                placed = 0
                for r in range(self.H):
                    for c in range(self.W // 3):
                        if placed >= count:
                            break
                        if kg[r, c] == EMPTY:
                            kg[r, c] = kind_idx
                            placed += 1
                continue
            elif ak.initial_placement == "right":
                placed = 0
                for r in range(self.H):
                    for c in range(self.W - 1, 2 * self.W // 3 - 1, -1):
                        if placed >= count:
                            break
                        if kg[r, c] == EMPTY:
                            kg[r, c] = kind_idx
                            placed += 1
                continue

            for _ in range(count):
                if pos_idx >= len(all_positions):
                    break
                r, c = all_positions[pos_idx]
                kg[r, c] = kind_idx
                for fname, fval in ak.initial_state.items():
                    if fname in sg:
                        sg[fname][r, c] = float(fval)
                # Random initial heading for mobile agents with direction state
                if ak.mobile and "dir_r" in sg and "dir_r" in ak.state_schema:
                    angle = self.rng.uniform(0, 2 * np.pi)
                    sg["dir_r"][r, c] = np.sin(angle)
                    sg["dir_c"][r, c] = np.cos(angle)
                pos_idx += 1

        return WorldState(kg, sg, eg)

    def _step(self) -> tuple[WorldState, dict[str, float], Neighbourhood]:
        old = self.state
        nbhd = compute_neighbourhood(old.kind_grid, len(self.kind_names), self._offsets)
        new = old.copy()

        # Phase 0.5: declarative behavior pipelines (replaces legacy Phase 0)
        scratch: AgentScratch | None = None
        if self._has_pipelines:
            scratch = AgentScratch()
            for rule in self._pipeline_rules:
                mask = match_condition(
                    rule.condition, old.kind_grid, old.state_grids, old.env_grids,
                    nbhd, self.kind_index, self.signals,
                )
                mask = apply_stochastic(mask, rule, self.rng)
                if mask.any():
                    execute_pipeline(
                        rule.action.pipeline, mask,
                        new.kind_grid, new.state_grids, new.env_grids,
                        scratch, self._ctx,
                    )

        # Phase 0 (legacy): pre-compute flags for antisocial/susceptibility mode
        # Skipped when declarative pipelines are active.
        if not self._has_pipelines and self._p_social is not None:
            # Antisocial mode: _social controls both deposit and following
            if "_social" not in new.state_grids:
                new.state_grids["_social"] = np.zeros_like(new.kind_grid, dtype=np.float64)
            social = new.state_grids["_social"]
            social[:] = 0.0
            ants = new.kind_grid != EMPTY
            rolls = self.rng.random(size=social.shape)
            social[ants] = (rolls[ants] < self._p_social).astype(np.float64)

        if not self._has_pipelines and self._susceptibility is not None:
            # Susceptibility mode: _follow controls movement.
            # If coupled_deposition, _social mirrors _follow (non-followers don't deposit).
            if "_follow" not in new.state_grids:
                new.state_grids["_follow"] = np.zeros_like(new.kind_grid, dtype=np.float64)
            follow = new.state_grids["_follow"]
            follow[:] = 0.0
            ants = new.kind_grid != EMPTY
            field = new.env_grids.get(self._susceptibility_field)
            if field is not None:
                # Max pheromone in Moore neighbourhood (ant's sensing range)
                max_nbr = np.copy(field)
                for dr, dc in self._offsets:
                    shifted = np.roll(np.roll(field, -dr, axis=0), -dc, axis=1)
                    np.maximum(max_nbr, shifted, out=max_nbr)
                if self._hill_steepness is not None:
                    # Soft sigmoid: P(follow) = φ^n / (φ^n + s^n)
                    n = self._hill_steepness
                    s = self._susceptibility
                    phi = max_nbr[ants]
                    phi_n = np.power(phi, n)
                    s_n = s ** n
                    p_follow = phi_n / (phi_n + s_n)
                    rolls = self.rng.random(size=p_follow.shape)
                    follow[ants] = (rolls < p_follow).astype(np.float64)
                else:
                    follow[ants] = (max_nbr[ants] > self._susceptibility).astype(np.float64)

            # Habituation: override follow for habituated ants
            if self._habituation is not None:
                tau_h = self._habituation
                recovery_rate = self._hab_recovery
                if "_hab" not in new.state_grids:
                    hab_init = np.zeros_like(new.kind_grid, dtype=np.float64)
                    hab_init[ants] = self.rng.uniform(0, tau_h, size=int(ants.sum()))
                    new.state_grids["_hab"] = hab_init
                if "_hab_deaf" not in new.state_grids:
                    deaf_init = np.zeros_like(new.kind_grid, dtype=np.float64)
                    deaf_init[ants] = (new.state_grids["_hab"][ants] >= tau_h).astype(np.float64)
                    new.state_grids["_hab_deaf"] = deaf_init
                hab = new.state_grids["_hab"]
                deaf = new.state_grids["_hab_deaf"]

                is_deaf = deaf[ants] > 0.5
                follow_vals = follow[ants].copy()
                follow_vals[is_deaf] = 0.0
                follow[ants] = follow_vals

                actually_following = follow[ants] > 0.5
                hab_vals = hab[ants].copy()
                hab_vals[actually_following] += 1.0
                hab_vals[~actually_following] -= recovery_rate
                np.clip(hab_vals, 0.0, tau_h * 2, out=hab_vals)
                hab[ants] = hab_vals

                deaf_vals = deaf[ants].copy()
                deaf_vals[hab[ants] >= tau_h] = 1.0
                deaf_vals[hab[ants] <= 0] = 0.0
                deaf[ants] = deaf_vals

                if "_social" not in new.state_grids:
                    new.state_grids["_social"] = np.ones_like(new.kind_grid, dtype=np.float64)
                social = new.state_grids["_social"]
                social[:] = 1.0
                social[deaf > 0.5] = 0.0

            # Crowding avoidance
            if self._crowding_threshold is not None:
                ant_count = np.zeros_like(new.kind_grid, dtype=np.float64)
                ant_mask_float = (new.kind_grid != EMPTY).astype(np.float64)
                for dr, dc in self._offsets:
                    ant_count += np.roll(np.roll(ant_mask_float, -dr, axis=0), -dc, axis=1)
                crowded = ant_count > self._crowding_threshold
                follow[ants & crowded] = 0.0

            # Coupled deposition
            if self._coupled_deposition:
                if "_social" not in new.state_grids:
                    new.state_grids["_social"] = np.ones_like(new.kind_grid, dtype=np.float64)
                social = new.state_grids["_social"]
                social[:] = 0.0
                social[follow > 0.5] = 1.0

        if not self._has_pipelines and self._pheromone_saturation is not None:
            sat = self._pheromone_saturation
            field = new.env_grids.get(self._susceptibility_field or "pheromone")
            if field is not None:
                effective = field * np.exp(-field / sat)
                new.env_grids["_pheromone_capped"] = effective

        # Phase 1: stackable effects (modify_state)
        for rule in self._state_rules:
            mask = match_condition(
                rule.condition, old.kind_grid, old.state_grids, old.env_grids,
                nbhd, self.kind_index, self.signals,
            )
            mask = apply_stochastic(mask, rule, self.rng)
            if mask.any():
                # Pipeline deposit_gate: restrict deposition to agents where gate > 0.5
                if rule.action.deposit_gate and scratch is not None:
                    gate = scratch.grids.get(rule.action.deposit_gate)
                    if gate is not None:
                        mask = mask & (gate > 0.5)
                effect_modify_state(rule.action, mask, new.state_grids, new.env_grids)

        # Phase 2: exclusive effects
        claimed = np.zeros((self.H, self.W), dtype=bool)
        for rule in self._main_rules:
            mask = match_condition(
                rule.condition, old.kind_grid, new.state_grids, new.env_grids,
                nbhd, self.kind_index, self.signals,
            )
            mask = apply_stochastic(mask, rule, self.rng) & ~claimed
            if mask.any():
                apply_effect(rule, mask, new.kind_grid, new.state_grids, new.env_grids, self._ctx, scratch)
                if rule.action.kind != "noop":
                    claimed |= mask

        # Phase 3: field operations (decay, diffuse)
        apply_field_operations(new.env_grids, self.spec.field_operations, self._offsets)

        obs = compute_observables(
            new.kind_grid, new.state_grids, new.env_grids,
            self.kind_index, self.kind_names,
            self.spec.observables, nbhd,
        )
        return new, obs, nbhd

    def run(self, callback: Any = None) -> Trace:
        trace = Trace(spec=self.spec, kind_names=list(self.kind_names))

        # Detect autocorrelation observables and their lag fields.
        # Use a ring buffer of 2 snapshots to always have a snapshot
        # that is between [lag, 2*lag) steps old — avoids measuring
        # right after a refresh (which would always give 1.0).
        autocorr_fields: dict[str, int] = {}
        autocorr_bufs: dict[str, list[np.ndarray | None]] = {}
        for o in self.spec.observables:
            if o.kind == "field_autocorrelation":
                fname = o.params.get("field", "")
                lag = int(o.params.get("lag", 20))
                if fname:
                    autocorr_fields[fname] = lag
                    autocorr_bufs[fname] = [None, None]  # [older, newer]

        for step in range(self.spec.max_steps):
            # Rotate ring buffer every lag steps
            for fname, lag in autocorr_fields.items():
                if fname in self.state.env_grids and step % lag == 0:
                    buf = autocorr_bufs[fname]
                    buf[0] = buf[1]  # older = previous newer
                    buf[1] = self.state.env_grids[fname].copy()
                    # Expose the OLDER snapshot for autocorrelation
                    if buf[0] is not None:
                        self.state.env_grids[f"_prev_{fname}"] = buf[0]

            new_state, obs, _ = self._step()
            self.state = new_state
            snapshot = (
                self.state.kind_grid.copy()
                if step % self.spec.snapshot_interval == 0
                else None
            )
            trace.frames.append(TraceFrame(step=step, observables=obs, snapshot=snapshot))
            if callback:
                callback(step, self)
        return trace

    @property
    def kind_grid(self) -> np.ndarray:
        return self.state.kind_grid

    @kind_grid.setter
    def kind_grid(self, value: np.ndarray) -> None:
        self.state.kind_grid = value

    @property
    def state_grids(self) -> dict[str, np.ndarray]:
        return self.state.state_grids

    @state_grids.setter
    def state_grids(self, value: dict[str, np.ndarray]) -> None:
        self.state.state_grids = value


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_engine(spec: SimulationSpec) -> GridEngine:
    if spec.topology.kind == "grid":
        return GridEngine(spec)
    raise NotImplementedError(
        f"Topology kind '{spec.topology.kind}' not yet implemented. Use 'grid'."
    )
