"""Simulation specification models — everything needed to define a simulation.

The SimulationSpec is a complete, self-contained description of an agent-based
model.  It can be serialised to/from JSON and contains no code — the engine
interprets it generically.

Architecture
────────────
  SimulationSpec
  ├── Topology          — grid dimensions, wrapping, neighbourhood
  ├── AgentKind[]       — species: name, state fields, initial placement
  ├── Rule[]            — condition → action pairs (the model's dynamics)
  │   ├── Condition     — predicates over self/neighbour/env/signal state
  │   └── Action        — transform/die/spawn/move/modify_state/noop
  │       ├── BehaviorStep[]   — perception/decision pipeline (sense → gate)
  │       ├── MoveWeight[]     — equation of motion P(j) = ∏ term(j)
  │       └── HeadingSpec      — directional persistence config
  ├── EnvField[]        — persistent environment layers (pheromone, food…)
  ├── FieldOperation[]  — per-step field dynamics (decay, diffuse)
  ├── GlobalSignal[]    — exogenous/endogenous global variables
  └── ObservableSpec[]  — what to measure each step
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Topology
# ──────────────────────────────────────────────────────────────────────


class Topology(BaseModel):
    """The space agents live on."""

    kind: Literal["grid", "hexgrid", "network"] = Field(
        "grid", description="Spatial substrate type.",
    )
    width: int | None = Field(None, description="Grid columns.", ge=1)
    height: int | None = Field(None, description="Grid rows.", ge=1)
    wrap: bool = Field(False, description="Toroidal wrapping (edges connect).")

    # Network
    graph_type: Literal[
        "erdos_renyi", "barabasi_albert", "watts_strogatz", "complete", "lattice",
    ] | None = Field(None, description="Network topology generator.")
    n_nodes: int | None = Field(None, description="Number of network nodes.", ge=1)
    graph_params: dict[str, Any] = Field(
        default_factory=dict, description="Extra params for the graph generator.",
    )

    # Shared
    neighbourhood: Literal["von_neumann", "moore", "graph_adjacent"] = Field(
        "moore", description="Which cells count as neighbours.",
    )
    neighbourhood_radius: int = Field(
        1, description="Neighbourhood radius (1 = immediate neighbours).", ge=1,
    )


# ──────────────────────────────────────────────────────────────────────
# Agent species
# ──────────────────────────────────────────────────────────────────────


class AgentKind(BaseModel):
    """A species definition.

    Each kind is an independent agent type with its own state fields,
    visual appearance, and initial placement.  The special kind name
    ``__empty__`` refers to unoccupied cells and cannot be used here.
    """

    name: str = Field(..., description="Unique species identifier (e.g. 'ant', 'tree').")
    state_schema: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-agent state fields and their types.  Keys are field names, "
            "values are type hints (currently only 'float' is supported).  "
            "Example: {'hunger': 'float', 'dir_r': 'float', 'dir_c': 'float'}"
        ),
    )
    color: str = Field("#888888", description="Hex colour for visualisation.")
    symbol: str = Field(".", description="Single character for text rendering.")
    initial_fraction: float | None = Field(
        None, description="Fraction of grid cells to seed with this kind (0–1).",
    )
    initial_count: int | None = Field(
        None, description="Exact number of agents to seed (overrides fraction).",
    )
    initial_placement: Literal[
        "uniform_random", "clustered", "left", "right", "center", "border", "custom",
    ] = Field("uniform_random", description="Spatial seeding strategy.")
    initial_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Starting values for state_schema fields.  Example: {'hunger': 0}",
    )
    mobile: bool = Field(
        False,
        description="Whether agents of this kind can move.  Immobile agents ignore move rules.",
    )


# ──────────────────────────────────────────────────────────────────────
# Conditions
# ──────────────────────────────────────────────────────────────────────


class NumberPredicate(BaseModel):
    """Flexible numeric comparison — all specified bounds must hold."""

    eq: float | None = None
    lt: float | None = None
    lte: float | None = None
    gt: float | None = None
    gte: float | None = None

    def matches(self, value: float) -> bool:
        if self.eq is not None and value != self.eq:
            return False
        if self.lt is not None and value >= self.lt:
            return False
        if self.lte is not None and value > self.lte:
            return False
        if self.gt is not None and value <= self.gt:
            return False
        if self.gte is not None and value < self.gte:
            return False
        return True


class Condition(BaseModel):
    """A predicate over local state + neighbourhood.

    All specified fields are ANDed together.  Omitted fields are
    unconstrained (match everything).
    """

    self_kind: str | None = Field(
        None,
        description="Agent kind at this cell (e.g. 'ant').  Use '__empty__' for vacant cells.",
    )
    self_state: dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            "Predicates on the agent's own state fields.  "
            "Example: {'hunger': {'gte': 10}} — agent whose hunger ≥ 10."
        ),
    )
    neighbour_kind_count: dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            "Predicates on raw count of neighbours of a given kind.  "
            "Example: {'prey': {'gte': 1}} — at least one prey neighbour."
        ),
    )
    neighbour_kind_fraction: dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            "Predicates on fraction of neighbours that are a given kind.  "
            "Example: {'red': {'lt': 0.3}} — fewer than 30% red neighbours."
        ),
    )
    global_signal: dict[str, dict[str, float]] | None = Field(
        None, description="Predicates on global signal values.",
    )
    random_threshold: float | None = Field(
        None,
        description="Rule fires with this probability (0–1).  Acts as a stochastic gate.",
    )
    env_state: dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            "Predicates on environment field values at this cell.  "
            "Example: {'temperature': {'gt': 100}} — cell hotter than 100."
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Behavior pipeline (perception → decision)
# ──────────────────────────────────────────────────────────────────────


class BehaviorStep(BaseModel):
    """One step in a declarative behavior pipeline.

    Pipeline steps execute sequentially before the move action.
    They compute per-agent scratch variables that control movement
    and deposition.

    Step kinds
    ──────────
    sense       Read an env field in the neighbourhood, reduce to scalar.
                params: field, method (max_neighbour|sum_neighbour|at_cell),
                        output (default=field name)

    threshold   Hard threshold: output=1 if input > value, else 0.
                params: input, value, output (default="_follow")

    hill        Soft sigmoid: P = φⁿ/(φⁿ+sⁿ), then coin flip.
                params: input, half_max, steepness, output (default="_follow")

    coin        Simple probability gate: output=1 with probability p.
                params: p, output (default="_follow")

    habituation Temporal hysteresis with persistent counters.
                params: input (default="_follow"), tau, recovery (default=1.0),
                        gate_deposit (default=false), deposit_var (default="_deposit")

    crowding    Override follow when too many agent neighbours.
                params: max_neighbours, input (default="_follow"),
                        output (default="_follow")

    saturation  Peaked response: effective = field × exp(−field/peak).
                params: field, peak, output_field (default="_effective_{field}")

    gate        Gate a modify_state action on a scratch var > 0.5.
                params: requires
    """

    kind: Literal[
        "sense", "threshold", "hill", "coin",
        "habituation", "crowding", "saturation", "gate",
    ] = Field(..., description="Pipeline step type.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Step-specific parameters (see docstring).",
    )


# ──────────────────────────────────────────────────────────────────────
# Movement equation of motion
# ──────────────────────────────────────────────────────────────────────


class MoveWeight(BaseModel):
    """A multiplicative term in the movement probability equation.

    For each candidate destination cell j the total weight is:

        W(j) = ∏_k  term_k(j)

    The agent samples its destination from P(j) = W(j) / ΣW.

    Terms with a ``requires`` field are only active when the named
    scratch variable (set by the behavior pipeline) is > 0.5 for the
    moving agent.  When inactive, the term contributes 1.0 (neutral).

    Kinds
    ─────
    field       w *= (floor + env_field[j])^power
                Attraction toward higher values of an environment field.
                params: field (str), floor (float=0.01), power (float=1.0)

    momentum    w *= max(floor, 1 + strength × cos(heading, dir_to_j))
                Directional persistence using the agent's heading state.
                params: strength (float), floor (float=0.01)

    avoid_kind  w *= max(floor, 1 − kind_count_near_j / threshold)
                Repulsion from directions crowded with a given kind.
                params: kind (str), threshold (float), floor (float=0.01)
    """

    kind: Literal["field", "momentum", "avoid_kind"] = Field(
        ..., description="Weight term type.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Term-specific parameters (see docstring).",
    )
    requires: str | None = Field(
        None,
        description=(
            "Scratch variable that must be > 0.5 for this term to apply.  "
            "When the variable is ≤ 0.5, the term is skipped (contributes 1.0)."
        ),
    )


class HeadingSpec(BaseModel):
    """How to update agent heading after each move.

    Heading is stored as a 2D vector in two agent state fields.
    Updated via exponential moving average after each step:

        h_new = (1 − alpha) × h_old + alpha × move_direction
    """

    fields: list[str] = Field(
        ["dir_r", "dir_c"],
        description="State field names for the heading vector components.",
    )
    alpha: float = Field(
        0.3,
        description="EMA blending factor (0 = never update, 1 = instant snap to new direction).",
        ge=0.0, le=1.0,
    )


# ──────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────


class Action(BaseModel):
    """What happens when a rule fires.

    Action kinds
    ────────────
    transform     Change agent to a different kind (+ optional state update).
    spawn         Create a new agent of ``new_kind`` in an adjacent empty cell.
    die           Remove the agent; agent state fields are zeroed, env fields persist.
    move          Relocate the agent (local, global, biased, or weighted).
    swap          Exchange position with a neighbour (not yet implemented).
    modify_state  Update state or env field values (stackable, all matching fire).
    noop          Do nothing (useful for explicit survival rules).
    """

    kind: Literal[
        "transform", "spawn", "die", "move", "swap", "modify_state", "noop",
    ] = Field(..., description="Action type.")
    new_kind: str | None = Field(
        None, description="Target kind for transform/spawn actions.",
    )
    state_update: dict[str, Any] | None = Field(
        None,
        description=(
            "Field updates for transform/modify_state.  "
            "Values can be absolute ('0') or incremental ('+1', '+5.0').  "
            "Works on both agent state_grids and env_grids."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (e.g. scope='global' for move).",
    )
    pipeline: list[BehaviorStep] | None = Field(
        None,
        description=(
            "Declarative behavior pipeline — composable perception/decision steps "
            "that compute scratch variables before movement."
        ),
    )
    move_weights: list[MoveWeight] | None = Field(
        None,
        description=(
            "Equation of motion — multiplicative weight terms that define "
            "P(destination) for movement.  Replaces hardcoded weight formulas."
        ),
    )
    heading: HeadingSpec | None = Field(
        None,
        description="Heading update config for agents with directional persistence.",
    )
    deposit_gate: str | None = Field(
        None,
        description=(
            "Scratch variable name that gates a modify_state action.  "
            "Only agents where this variable > 0.5 will apply the state_update."
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Rules
# ──────────────────────────────────────────────────────────────────────


class Rule(BaseModel):
    """A reaction: condition → action.

    Rules are evaluated in priority order (highest first).  Within the
    same priority, evaluation order follows the list.

    Two rule phases exist:
      - **modify_state** rules are stackable (all matching fire, no claiming).
      - All other rules are exclusive (first match claims the cell).
    """

    name: str = Field(..., description="Human-readable rule identifier.")
    condition: Condition = Field(..., description="When this rule fires.")
    action: Action = Field(..., description="What happens when it fires.")
    priority: int = Field(
        0, description="Higher priority rules are evaluated first.",
    )
    rate: float = Field(
        1.0,
        description="Probability that a matched rule actually fires (0–1).",
        ge=0.0, le=1.0,
    )
    description: str = Field("", description="Optional human-readable explanation.")


# ──────────────────────────────────────────────────────────────────────
# Environment & signals
# ──────────────────────────────────────────────────────────────────────


class GlobalSignal(BaseModel):
    """An exogenous or endogenous global variable."""

    name: str = Field(..., description="Signal identifier.")
    initial_value: float = Field(0.0, description="Value at t=0.")
    update_rule: Literal["constant", "linear", "sinusoidal", "from_aggregate"] = Field(
        "constant", description="How the signal evolves each step.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the update rule.",
    )


class EnvField(BaseModel):
    """A persistent environment field — not tied to agent presence.

    Unlike agent state_schema fields, env fields persist when agents
    die or move away.  Used for pheromone trails, resource maps,
    temperature gradients, etc.
    """

    name: str = Field(..., description="Field identifier (e.g. 'pheromone').")
    initial_value: float = Field(0.0, description="Value at every cell at t=0.")


class FieldOperation(BaseModel):
    """A global operation applied to an env field every step.

    Runs after all rules, before observables.
    """

    field: str = Field(..., description="Which env field to operate on.")
    kind: Literal["decay", "diffuse"] = Field(
        ...,
        description=(
            "'decay': field *= (1 − rate).  "
            "'diffuse': field += rate × laplacian(field)."
        ),
    )
    rate: float = Field(..., description="Operation strength per step.")


# ──────────────────────────────────────────────────────────────────────
# Observables
# ──────────────────────────────────────────────────────────────────────


class ObservableSpec(BaseModel):
    """What to measure each step.

    Observable kinds
    ────────────────
    count               Number of agents of a kind.
    density             Fraction of cells occupied by a kind.
    entropy             Shannon entropy over kind proportions.
    aggregate_state     Mean/sum/max/min of a state or env field.
    same_kind_fraction  Mean fraction of same-kind neighbours.
    spatial_entropy     Shannon entropy of spatial distribution (block-based).
    env_max / env_mean  Max or mean of an environment field.
    directional_alignment  Nematic order parameter for heading vectors.
    trail_linearity     Fraction of high-field cells forming linear chains.
    field_autocorrelation  Temporal correlation of a field (trail turnover).
    """

    name: str = Field(..., description="Observable identifier for time series.")
    kind: Literal[
        "count", "density", "spatial_cluster", "boundary_length",
        "aggregate_state", "entropy", "same_kind_fraction", "spatial_entropy",
        "env_max", "env_mean", "directional_alignment", "trail_linearity",
        "field_autocorrelation",
    ] = Field(..., description="Measurement type.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific params (e.g. {'kind': 'ant'} for count).",
    )


# ──────────────────────────────────────────────────────────────────────
# Top-level spec
# ──────────────────────────────────────────────────────────────────────


class SimulationSpec(BaseModel):
    """Complete specification for a simulation run.

    This is the single object that fully defines an agent-based model.
    It contains no code — only data.  The engine interprets it generically.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                # ── Example 1: Game of Life ──────────────────────────
                # Simplest possible model: 1 species, neighbourhood
                # count rules, transform/die/noop actions.
                {
                    "name": "Game of Life",
                    "description": "Conway's Game of Life — emergent structure from 3 rules",
                    "topology": {
                        "kind": "grid", "width": 100, "height": 100,
                        "wrap": True, "neighbourhood": "moore",
                    },
                    "agent_kinds": [
                        {"name": "alive", "color": "#000000", "initial_fraction": 0.3},
                    ],
                    "rules": [
                        {
                            "name": "underpopulation",
                            "condition": {"self_kind": "alive", "neighbour_kind_count": {"alive": {"lt": 2}}},
                            "action": {"kind": "die"},
                            "priority": 1,
                        },
                        {
                            "name": "overpopulation",
                            "condition": {"self_kind": "alive", "neighbour_kind_count": {"alive": {"gt": 3}}},
                            "action": {"kind": "die"},
                            "priority": 1,
                        },
                        {
                            "name": "survival",
                            "condition": {"self_kind": "alive", "neighbour_kind_count": {"alive": {"gte": 2, "lte": 3}}},
                            "action": {"kind": "noop"},
                            "priority": 1,
                        },
                        {
                            "name": "birth",
                            "condition": {"self_kind": "__empty__", "neighbour_kind_count": {"alive": {"eq": 3}}},
                            "action": {"kind": "transform", "new_kind": "alive"},
                            "priority": 0,
                        },
                    ],
                    "observables": [
                        {"name": "population", "kind": "count", "params": {"kind": "alive"}},
                        {"name": "density", "kind": "density", "params": {"kind": "alive"}},
                    ],
                    "max_steps": 200,
                    "snapshot_interval": 20,
                    "seed": 42,
                },

                # ── Example 2: Predator-Prey ─────────────────────────
                # Multi-species with per-agent state, reproduction,
                # starvation, and stochastic movement rates.
                {
                    "name": "Predator-Prey",
                    "description": "Spatial Lotka-Volterra with hunger, starvation, and reproduction",
                    "topology": {
                        "kind": "grid", "width": 80, "height": 80,
                        "wrap": True, "neighbourhood": "moore",
                    },
                    "agent_kinds": [
                        {
                            "name": "prey", "color": "#2ecc71",
                            "initial_fraction": 0.3, "mobile": True,
                            "state_schema": {"age": "float"},
                            "initial_state": {"age": 0},
                        },
                        {
                            "name": "predator", "color": "#e74c3c",
                            "initial_fraction": 0.05, "mobile": True,
                            "state_schema": {"hunger": "float"},
                            "initial_state": {"hunger": 0},
                        },
                    ],
                    "rules": [
                        {
                            "name": "predator_eats",
                            "description": "Predator near prey resets hunger",
                            "condition": {"self_kind": "predator", "neighbour_kind_count": {"prey": {"gte": 1}}},
                            "action": {"kind": "modify_state", "state_update": {"hunger": "0"}},
                            "priority": 3,
                        },
                        {
                            "name": "predator_starve",
                            "description": "Predator dies after 10 steps without eating",
                            "condition": {"self_kind": "predator", "self_state": {"hunger": {"gte": 10}}},
                            "action": {"kind": "die"},
                            "priority": 2,
                        },
                        {
                            "name": "predator_hunger_tick",
                            "condition": {"self_kind": "predator"},
                            "action": {"kind": "modify_state", "state_update": {"hunger": "+1"}},
                            "priority": -1,
                        },
                        {
                            "name": "prey_reproduce",
                            "condition": {"self_kind": "prey", "random_threshold": 0.04},
                            "action": {"kind": "spawn", "new_kind": "prey"},
                            "priority": 1,
                        },
                        {
                            "name": "predator_reproduce",
                            "condition": {"self_kind": "predator", "self_state": {"hunger": {"lt": 3}}, "random_threshold": 0.02},
                            "action": {"kind": "spawn", "new_kind": "predator"},
                            "priority": 1,
                        },
                        {
                            "name": "prey_move",
                            "condition": {"self_kind": "prey"},
                            "action": {"kind": "move"},
                            "priority": 0, "rate": 0.5,
                        },
                        {
                            "name": "predator_move",
                            "condition": {"self_kind": "predator"},
                            "action": {"kind": "move"},
                            "priority": 0, "rate": 0.8,
                        },
                    ],
                    "observables": [
                        {"name": "prey", "kind": "count", "params": {"kind": "prey"}},
                        {"name": "predators", "kind": "count", "params": {"kind": "predator"}},
                    ],
                    "max_steps": 500,
                    "snapshot_interval": 25,
                    "seed": 42,
                },

                # ── Example 3: Stigmergic Ants (full pipeline) ───────
                # Environment fields, field operations, declarative
                # behavior pipeline (sense → hill → habituation → crowding),
                # equation-of-motion weights, heading update, and
                # gated deposition.  This is the most complex spec.
                {
                    "name": "Stigmergic Ants",
                    "description": "Pheromone trail formation with Hill-function detection, habituation, and crowding avoidance",
                    "topology": {
                        "kind": "grid", "width": 80, "height": 80,
                        "wrap": True, "neighbourhood": "moore",
                    },
                    "agent_kinds": [
                        {
                            "name": "ant", "color": "#e74c3c",
                            "initial_fraction": 0.04, "mobile": True,
                            "state_schema": {"dir_r": "float", "dir_c": "float"},
                            "initial_state": {"dir_r": 0, "dir_c": 0},
                        },
                    ],
                    "env_fields": [
                        {"name": "pheromone", "initial_value": 0.0},
                    ],
                    "field_operations": [
                        {"field": "pheromone", "kind": "decay", "rate": 0.15},
                        {"field": "pheromone", "kind": "diffuse", "rate": 0.005},
                    ],
                    "rules": [
                        {
                            "name": "deposit_pheromone",
                            "description": "Ants deposit pheromone — gated by _deposit (deaf ants don't deposit)",
                            "condition": {"self_kind": "ant"},
                            "action": {
                                "kind": "modify_state",
                                "state_update": {"pheromone": "+5.0"},
                                "deposit_gate": "_deposit",
                            },
                            "priority": 1,
                        },
                        {
                            "name": "move",
                            "description": "Sense pheromone → Hill function decision → habituation → crowding → weighted move",
                            "condition": {"self_kind": "ant"},
                            "action": {
                                "kind": "move",
                                "pipeline": [
                                    {"kind": "sense", "params": {"field": "pheromone", "method": "max_neighbour", "output": "phi"}},
                                    {"kind": "hill", "params": {"input": "phi", "half_max": 3.0, "steepness": 2.0, "output": "_follow"}},
                                    {"kind": "habituation", "params": {"input": "_follow", "tau": 50, "recovery": 3.0, "gate_deposit": True, "deposit_var": "_deposit"}},
                                    {"kind": "crowding", "params": {"max_neighbours": 4, "input": "_follow", "output": "_follow"}},
                                ],
                                "move_weights": [
                                    {"kind": "field", "params": {"field": "pheromone", "floor": 0.01, "power": 1.0}, "requires": "_follow"},
                                    {"kind": "momentum", "params": {"strength": 2.0}},
                                ],
                                "heading": {"fields": ["dir_r", "dir_c"], "alpha": 0.3},
                            },
                            "priority": 0,
                        },
                    ],
                    "observables": [
                        {"name": "ants", "kind": "count", "params": {"kind": "ant"}},
                        {"name": "pheromone_max", "kind": "env_max", "params": {"field": "pheromone"}},
                        {"name": "pheromone_mean", "kind": "env_mean", "params": {"field": "pheromone"}},
                        {"name": "trail_linearity", "kind": "trail_linearity", "params": {"field": "pheromone", "threshold_frac": 0.3}},
                        {"name": "trail_turnover", "kind": "field_autocorrelation", "params": {"field": "pheromone", "lag": 100}},
                    ],
                    "max_steps": 1000,
                    "snapshot_interval": 100,
                    "seed": 42,
                },

                # ── Example 4: Forest Fire ───────────────────────────
                # Self-organised criticality: species transform chains,
                # stochastic ignition, von Neumann neighbourhood.
                {
                    "name": "Forest Fire",
                    "description": "Drossel-Schwabl: growth, lightning, fire spread, burnout",
                    "topology": {
                        "kind": "grid", "width": 150, "height": 150,
                        "wrap": False, "neighbourhood": "von_neumann",
                    },
                    "agent_kinds": [
                        {"name": "tree", "color": "#2ecc71", "initial_fraction": 0.6},
                        {"name": "burning", "color": "#e74c3c"},
                    ],
                    "rules": [
                        {
                            "name": "burnout",
                            "condition": {"self_kind": "burning"},
                            "action": {"kind": "die"},
                            "priority": 3,
                        },
                        {
                            "name": "fire_spread",
                            "condition": {"self_kind": "tree", "neighbour_kind_count": {"burning": {"gte": 1}}},
                            "action": {"kind": "transform", "new_kind": "burning"},
                            "priority": 2,
                        },
                        {
                            "name": "lightning",
                            "condition": {"self_kind": "tree", "random_threshold": 0.00005},
                            "action": {"kind": "transform", "new_kind": "burning"},
                            "priority": 1,
                        },
                        {
                            "name": "regrowth",
                            "condition": {"self_kind": "__empty__", "random_threshold": 0.01},
                            "action": {"kind": "transform", "new_kind": "tree"},
                            "priority": 0,
                        },
                    ],
                    "observables": [
                        {"name": "trees", "kind": "count", "params": {"kind": "tree"}},
                        {"name": "burning", "kind": "count", "params": {"kind": "burning"}},
                        {"name": "tree_density", "kind": "density", "params": {"kind": "tree"}},
                    ],
                    "max_steps": 500,
                    "snapshot_interval": 25,
                    "seed": 42,
                },

                # ── Example 5: Schelling Segregation ─────────────────
                # Two species, fraction-based neighbourhood conditions,
                # global (teleport) movement.
                {
                    "name": "Schelling Segregation",
                    "description": "Mild preference (30%) produces extreme segregation",
                    "topology": {
                        "kind": "grid", "width": 50, "height": 50,
                        "wrap": True, "neighbourhood": "moore",
                    },
                    "agent_kinds": [
                        {"name": "red", "color": "#e74c3c", "initial_fraction": 0.45, "mobile": True},
                        {"name": "blue", "color": "#3498db", "initial_fraction": 0.45, "mobile": True},
                    ],
                    "rules": [
                        {
                            "name": "red_unhappy",
                            "condition": {"self_kind": "red", "neighbour_kind_fraction": {"red": {"lt": 0.3}}},
                            "action": {"kind": "move", "params": {"scope": "global"}},
                        },
                        {
                            "name": "blue_unhappy",
                            "condition": {"self_kind": "blue", "neighbour_kind_fraction": {"blue": {"lt": 0.3}}},
                            "action": {"kind": "move", "params": {"scope": "global"}},
                        },
                    ],
                    "observables": [
                        {"name": "red", "kind": "count", "params": {"kind": "red"}},
                        {"name": "blue", "kind": "count", "params": {"kind": "blue"}},
                        {"name": "segregation", "kind": "same_kind_fraction"},
                    ],
                    "max_steps": 100,
                    "snapshot_interval": 10,
                    "seed": 42,
                },
            ],
        },
    }

    name: str = Field(..., description="Model name.")
    description: str = Field("", description="Human-readable model description.")
    topology: Topology = Field(..., description="Spatial substrate configuration.")
    agent_kinds: list[AgentKind] = Field(
        ..., description="Species definitions.  At least one required.",
    )
    rules: list[Rule] = Field(
        ..., description="The model's dynamics — condition → action pairs.",
    )
    global_signals: list[GlobalSignal] = Field(
        default_factory=list, description="Exogenous/endogenous global variables.",
    )
    env_fields: list[EnvField] = Field(
        default_factory=list,
        description="Persistent environment layers (pheromone, temperature, etc.).",
    )
    field_operations: list[FieldOperation] = Field(
        default_factory=list,
        description="Per-step field dynamics (evaporation, diffusion).",
    )
    observables: list[ObservableSpec] = Field(
        default_factory=list, description="Measurements recorded each step.",
    )
    max_steps: int = Field(1000, description="Simulation duration.", ge=1)
    snapshot_interval: int = Field(
        10, description="Steps between full grid snapshots.", ge=1,
    )
    seed: int | None = Field(
        None, description="RNG seed for reproducibility (None = random).",
    )
