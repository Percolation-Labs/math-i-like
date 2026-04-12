# sip-sim

Generalized process simulation engine for grids and networks, with a neural architecture research module implementing **stigmergic (social field) attention**.

Two components:
1. **Simulation engine** — define agent-based models as pure declarative specs (JSON/dict). The engine runs any spec without model-specific code.
2. **Neural module** — GPT with chunk-parallel stigmergic attention, inspired by ant pheromone trail dynamics.

## Quick start

```bash
uv sync

# Simulation engine
uv run python -m sip_sim ant_pheromone        # live window
uv run python -m sip_sim game_of_life --spf 3  # 3x speed

# Neural module
uv run python -c "from sip_sim.neural import GPT; print(GPT())"
```

## Installation

```bash
# Core simulation only (no torch dependency)
uv sync

# With neural module
uv sync --extra neural

# Everything (including experiment deps)
uv sync --extra all
```

## Built-in simulation models

| Model | Key phenomenon | Grid |
|---|---|---|
| `game_of_life` | Emergent structure from 3 rules | 100x100 |
| `schelling` | Mild preference → extreme segregation | 50x50 |
| `forest_fire` | Self-organised criticality | 150x150 |
| `predator_prey` | Spatial Lotka-Volterra coexistence | 80x80 |
| `ant_clustering` | Deneubourg density-dependent mobility | 80x80 |
| `ant_pheromone` | Convergent pheromone trails (equilibrium) | 100x100 |
| `ant_pheromone_transient` | Transient trails (edge of chaos) | 100x100 |

## Simulation API

```python
from sip_sim import load, build_engine, SimulationSpec

# Load and run a built-in model
spec = load("ant_pheromone")
trace = build_engine(spec).run()

# Custom spec from dict/JSON
spec = SimulationSpec(**my_dict)
trace = build_engine(spec).run()

# Live visualization
from sip_sim.viz import live
live(build_engine(load("ant_pheromone")), fps=20, env_field="pheromone")
```

## Neural module

GPT with optional **stigmergic attention** — a chunk-parallel recurrence that modulates key geometry via a deposit-evaporate-modulate cycle, inspired by ant pheromone trail dynamics.

```python
from sip_sim.neural import GPT, get_device, get_tokenizer

model = GPT(
    vocab_size=50257, d_model=384, n_head=6, n_layer=6,
    use_field=True,      # enable social field
    evap_rate=0.05,      # field evaporation rate
    chunk_size=128,      # chunk-parallel attention
    mod_type="additive", # key modulation type
)

# Training utilities
from sip_sim.neural import (
    estimate_loss, generate_sample, save_checkpoint,
    load_checkpoint, cosine_lr_schedule, setup_training_precision,
)
```

### Architecture

- **Chunk-parallel attention**: O(T/C) serial steps instead of O(T). Within each chunk, standard parallel causal attention. Between chunks, field state carries information.
- **Social field**: per-head vector recurrence that modulates key geometry. Deposit from attention output, decay (evaporate) between chunks, modulate keys in next chunk.
- **Two modulation types**: `"additive"` (key shift) or `"gating"` (key scaling).
- **Components**: RMSNorm, rotary position embeddings (RoPE), weight tying.

## Experiments

Research experiments live in `experiments/`:

```
experiments/
├── tinystories/   # 30M GPT on TinyStories (baseline vs social field)
├── owt/           # 125M GPT on OpenWebText (vs GPT-2 Small)
├── gates/         # Synthetic task experiments (modular arithmetic, parity)
├── sweeps/        # Stigmergic attention parameter sweeps (ε, dropout, etc.)
├── sim_sweeps/    # Ant pheromone parameter sweeps (noise × evaporation)
├── results/       # CSV outputs from sweeps
└── docs/          # Research notes and experiment logs
```

### Running experiments

```bash
# TinyStories training
cd experiments/tinystories
python train.py --name baseline --use-field false
python train.py --name social --use-field true

# Text generation
python generate.py --checkpoint checkpoints/social_final.pt --interactive

# Analysis
python analyze.py --baseline checkpoints/baseline_latest.pt \
                   --social checkpoints/social_latest.pt

# OWT 125M training (A100 recommended)
cd experiments/owt
python train.py --name owt_social --use-field true
```

## File structure

```
src/sip_sim/
├── __init__.py          # simulation API
├── __main__.py          # CLI entry point
├── spec.py              # Pydantic spec models
├── engine.py            # simulation engine
├── examples.py          # 7 built-in model specs
├── viz.py               # visualization
└── neural/
    ├── __init__.py      # neural API
    ├── model.py         # GPT + Block
    ├── attention.py     # ChunkParallelAttention
    ├── components.py    # RMSNorm, RoPE
    ├── training.py      # training utilities
    └── data.py          # tokenizer

experiments/             # research scripts (not part of library)
paper/                   # LaTeX papers
validate.py              # simulation validation suite (33 checks)
```

## Validation

```bash
uv run python validate.py
```

Runs all 7 simulation models against quantitative expectations from the literature.

## CLI

```bash
uv run python -m sip_sim <model_name> [--fps 20] [--spf 1] [--env pheromone] [--save out.mp4]
```
