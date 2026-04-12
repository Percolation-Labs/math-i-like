"""sip-sim: Generalized process simulation on grids and networks.

Quick start
───────────
  from sip_sim import SimulationSpec, build_engine, load

  # Run a built-in example
  spec = load("game_of_life")
  trace = build_engine(spec).run()

  # Or build from a dict / JSON
  spec = SimulationSpec(**my_dict)
  trace = build_engine(spec).run()

  # Get the JSON schema (for LLM prompts)
  schema = SimulationSpec.model_json_schema()
"""

from sip_sim.engine import GridEngine, Trace, WorldState, build_engine
from sip_sim.examples import MODELS, load
from sip_sim.spec import SimulationSpec

__all__ = ["SimulationSpec", "GridEngine", "Trace", "build_engine", "load", "MODELS"]
