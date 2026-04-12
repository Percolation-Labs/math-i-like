"""Run a model live: python -m sip_sim <model_name> [--save out.mp4]

Available models: game_of_life, schelling, forest_fire, predator_prey, ant_clustering, ant_pheromone
"""

import argparse
import sys

from sip_sim import MODELS, build_engine, load
from sip_sim.viz import live


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sip-sim models with live visualization")
    parser.add_argument("model", choices=list(MODELS.keys()), help="Model to run")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (default: 20)")
    parser.add_argument("--spf", type=int, default=1, help="Simulation steps per frame (default: 1)")
    parser.add_argument("--env", type=str, default=None, help="Environment field to show as heatmap")
    parser.add_argument("--save", type=str, default=None, help="Save animation to file (.mp4 or .gif)")
    args = parser.parse_args()

    spec = load(args.model)
    engine = build_engine(spec)

    # Auto-detect env field for known models
    env_field = args.env
    if env_field is None and spec.env_fields:
        env_field = spec.env_fields[0].name

    colors = ["#ffffff"] + [ak.color for ak in spec.agent_kinds]

    print(f"Running {spec.name} ({spec.topology.width}x{spec.topology.height}, {spec.max_steps} steps)")
    if env_field:
        print(f"Showing env field: {env_field}")

    live(
        engine,
        fps=args.fps,
        steps_per_frame=args.spf,
        env_field=env_field,
        colors=colors,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
