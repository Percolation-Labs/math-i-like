"""Visualization utilities for simulation traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from sip_sim.engine import EMPTY, GridEngine, Trace
from sip_sim.spec import SimulationSpec


def make_colormap(kind_names: list[str], kind_specs: dict) -> ListedColormap:
    """Build a colormap from kind colors. Index 0 = empty = white."""
    colors = ["#ffffff"]  # empty
    for i, name in enumerate(kind_names[1:], start=1):
        if i in kind_specs:
            colors.append(kind_specs[i].color)
        else:
            colors.append("#888888")
    return ListedColormap(colors)


def plot_grid(
    kind_grid: np.ndarray,
    kind_names: list[str],
    colors: list[str] | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot a single grid state."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    if colors is None:
        colors = ["#ffffff"] + [f"C{i}" for i in range(len(kind_names) - 1)]

    cmap = ListedColormap(colors[: len(kind_names)])
    ax.imshow(kind_grid, cmap=cmap, vmin=0, vmax=len(kind_names) - 1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_observables(trace: Trace, names: list[str] | None = None, save_path: str | None = None) -> plt.Figure:
    """Plot observable time series from a trace."""
    if names is None:
        # Collect all observable names
        all_names: set[str] = set()
        for f in trace.frames:
            all_names.update(f.observables.keys())
        names = sorted(all_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        steps, values = trace.observable_series(name)
        ax.plot(steps, values, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title(f"{trace.spec.name} — Observables")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_snapshots(
    trace: Trace,
    colors: list[str] | None = None,
    max_panels: int = 12,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot grid snapshots in a panel grid."""
    snaps = trace.snapshots()
    if not snaps:
        raise ValueError("No snapshots in trace")

    # Subsample if too many
    if len(snaps) > max_panels:
        indices = np.linspace(0, len(snaps) - 1, max_panels, dtype=int)
        snaps = [snaps[i] for i in indices]

    n = len(snaps)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    if colors is None:
        colors = ["#ffffff"] + [f"C{i}" for i in range(len(trace.kind_names) - 1)]

    cmap = ListedColormap(colors[: len(trace.kind_names)])

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (step, grid) in enumerate(snaps):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(trace.kind_names) - 1, interpolation="nearest")
        ax.set_title(f"t={step}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(trace.spec.name, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def live(
    engine: GridEngine,
    *,
    fps: int = 20,
    steps_per_frame: int = 1,
    env_field: str | None = None,
    colors: list[str] | None = None,
    save_path: str | None = None,
) -> FuncAnimation:
    """Run simulation live with real-time visualization.

    Args:
        engine: A GridEngine (already initialized, not yet run).
        fps: Target frames per second.
        steps_per_frame: How many simulation steps per rendered frame.
        env_field: If set, show this env field as a heatmap alongside agents.
        colors: Per-kind colors (index 0 = empty). Auto-generated if None.
        save_path: If set, save animation to file (.mp4 or .gif).

    Returns:
        The FuncAnimation object (keep a reference to prevent GC).
    """
    spec = engine.spec
    n_kinds = len(engine.kind_names)

    if colors is None:
        colors = ["#ffffff"]
        for i, ak in enumerate(spec.agent_kinds):
            colors.append(ak.color)

    cmap = ListedColormap(colors[:n_kinds])
    show_env = env_field is not None and env_field in engine.state.env_grids

    if show_env:
        fig, (ax_grid, ax_env) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(7, 7))
        ax_env = None

    # Initial images
    im_grid = ax_grid.imshow(
        engine.state.kind_grid, cmap=cmap, vmin=0, vmax=n_kinds - 1,
        interpolation="nearest",
    )
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    title = ax_grid.set_title(f"{spec.name}  t=0", fontsize=12)

    im_env = None
    if show_env and ax_env is not None:
        im_env = ax_env.imshow(
            engine.state.env_grids[env_field], cmap="hot",
            interpolation="nearest", vmin=0, vmax=1,
        )
        ax_env.set_xticks([])
        ax_env.set_yticks([])
        ax_env.set_title(env_field, fontsize=12)
        plt.colorbar(im_env, ax=ax_env, shrink=0.8)

    fig.tight_layout()

    step_counter = [0]

    def update(_frame):
        for _ in range(steps_per_frame):
            new_state, obs, _ = engine._step()
            engine.state = new_state
            step_counter[0] += 1

        im_grid.set_data(engine.state.kind_grid)
        title.set_text(f"{spec.name}  t={step_counter[0]}")

        if im_env is not None and show_env:
            field_data = engine.state.env_grids[env_field]
            im_env.set_data(field_data)
            fmax = field_data.max()
            if fmax > 0:
                im_env.set_clim(0, fmax)

        return (im_grid, title) + ((im_env,) if im_env else ())

    n_frames = spec.max_steps // steps_per_frame
    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    if save_path:
        suffix = Path(save_path).suffix.lower()
        if suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return anim
