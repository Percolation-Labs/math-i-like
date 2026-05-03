"""
rdft.crn.viz
============

Render a ``Diagram`` to a PDF panel with full annotation:
the figure, its symmetry factor, the 1PI verdict, and a lineage footer
listing how each field was derived.

This module reuses the drawing primitives from the legacy ``poc_topologies.py``
and ``brw_figures.py`` but plugs them into the unified ``Diagram`` API.
"""
from __future__ import annotations
import math
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rdft.crn.diagram import Diagram


COLOR_A = "black"
COLOR_B = "darkorange"


# ---------------------------------------------------------------------------
# Single-vertex corolla (used for the BRW vertex primitives)
# ---------------------------------------------------------------------------

def draw_corolla(ax, in_a: int, in_b: int, out_a: int, out_b: int,
                 title: Optional[str] = None) -> None:
    """Draw a single vertex with directional, color-coded legs."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.plot(0, 0, "ko", ms=10, zorder=5)

    legs: List[tuple] = []
    legs += [("A", "in")] * in_a
    legs += [("B", "in")] * in_b
    legs += [("A", "out")] * out_a
    legs += [("B", "out")] * out_b
    n = len(legs)
    if n == 0:
        return
    angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]
    R = 0.85
    for (species, direction), angle in zip(legs, angles):
        color = COLOR_A if species == "A" else COLOR_B
        x_tip = R * math.cos(angle)
        y_tip = R * math.sin(angle)
        ax.plot([0, x_tip], [0, y_tip], color=color, lw=1.5)
        if direction == "out":
            ax.annotate("", xy=(x_tip, y_tip),
                        xytext=(0.55 * math.cos(angle), 0.55 * math.sin(angle)),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        else:
            ax.annotate("", xy=(0.30 * math.cos(angle), 0.30 * math.sin(angle)),
                        xytext=(x_tip, y_tip),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    if title is not None:
        ax.set_title(title, fontsize=10, pad=8)


# ---------------------------------------------------------------------------
# 1-loop bubble (2 vertices joined by 2 internal lines)
# ---------------------------------------------------------------------------

def draw_bubble(ax, diagram: Diagram, title: Optional[str] = None) -> None:
    """Render a 1-loop bubble Diagram. Internal-line metadata is in
    ``diagram.internal_lines = ((species, dir), (species, dir))``."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.2, 1.2)

    v1_pos, v2_pos = (-0.55, 0), (0.55, 0)
    ax.plot(*v1_pos, "ko", ms=9, zorder=5)
    ax.plot(*v2_pos, "ko", ms=9, zorder=5)

    (s1, d1), (s2, d2) = diagram.internal_lines
    color1 = COLOR_A if s1 == "A" else COLOR_B
    color2 = COLOR_A if s2 == "A" else COLOR_B

    theta = np.linspace(0, math.pi, 60)
    rx, ry = 0.55, 0.40
    ax.plot(rx * np.cos(theta), ry * np.sin(theta), color=color1, lw=1.6)
    ax.plot(rx * np.cos(theta), -ry * np.sin(theta), color=color2, lw=1.6)

    def arrow_on_arc(x, y, dx, dy, color):
        ax.annotate("", xy=(x + 0.08 * dx, y + 0.08 * dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
    arrow_on_arc(0, ry, +1 if d1 == "lr" else -1, 0, color1)
    arrow_on_arc(0, -ry, +1 if d2 == "lr" else -1, 0, color2)

    if title is not None:
        ax.set_title(title, fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Annotation: relevance + s(G) + lineage
# ---------------------------------------------------------------------------

def _aut_explanation(diagram: Diagram) -> str:
    """One-line note explaining where |Aut| came from, drawn from lineage."""
    for prov in diagram.lineage:
        if "Aut" in prov.detail or "|Aut|" in prov.detail:
            return prov.detail
    return f"|Aut| = {diagram.s_aut}"


def annotate(ax, diagram: Diagram, fontsize: float = 7) -> None:
    """Stamp the relevance verdict + s(G) + lineage onto a diagram axis."""
    color = "tab:green" if diagram.is_1PI else "tab:red"
    verdict = "1PI" if diagram.is_1PI else "REDUCIBLE"
    ax.text(0.5, -0.05, verdict, ha="center", va="top", fontsize=fontsize + 2,
            fontweight="bold", color=color, transform=ax.transAxes)
    if diagram.s_aut is not None:
        ax.text(0.5, -0.13, f"$s(G)={diagram.s_aut}$",
                ha="center", va="top", fontsize=fontsize + 1,
                transform=ax.transAxes)
    aut_note = _aut_explanation(diagram)
    ax.text(0.5, -0.21, aut_note, ha="center", va="top",
            fontsize=fontsize - 1, color="dimgray", transform=ax.transAxes,
            wrap=True)


# ---------------------------------------------------------------------------
# Multi-diagram grid
# ---------------------------------------------------------------------------

def render_diagram_grid(diagrams: Iterable[Diagram], out_path: str,
                        cols: int = 4, kind: str = "bubble",
                        suptitle: Optional[str] = None,
                        figsize_per_panel: tuple = (3.5, 4.0)) -> None:
    """Render a grid of Diagrams to a single PDF.

    ``kind``: ``"bubble"`` or ``"corolla"`` -- selects the drawing primitive.
    """
    diagrams = list(diagrams)
    n = len(diagrams)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * figsize_per_panel[0],
                                       rows * figsize_per_panel[1]))
    if rows == 1:
        axes = np.atleast_2d(axes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.99)

    for i, d in enumerate(diagrams):
        ax = axes[i // cols][i % cols]
        if kind == "bubble":
            t = (f"{'+'.join(d.vertex_names)}    int: "
                 f"{d.internal_lines[0][0]}{d.internal_lines[1][0]}\n"
                 f"$s(G)={d.s_aut}$")
            draw_bubble(ax, d, title=t)
        elif kind == "corolla":
            # vertex_names should hold a single name; legs from external_legs
            ext = d.external_legs
            draw_corolla(ax,
                         in_a=ext.get(("A", "in"), 0),
                         in_b=ext.get(("B", "in"), 0),
                         out_a=ext.get(("A", "out"), 0),
                         out_b=ext.get(("B", "out"), 0),
                         title=d.id)
        annotate(ax, d)

    # blank out extra cells
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95 if suptitle else 1.0])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
