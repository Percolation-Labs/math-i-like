"""
rdft.graphs.tikz
=================
TikZ code generation for Feynman diagrams in the thesis style.

Produces publication-quality TikZ code matching the style of
Amarteifio (2019) thesis: directed edges with arrows, vertices
as filled dots, external legs as dashed lines, species distinguished
by colour (black for A, orange for B).

Usage:
    from rdft.graphs.tikz import diagram_to_tikz, corolla_to_tikz
    tikz_code = diagram_to_tikz(feynman_graph)
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import math
import sympy as sp

from .incidence import FeynmanGraph


def corolla_to_tikz(m_out: int, n_in: int, coupling: str = '',
                     label: str = '', scale: float = 0.8) -> str:
    """
    Generate TikZ code for a single corolla (amputated vertex).

    Draws a central dot with m_out outgoing arrows and n_in incoming arrows,
    arranged radially. Matches thesis Figure 3.3 style.
    """
    lines = [f'% Corolla: φ̃^{m_out} φ^{n_in}  coupling={coupling}']
    lines.append(r'\begin{tikzpicture}[>=Stealth, scale=' + f'{scale}]')

    total = m_out + n_in
    if total == 0:
        lines.append(r'  \fill (0,0) circle (2pt);')
        lines.append(r'\end{tikzpicture}')
        return '\n'.join(lines)

    # Central vertex
    lines.append(r'  \fill (0,0) circle (2.5pt);')

    # Arrange legs radially
    angle_step = 360.0 / total if total > 0 else 0
    leg_length = 0.8

    idx = 0
    # Outgoing legs (φ̃): arrows pointing outward
    for i in range(m_out):
        angle = 90 + idx * angle_step  # start from top
        rad = math.radians(angle)
        x = leg_length * math.cos(rad)
        y = leg_length * math.sin(rad)
        lines.append(f'  \\draw[->, thick] (0,0) -- ({x:.3f},{y:.3f});')
        idx += 1

    # Incoming legs (φ): arrows pointing inward
    for i in range(n_in):
        angle = 90 + idx * angle_step
        rad = math.radians(angle)
        x = leg_length * math.cos(rad)
        y = leg_length * math.sin(rad)
        lines.append(f'  \\draw[<-, thick] (0,0) -- ({x:.3f},{y:.3f});')
        idx += 1

    # Label
    if label:
        lines.append(f'  \\node[below=4pt] at (0,{-leg_length-0.1:.2f}) '
                     f'{{\\small ${label}$}};')

    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


def diagram_to_tikz(graph: FeynmanGraph,
                     title: str = '',
                     show_labels: bool = True,
                     vertex_labels: Optional[Dict[int, str]] = None,
                     scale: float = 1.0) -> str:
    """
    Generate TikZ code for a Feynman diagram in thesis style.

    Style: vertices as filled dots, internal edges as thick directed
    arrows with Schwinger parameter labels, external legs as thin
    dashed arrows.
    """
    n_v = graph.n_vertices_int
    lines = [f'% Feynman diagram: V={n_v}, E_int={graph.n_internal_edges}, L={graph.L}']
    lines.append(r'\begin{tikzpicture}[>=Stealth, scale=' + f'{scale}, '
                 r'every node/.style={{font=\small}}]')

    # Vertex positions: arrange on a circle for > 2 vertices,
    # or horizontally for 2 vertices
    positions = _vertex_positions(n_v, scale)

    # Draw vertices
    for v in range(n_v):
        x, y = positions[v]
        label = vertex_labels.get(v, '') if vertex_labels else ''
        lines.append(f'  \\fill ({x:.2f},{y:.2f}) circle (3pt);')
        if label:
            lines.append(f'  \\node[above=4pt] at ({x:.2f},{y:.2f}) '
                        f'{{${label}$}};')

    # Draw internal edges
    alpha_idx = 0
    edge_counts: Dict[Tuple[int,int], int] = {}  # track multi-edges

    for i, (src, tgt, is_ext) in enumerate(graph.edges):
        if is_ext:
            continue

        key = (min(src, tgt), max(src, tgt))
        count = edge_counts.get(key, 0)
        edge_counts[key] = count + 1

        x1, y1 = positions[src] if src < n_v else (0, 0)
        x2, y2 = positions[tgt] if tgt < n_v else (0, 0)

        # Self-loop
        if src == tgt:
            x, y = positions[src]
            loop_dir = 'above' if y >= 0 else 'below'
            lines.append(f'  \\draw[->, thick, blue!70!black] '
                        f'({x:.2f},{y:.2f}) to[loop {loop_dir}, '
                        f'looseness=8, min distance=15pt] '
                        f'({x:.2f},{y:.2f});')
        else:
            # Bend for multiple edges between same vertices
            bend = count * 25 if count > 0 else 0
            bend_str = f', bend left={bend}' if bend > 0 else ''

            alpha_label = ''
            if show_labels and alpha_idx < len(graph._alpha_syms):
                alpha_label = str(graph._alpha_syms[alpha_idx])

            label_pos = 'above' if count % 2 == 0 else 'below'

            lines.append(f'  \\draw[->, thick, blue!70!black{bend_str}] '
                        f'({x1:.2f},{y1:.2f}) to'
                        f' node[{label_pos}, font=\\footnotesize] '
                        f'{{${alpha_label}$}} '
                        f'({x2:.2f},{y2:.2f});')

        alpha_idx += 1

    # Draw external legs
    ext_idx = 0
    for i, (src, tgt, is_ext) in enumerate(graph.edges):
        if not is_ext:
            continue

        # Determine which vertex is internal
        if src < n_v:
            vx, vy = positions[src]
            # Outgoing external: arrow from vertex outward
            angle = _external_angle(positions, src, ext_idx, graph.n_external_edges)
            ex = vx + 0.9 * math.cos(angle)
            ey = vy + 0.9 * math.sin(angle)
            lines.append(f'  \\draw[->, thin, dashed, gray] '
                        f'({vx:.2f},{vy:.2f}) -- ({ex:.2f},{ey:.2f});')
        elif tgt < n_v:
            vx, vy = positions[tgt]
            # Incoming external: arrow from outside to vertex
            angle = _external_angle(positions, tgt, ext_idx, graph.n_external_edges)
            ex = vx + 0.9 * math.cos(angle)
            ey = vy + 0.9 * math.sin(angle)
            lines.append(f'  \\draw[<-, thin, dashed, gray] '
                        f'({vx:.2f},{vy:.2f}) -- ({ex:.2f},{ey:.2f});')
        ext_idx += 1

    # Title/caption
    if title:
        y_min = min(y for _, y in positions.values()) - 1.0
        lines.append(f'  \\node at (0,{y_min:.2f}) {{\\small {title}}};')

    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


def one_loop_bubble_tikz(label: str = '', alpha0: str = r'\alpha_0',
                          alpha1: str = r'\alpha_1',
                          scale: float = 1.0) -> str:
    """
    The canonical one-loop self-energy bubble diagram in thesis style.

    Two vertices connected by two propagators forming a loop,
    with two external legs. This is the diagram from thesis Eq. (3.24).
    """
    lines = [f'% One-loop self-energy bubble (thesis style)']
    lines.append(r'\begin{tikzpicture}[>=Stealth, scale=' + f'{scale}]')

    # Two vertices
    lines.append(r'  \fill (-1.2, 0) circle (3pt) node[below=5pt] {$v_0$};')
    lines.append(r'  \fill ( 1.2, 0) circle (3pt) node[below=5pt] {$v_1$};')

    # Two internal edges forming a loop (bend above and below)
    lines.append(r'  \draw[->, thick, blue!70!black, bend left=40] '
                r'(-1.2,0) to node[above, font=\footnotesize] '
                f'{{${alpha0}$}} (1.2,0);')
    lines.append(r'  \draw[->, thick, blue!70!black, bend left=40] '
                r'(1.2,0) to node[below, font=\footnotesize] '
                f'{{${alpha1}$}} (-1.2,0);')

    # External legs (dashed)
    lines.append(r'  \draw[<-, thin, dashed, gray] (-1.2,0) -- (-2.2,0) '
                r'node[left] {$q_0$};')
    lines.append(r'  \draw[->, thin, dashed, gray] (1.2,0) -- (2.2,0) '
                r'node[right] {$q_1$};')

    # Info line
    if label:
        lines.append(f'  \\node at (0,-1.0) {{\\small {label}}};')

    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


def one_loop_circle_tikz(coupling: str = r'\lambda',
                          symmetry: str = '2',
                          scale: float = 1.0) -> str:
    """
    The one-loop diagram as a circle with external legs, matching
    thesis Eq. (3.24-3.25) style.
    """
    lines = [r'% One-loop circle diagram (thesis Eq. 3.24 style)']
    lines.append(r'\begin{tikzpicture}[>=Stealth, scale=' + f'{scale}]')

    # Circle
    lines.append(r'  \draw[thick, blue!70!black] (0,0) circle (0.7);')

    # Vertices on the circle (top and bottom, or left and right)
    lines.append(r'  \fill (0, 0.7) circle (3pt);')
    lines.append(r'  \fill (0,-0.7) circle (3pt);')

    # Arrow decoration on the circle (clockwise)
    lines.append(r'  \draw[->, thick, blue!70!black] '
                r'(0.05, 0.7) arc (85:-85:0.7);')
    lines.append(r'  \draw[->, thick, blue!70!black] '
                r'(-0.05,-0.7) arc (265:95:0.7);')

    # External legs
    lines.append(r'  \draw[<-, thin, dashed, gray] (0,-0.7) -- (0,-1.5) '
                r'node[below] {};')
    lines.append(r'  \draw[->, thin, dashed, gray] (0, 0.7) -- (0, 1.5) '
                r'node[above] {};')

    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


def corollas_grid_tikz(vertices: Dict[tuple, sp.Expr]) -> str:
    """
    Generate a grid of all corollas for a theory, matching thesis Figure 3.3.
    """
    lines = [r'% Corollas (amputated vertices) — cf. thesis Figure 3.3']
    lines.append(r'\begin{tikzpicture}[>=Stealth, scale=0.8]')

    items = [(k, v) for k, v in vertices.items() if v != 0]
    n = len(items)
    cols = min(n, 4)

    for i, ((m, nn), g) in enumerate(items):
        col = i % cols
        row = i // cols
        x_off = col * 2.5
        y_off = -row * 2.0

        # Central vertex
        lines.append(f'  \\fill ({x_off:.1f},{y_off:.1f}) circle (2.5pt);')

        total = m + nn
        if total == 0:
            continue

        angle_step = 180.0 / max(total - 1, 1) if total > 1 else 0
        leg_len = 0.6

        idx = 0
        # Outgoing (φ̃)
        for j in range(m):
            if total == 1:
                angle = 0
            else:
                angle = 180 - idx * angle_step
            rad = math.radians(angle)
            ex = x_off + leg_len * math.cos(rad)
            ey = y_off + leg_len * math.sin(rad)
            lines.append(f'  \\draw[->, thick] ({x_off:.1f},{y_off:.1f}) '
                        f'-- ({ex:.2f},{ey:.2f});')
            idx += 1

        # Incoming (φ)
        for j in range(nn):
            if total == 1:
                angle = 180
            else:
                angle = 180 - idx * angle_step
            rad = math.radians(angle)
            ex = x_off + leg_len * math.cos(rad)
            ey = y_off + leg_len * math.sin(rad)
            lines.append(f'  \\draw[<-, thick] ({x_off:.1f},{y_off:.1f}) '
                        f'-- ({ex:.2f},{ey:.2f});')
            idx += 1

        # Label below
        lines.append(f'  \\node[below=8pt, font=\\scriptsize] at '
                    f'({x_off:.1f},{y_off - 0.2:.1f}) '
                    f'{{$\\tilde{{\\phi}}^{m}\\phi^{nn}$: ${sp.latex(g)}$}};')

    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Helper functions                                                    #
# ------------------------------------------------------------------ #

def _vertex_positions(n_v: int, scale: float = 1.0) -> Dict[int, Tuple[float, float]]:
    """Compute vertex positions for drawing."""
    if n_v == 1:
        return {0: (0.0, 0.0)}
    elif n_v == 2:
        return {0: (-1.2 * scale, 0.0), 1: (1.2 * scale, 0.0)}
    elif n_v == 3:
        # Triangle
        return {
            0: (0.0, 1.0 * scale),
            1: (-1.0 * scale, -0.5 * scale),
            2: (1.0 * scale, -0.5 * scale),
        }
    else:
        # Arrange on circle
        positions = {}
        for i in range(n_v):
            angle = 2 * math.pi * i / n_v + math.pi / 2
            positions[i] = (1.2 * scale * math.cos(angle),
                          1.2 * scale * math.sin(angle))
        return positions


def _external_angle(positions: Dict[int, Tuple[float, float]],
                     vertex: int, ext_idx: int, n_ext: int) -> float:
    """Compute angle for an external leg, pointing away from graph centre."""
    vx, vy = positions[vertex]
    # Centre of all vertices
    cx = sum(x for x, y in positions.values()) / len(positions)
    cy = sum(y for x, y in positions.values()) / len(positions)
    # Point away from centre
    dx, dy = vx - cx, vy - cy
    base_angle = math.atan2(dy, dx)
    # Spread multiple external legs
    spread = 0.4 * (ext_idx - n_ext / 2)
    return base_angle + spread
