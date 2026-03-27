"""
rdft.graphs.render
==================
Feynman diagram rendering for RDFT.

Renders FeynmanGraph objects as visual diagrams using graphviz (DOT format)
and optionally matplotlib for simple inline display.

Each diagram shows:
  - Internal vertices (labelled by vertex type and coupling)
  - Internal edges (labelled by Schwinger parameter α_e)
  - External legs (connecting to v_∞, labelled by species/momentum)
  - Metadata: loop number L, symmetry factor s(G), σ_d(G)
"""

from __future__ import annotations
from typing import Optional, Dict, List
import sympy as sp

from .incidence import FeynmanGraph


def to_dot(graph: FeynmanGraph,
           title: str = '',
           vertex_labels: Optional[Dict[int, str]] = None,
           show_alphas: bool = True,
           show_external: bool = True) -> str:
    """
    Convert a FeynmanGraph to GraphViz DOT format.

    Parameters
    ----------
    graph : FeynmanGraph to render
    title : optional title for the diagram
    vertex_labels : dict mapping vertex index → label string
    show_alphas : show Schwinger parameter labels on edges
    show_external : show external legs and v_∞

    Returns
    -------
    DOT format string
    """
    lines = ['digraph FeynmanDiagram {']
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=circle, style=filled, fillcolor=lightblue, '
                 'fontsize=10, width=0.4];')
    lines.append('  edge [fontsize=8];')

    if title:
        lines.append(f'  label="{title}";')
        lines.append('  labelloc=t;')
        lines.append('  fontsize=12;')

    # Metadata as subtitle
    meta = f'L={graph.L}, E_int={graph.n_internal_edges}, σ_d={graph.degree_of_divergence()}'
    lines.append(f'  sublabel [shape=none, label="{meta}", fontsize=9, fontcolor=gray];')

    # Internal vertices
    for v in range(graph.n_vertices_int):
        label = vertex_labels.get(v, f'v{v}') if vertex_labels else f'v{v}'
        lines.append(f'  v{v} [label="{label}"];')

    # v_∞ (external boundary)
    if show_external:
        lines.append(f'  vinf [label="v∞", shape=doublecircle, '
                     f'fillcolor=lightyellow, width=0.3, fontsize=8];')

    # Edges
    alpha_idx = 0
    ext_idx = 0
    for i, (src, tgt, is_ext) in enumerate(graph.edges):
        src_name = f'v{src}' if src < graph.n_vertices_int else 'vinf'
        tgt_name = f'v{tgt}' if tgt < graph.n_vertices_int else 'vinf'

        if is_ext:
            if not show_external:
                continue
            label = f'q{ext_idx}'
            ext_idx += 1
            lines.append(f'  {src_name} -> {tgt_name} '
                        f'[label="{label}", style=dashed, color=gray];')
        else:
            if show_alphas and alpha_idx < len(graph._alpha_syms):
                label = str(graph._alpha_syms[alpha_idx])
            else:
                label = f'p{alpha_idx}'
            alpha_idx += 1

            # Self-loops
            if src == tgt:
                lines.append(f'  {src_name} -> {tgt_name} '
                            f'[label="{label}", color=blue];')
            else:
                lines.append(f'  {src_name} -> {tgt_name} '
                            f'[label="{label}", color=blue, penwidth=1.5];')

    lines.append('}')
    return '\n'.join(lines)


def render_svg(graph: FeynmanGraph, filename: str = None, **kwargs) -> Optional[str]:
    """
    Render a FeynmanGraph to SVG via graphviz.

    Parameters
    ----------
    graph : FeynmanGraph
    filename : output file path (without extension). If None, returns SVG string.
    **kwargs : passed to to_dot()

    Returns
    -------
    SVG string if filename is None, otherwise writes to file.
    """
    dot_str = to_dot(graph, **kwargs)

    try:
        import subprocess
        result = subprocess.run(
            ['dot', '-Tsvg'],
            input=dot_str.encode(),
            capture_output=True,
            timeout=10
        )
        svg = result.stdout.decode()

        if filename:
            with open(f'{filename}.svg', 'w') as f:
                f.write(svg)
            return filename + '.svg'
        return svg

    except FileNotFoundError:
        print("Warning: graphviz 'dot' not found. Install with: brew install graphviz")
        if filename:
            with open(f'{filename}.dot', 'w') as f:
                f.write(dot_str)
            return filename + '.dot'
        return dot_str


def render_ascii(graph: FeynmanGraph) -> str:
    """
    Simple ASCII representation of a Feynman graph.

    For quick terminal display without graphviz.
    """
    lines = [f'FeynmanGraph: V={graph.n_vertices_int}, '
             f'E_int={graph.n_internal_edges}, L={graph.L}']
    lines.append(f'  1PI: {graph.is_1pi()}, σ_d = {graph.degree_of_divergence()}')
    lines.append('')

    # Show edges
    alpha_idx = 0
    ext_idx = 0
    for src, tgt, is_ext in graph.edges:
        src_name = f'v{src}' if src < graph.n_vertices_int else 'v∞'
        tgt_name = f'v{tgt}' if tgt < graph.n_vertices_int else 'v∞'
        if is_ext:
            lines.append(f'  {src_name} --q{ext_idx}--> {tgt_name}  (external)')
            ext_idx += 1
        else:
            sym = graph._alpha_syms[alpha_idx] if alpha_idx < len(graph._alpha_syms) else f'α{alpha_idx}'
            lines.append(f'  {src_name} =={sym}==> {tgt_name}  (internal)')
            alpha_idx += 1

    # Kirchhoff polynomial
    try:
        K = graph.kirchhoff_polynomial()
        lines.append(f'\n  K(α) = {K}')
    except Exception:
        pass

    return '\n'.join(lines)


def render_all(diagrams: List[dict], prefix: str = 'diagram',
               format: str = 'ascii') -> List[str]:
    """
    Render a list of diagrams (from expansion or enumeration).

    Parameters
    ----------
    diagrams : list of dicts with 'graph' key (from FeynmanExpansion.expand())
    prefix : filename prefix for SVG output
    format : 'ascii', 'dot', or 'svg'

    Returns
    -------
    List of rendered strings (ascii/dot) or filenames (svg)
    """
    results = []
    for i, diag in enumerate(diagrams):
        graph = diag['graph']
        coupling = diag.get('coupling', '')
        sym = diag.get('symmetry_factor', '?')
        title = f'Diagram {i+1}: L={graph.L}, s(G)={sym}, c={coupling}'

        if format == 'ascii':
            results.append(render_ascii(graph))
        elif format == 'dot':
            results.append(to_dot(graph, title=title))
        elif format == 'svg':
            fname = render_svg(graph, filename=f'{prefix}_{i+1}', title=title)
            results.append(fname)

    return results
