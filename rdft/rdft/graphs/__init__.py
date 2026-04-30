"""
rdft.graphs — Feynman graphs, corollas, spectral dimension, rendering.

Pillar 1 (N-loop diagram processing) and Pillar 3 (visualisation).
"""
from .incidence import FeynmanGraph
from .corolla import HalfEdge, Corolla, corollas_from_liouvillian
from .enumerate import enumerate_diagrams, diagrams_for_process, summary
from .render import to_dot, render_svg, render_ascii, render_all
from .tikz import (
    corolla_to_tikz,
    diagram_to_tikz,
    one_loop_bubble_tikz,
    one_loop_circle_tikz,
    corollas_grid_tikz,
    multispecies_corolla_tikz,
    brw_corollas_figure_33,
)
from .spectral import (
    SpectralDimension,
    substitute_spectral_dimension,
    hypercubic_lattice,
    sierpinski_carpet,
    random_tree,
    preferential_attachment,
    brw_scaling_exponents,
)

__all__ = [
    # incidence-matrix graphs
    'FeynmanGraph',
    # corollas
    'HalfEdge', 'Corolla', 'corollas_from_liouvillian',
    # diagram enumeration
    'enumerate_diagrams', 'diagrams_for_process', 'summary',
    # rendering
    'to_dot', 'render_svg', 'render_ascii', 'render_all',
    # TikZ
    'corolla_to_tikz', 'diagram_to_tikz',
    'one_loop_bubble_tikz', 'one_loop_circle_tikz',
    'corollas_grid_tikz', 'multispecies_corolla_tikz',
    'brw_corollas_figure_33',
    # spectral / graph generators
    'SpectralDimension', 'substitute_spectral_dimension',
    'hypercubic_lattice', 'sierpinski_carpet',
    'random_tree', 'preferential_attachment',
    'brw_scaling_exponents',
]
