"""
rdft.core — CRN foundations: species, reactions, Liouvillians, Feynman
expansion primitives.

Pillar 2 (CRN → DSE mapping).
"""
from .reaction_network import ReactionNetwork, Species, Reaction
from .generators import (
    Liouvillian,
    generator_ordinary,
    generator_factorial,
    generator_action_term,
    brw_transmutation_vertices,
    brw_all_vertices,
)
from .expansion import (
    Vertex,
    LegAssignment,
    FeynmanExpansion,
    extract_vertices,
    contraction_to_graph,
    classify_diagrams,
    are_isomorphic,
)

__all__ = [
    # reaction-network primitives
    'ReactionNetwork', 'Species', 'Reaction',
    # Liouvillian generators
    'Liouvillian',
    'generator_ordinary', 'generator_factorial', 'generator_action_term',
    'brw_transmutation_vertices', 'brw_all_vertices',
    # Feynman expansion primitives
    'Vertex', 'LegAssignment', 'FeynmanExpansion',
    'extract_vertices', 'contraction_to_graph',
    'classify_diagrams', 'are_isomorphic',
]
