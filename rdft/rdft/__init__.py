"""
RDFT: Reaction-Diffusion Field Theory
======================================

From stoichiometry to critical exponents via analytic combinatorics.

Any chemical reaction network → Liouvillian → DSE → Lagrange equation
→ singularity → transfer theorem → critical exponents.

The AC route is the direct route. The Feynman diagrams are the scenic route.
Both arrive at the same answer: singularity type = universality class.

Quick start::

    from rdft import analyze
    results = analyze('pair_annihilation')
    results = analyze('gribov')  # BRW process

References:
    Amarteifio (2019) PhD thesis, Imperial College London
    Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590
    Amarteifio (2026) AC-QFT Tutorial
"""

__version__ = '0.1.0'

from .pipeline import analyze, brw_worked_example
from .core.reaction_network import ReactionNetwork, Species, Reaction
from .core.generators import Liouvillian
