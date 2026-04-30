"""
RDFT: Reaction-Diffusion Field Theory
======================================

From stoichiometry to critical exponents via analytic combinatorics.

Any chemical reaction network → Liouvillian → DSE → Lagrange equation
→ singularity → transfer theorem → critical exponents.

The AC route is the direct route. The Feynman diagrams are the scenic route.
Both arrive at the same answer: singularity type = universality class.

Four pillars
------------

    1. N-loop diagram processing       — rdft.graphs, rdft.integrals, rdft.rg
    2. CRN → DSE mapping               — rdft.core, rdft.ac
    3. Document / visualisation        — rdft.graphs.tikz, rdft.graphs.render
    4. Lattice simulations on graphs   — rdft.simulate, rdft.graphs.spectral

Quick start::

    from rdft import analyze
    results = analyze('pair_annihilation')
    results = analyze('gribov')  # BRW process

    # Or build a CRN and drive the pipeline manually:
    from rdft import ReactionNetwork, Species, Reaction, Liouvillian
    from rdft import FeynmanGraph, Singularity
    from rdft import run_brw

References:
    Amarteifio (2019) PhD thesis, Imperial College London
    Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590
    Amarteifio (2026) AC-QFT Tutorial
"""

__version__ = '0.1.0'

# Pillar 2 core: the canonical entry points ---------------------------
from .pipeline import analyze, brw_worked_example
from .core.reaction_network import ReactionNetwork, Species, Reaction
from .core.generators import Liouvillian

# Pillar 4: lattice simulations ---------------------------------------
from .simulate import run_brw, run_ensemble

# Pillar 1: diagrams, integrals, RG — re-export primary classes -------
from .graphs.incidence import FeynmanGraph
from .graphs.spectral import SpectralDimension
from .integrals.symanzik import SymanzikPolynomials
from .integrals.parametric import ParametricIntegral

# Pillar 2 AC: Tier-1 DSE/transfer entry points -----------------------
from .ac.transfer import Singularity, from_lagrange
from .ac.lagrange import LagrangeEquation

__all__ = [
    # pipeline
    'analyze', 'brw_worked_example',
    # CRN primitives (pillar 2)
    'ReactionNetwork', 'Species', 'Reaction', 'Liouvillian',
    # lattice simulations (pillar 4)
    'run_brw', 'run_ensemble',
    # diagrams & integrals (pillar 1)
    'FeynmanGraph', 'SpectralDimension',
    'SymanzikPolynomials', 'ParametricIntegral',
    # AC Tier-1 (pillar 2)
    'Singularity', 'from_lagrange', 'LagrangeEquation',
]
