"""
rdft.crn
========

A clean, auditable API for going from a chemical reaction network (CRN)
to its full RG programme. Replaces the scattered ``rdft/ac/gribov/*.py`` and
``letters/.../poc_*.py`` scripts with a unified surface.

Quick start
-----------

>>> from rdft.crn import CRN, RGProgram
>>> crn = CRN.reggeon_dp()
>>> rg = RGProgram(crn, loop_order=2)
>>> rg.run()
>>> rg.exponents.compare_to_jt05()    # zero residual
>>> rg.audit()                          # per-step provenance ledger

Module layout
-------------

* ``crn``         -- ``CRN``, ``Reaction``, ``Vertex``; Doi shift; ``phi(G)``.
* ``diagram``     -- ``Diagram``, ``Provenance``; per-graph data with lineage.
* ``enumerate``   -- plane-tree enumeration; bubble + tadpole + multi-species.
* ``symmetry``    -- ``aut_phi_tree`` (rule ``2^k(T)``); directed-graph ``Aut``.
* ``legendre``    -- symbolic ``Z -> W -> Gamma`` for any CRN.
* ``rg``          -- ``RGProgram``: Lagrange -> Hopf -> IBP -> Tauber -> exponents.
* ``audit``       -- walks the ``RGProgram`` history, prints per-step ledger.
* ``viz``         -- renders a ``Diagram`` to a PDF panel with annotations.
"""
from rdft.crn.crn import CRN, Reaction, Vertex
from rdft.crn.diagram import Diagram, Provenance

# RGProgram is imported lazily to avoid circular dependencies and to allow
# users to use the CRN/Diagram pieces without dragging in the full pipeline.

def __getattr__(name):
    if name == "RGProgram":
        from rdft.crn.rg import RGProgram
        return RGProgram
    raise AttributeError(f"module 'rdft.crn' has no attribute {name!r}")

__all__ = [
    "CRN", "Reaction", "Vertex",
    "Diagram", "Provenance",
    "RGProgram",
]
