"""
rdft.graphs.corolla
===================
Primitive corollas from Liouvillian vertices.

A corolla is a single-vertex Feynman graph with only external half-edges.
Each interaction vertex in the Doi-Peliti action produces one corolla type.

Reference: Amarteifio (2019) §2.5, Definition 11, Figure 3.3
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sympy as sp


@dataclass(frozen=True)
class HalfEdge:
    """A half-edge of a corolla.

    Parameters
    ----------
    vertex : int
        Which vertex instance this half-edge belongs to.
    index : int
        Unique index within the diagram's half-edge list.
    edge_type : str
        ``'out'`` for outgoing (phi-tilde) or ``'in'`` for incoming (phi).
    species : str
        Particle species label (default ``'A'``).
    """
    vertex: int
    index: int
    edge_type: str       # 'out' (phi-tilde) or 'in' (phi)
    species: str = 'A'


@dataclass
class Corolla:
    """
    A primitive corolla: a single vertex with half-edges.

    In the Doi-Peliti action each monomial  g * phi_tilde^m * phi^n
    corresponds to a vertex with *m* outgoing phi-tilde legs and *n*
    incoming phi legs.  The corolla is the star graph that represents
    this vertex before any propagator contractions.

    Parameters
    ----------
    vertex_type : tuple (m_out, n_in)
        Monomial type — number of outgoing and incoming half-edges.
    coupling : sympy.Expr
        Coupling constant (coefficient of the monomial).
    species : str
        Particle species label (default ``'A'``).
    """
    vertex_type: Tuple[int, int]
    coupling: sp.Expr
    species: str = 'A'

    @property
    def n_out(self) -> int:
        """Number of outgoing (phi-tilde) half-edges."""
        return self.vertex_type[0]

    @property
    def n_in(self) -> int:
        """Number of incoming (phi) half-edges."""
        return self.vertex_type[1]

    @property
    def n_legs(self) -> int:
        """Total number of half-edges."""
        return self.n_out + self.n_in

    def half_edges(self, vertex_id: int = 0) -> List[HalfEdge]:
        """Generate the half-edges for this corolla at a given vertex ID.

        Outgoing half-edges are listed first, then incoming.

        Parameters
        ----------
        vertex_id : int
            Label assigned to the vertex in the diagram being assembled.

        Returns
        -------
        list of HalfEdge
        """
        edges: List[HalfEdge] = []
        idx = 0
        for _ in range(self.n_out):
            edges.append(HalfEdge(vertex_id, idx, 'out', self.species))
            idx += 1
        for _ in range(self.n_in):
            edges.append(HalfEdge(vertex_id, idx, 'in', self.species))
            idx += 1
        return edges

    def __repr__(self) -> str:
        return (f'Corolla(\u03c6\u0303^{self.n_out}'
                f'\u03c6^{self.n_in}, g={self.coupling})')


# ------------------------------------------------------------------ #
#  Factory                                                              #
# ------------------------------------------------------------------ #

def corollas_from_liouvillian(liouvillian,
                              min_legs: int = 3) -> List[Corolla]:
    """
    Extract primitive corollas from a Liouvillian's vertices.

    Only vertices whose total leg count (m_out + n_in) is at least
    *min_legs* are kept.  By default ``min_legs=3`` drops the
    propagator correction ``(1,1)`` and source/sink terms ``(0,1)``,
    ``(1,0)`` which do not generate proper interaction vertices for
    diagram construction.

    Set ``min_legs=0`` to keep everything (useful for debugging).

    Parameters
    ----------
    liouvillian : Liouvillian
        Object from ``rdft.core.generators`` whose ``.vertices`` dict
        maps ``(m_out, n_in) -> coupling``.
    min_legs : int
        Minimum total half-edge count to include a vertex.

    Returns
    -------
    list of Corolla
    """
    corollas: List[Corolla] = []
    for (m_out, n_in), coupling in liouvillian.vertices.items():
        if m_out + n_in < min_legs:
            continue
        corollas.append(Corolla(
            vertex_type=(m_out, n_in),
            coupling=coupling,
        ))
    return corollas
