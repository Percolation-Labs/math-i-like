"""
rdft.crn.crn
============

``CRN`` is the entry point: a chemical reaction network plus the Doi shift that
produces the vertex dictionary and the polynomial ``phi(G)`` driving Lagrange
inversion.

The Doi shift turns each reaction ``sum_i k_ri A_i -> sum_i l_ri A_i`` at rate
``k_r`` into the closed-form vertex generator

    Q_r = k_r * [ prod_i (1 + psit_i)^{l_ri}
                  - prod_i (1 + psit_i)^{k_ri} ] * prod_i psi_i^{k_ri}.

Expanding the binomials produces a finite vertex dictionary: each monomial
``g * prod_i psit_i^{m_i} psi_i^{n_i}`` is a Doi-Peliti vertex with ``m_i``
out-legs and ``n_i`` in-legs of species ``A_i``, with rational coefficient ``g``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import sympy as sp


@dataclass(frozen=True)
class Reaction:
    """A single reaction: sum_i k_ri A_i -> sum_i l_ri A_i at rate k_r.

    ``reactants`` and ``products`` are dicts {species_name: stoichiometry}.
    """
    reactants: Tuple[Tuple[str, int], ...]
    products:  Tuple[Tuple[str, int], ...]
    rate:      sp.Expr

    def reactant_dict(self) -> Dict[str, int]:
        return dict(self.reactants)

    def product_dict(self) -> Dict[str, int]:
        return dict(self.products)


@dataclass(frozen=True)
class Vertex:
    """A Doi-Peliti vertex characterised by leg counts of each species.

    Convention: ``in`` legs are psit (annihilation), ``out`` legs are psi
    (creation). The coupling carries a sign that records the original action's
    sign (+1 if the action term is +g*..., -1 if -g*...).
    """
    name: str
    in_legs:  Tuple[Tuple[str, int], ...]   # ((species, count), ...)
    out_legs: Tuple[Tuple[str, int], ...]
    sign: int = 1
    coupling: Optional[sp.Expr] = None

    def in_dict(self) -> Dict[str, int]:
        return dict(self.in_legs)

    def out_dict(self) -> Dict[str, int]:
        return dict(self.out_legs)

    def n_legs(self) -> int:
        return sum(c for _, c in self.in_legs) + sum(c for _, c in self.out_legs)

    @property
    def n_in(self) -> int:
        return sum(c for _, c in self.in_legs)

    @property
    def n_out(self) -> int:
        return sum(c for _, c in self.out_legs)


@dataclass
class CRN:
    """A chemical reaction network with its derived vertex dictionary."""
    name: str
    species: Tuple[str, ...]
    reactions: Tuple[Reaction, ...]
    vertices: Tuple[Vertex, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Doi shift: reactions -> vertex dictionary
    # ------------------------------------------------------------------
    def doi_shift(self) -> Tuple[Vertex, ...]:
        """Compute the Doi-shifted vertex dictionary.

        For each reaction, expand
            Q_r = k_r [ prod_i (1+psit_i)^l_ri - prod_i (1+psit_i)^k_ri ]
                       * prod_i psi_i^k_ri
        symbolically, then read each monomial as a vertex.
        """
        psi_syms  = {s: sp.Symbol(f"psi_{s}")  for s in self.species}
        psit_syms = {s: sp.Symbol(f"psit_{s}") for s in self.species}

        vertices: List[Vertex] = []
        seen_names: Dict[Tuple, str] = {}
        idx = 0
        for r in self.reactions:
            k_r = r.rate
            kdict = r.reactant_dict()
            ldict = r.product_dict()

            shift_prod = sp.Integer(1)
            shift_react = sp.Integer(1)
            for sp_name in self.species:
                shift_prod *= (1 + psit_syms[sp_name])**ldict.get(sp_name, 0)
                shift_react *= (1 + psit_syms[sp_name])**kdict.get(sp_name, 0)
            psi_factor = sp.Integer(1)
            for sp_name in self.species:
                psi_factor *= psi_syms[sp_name]**kdict.get(sp_name, 0)

            Q_r = sp.expand(k_r * (shift_prod - shift_react) * psi_factor)

            # Each monomial of Q_r is a vertex
            poly = sp.Poly(Q_r, *(list(psi_syms.values()) + list(psit_syms.values())))
            psi_order = list(psi_syms.values())
            psit_order = list(psit_syms.values())
            for monom, coef in poly.terms():
                # monom indexed by (psi_a, psi_b, ..., psit_a, psit_b, ...)
                if coef == 0:
                    continue
                psi_counts = tuple((s, monom[i]) for i, s in enumerate(self.species)
                                   if monom[i] > 0)
                psit_counts = tuple((s, monom[len(self.species) + i])
                                    for i, s in enumerate(self.species)
                                    if monom[len(self.species) + i] > 0)
                # Free part (kinetic): skip terms with no psi at all
                if all(monom[i] == 0 for i in range(len(self.species))) and \
                   all(monom[len(self.species) + i] == 0 for i in range(len(self.species))):
                    continue
                # Determine sign
                try:
                    coef_simpl = sp.simplify(coef)
                    if coef_simpl.could_extract_minus_sign():
                        sign = -1
                    else:
                        sign = +1
                except Exception:
                    sign = +1

                key = (psit_counts, psi_counts, sign)
                if key in seen_names:
                    continue
                vname = f"V{idx}"
                idx += 1
                seen_names[key] = vname
                vertices.append(Vertex(
                    name=vname,
                    in_legs=psit_counts,
                    out_legs=psi_counts,
                    sign=sign,
                    coupling=coef,
                ))

        return tuple(vertices)

    def with_doi_vertices(self) -> "CRN":
        """Return self with vertex dictionary populated from Doi shift."""
        if self.vertices:
            return self
        v = self.doi_shift()
        return CRN(name=self.name, species=self.species,
                   reactions=self.reactions, vertices=v)

    # ------------------------------------------------------------------
    # phi(G): the cubic-node generating polynomial for the DSE
    # ------------------------------------------------------------------
    def interaction_vertices(self, max_legs: int = 3) -> Tuple[Vertex, ...]:
        """Vertices with at least 3 legs and at most ``max_legs`` legs.

        ``n_legs == 2`` vertices are kinetic / mass terms and are absorbed into
        the propagator, not the interaction. Higher-leg vertices may be
        irrelevant by power counting (at d_c=4, n_legs > 3 is irrelevant for
        cubic theories).
        """
        return tuple(v for v in self.vertices if 3 <= v.n_legs() <= max_legs)

    def phi_polynomial(self, G: sp.Symbol = None, max_legs: int = 3) -> sp.Expr:
        """Read off ``phi(G)`` from the vertex dictionary.

        Each interaction vertex with ``n_legs`` legs contributes ``G^{n_legs-1}``
        (one parent edge, ``n_legs-1`` children). Plus a constant 1 for "leaf".

        For Reggeon DP (cubic only): ``phi(G) = 1 + G^2``.
        For phi^4-style annihilation 2A->0 with quartic kept: ``phi = 1+G^2+G^3``.
        """
        if G is None:
            G = sp.Symbol("G")
        if not self.vertices:
            crn = self.with_doi_vertices()
            return crn.phi_polynomial(G, max_legs=max_legs)
        terms = sp.Integer(1)  # the leaf
        children_counts = set()
        for v in self.interaction_vertices(max_legs=max_legs):
            children_counts.add(v.n_legs() - 1)
        for k in sorted(children_counts):
            terms += G**k
        return terms

    # ------------------------------------------------------------------
    # Builders for the standard test CRNs
    # ------------------------------------------------------------------
    @staticmethod
    def reggeon_dp(name: str = "Reggeon DP") -> "CRN":
        """The Janssen-Tauber Reggeon directed-percolation CRN.

        Reactions:
            A -> 2A   at rate lambda*g
            2A -> A   at rate lambda*g
        """
        lam, g = sp.symbols("lambda g", positive=True)
        rate = lam * g
        R1 = Reaction(reactants=(("A", 1),), products=(("A", 2),), rate=rate)
        R2 = Reaction(reactants=(("A", 2),), products=(("A", 1),), rate=rate)
        crn = CRN(name=name, species=("A",), reactions=(R1, R2))
        return crn.with_doi_vertices()

    @staticmethod
    def dyadic_brw(name: str = "Dyadic BRW") -> "CRN":
        """Pure dyadic branching random walk: A -> 2A only.

        This is the BRW kernel from the thesis Eq. 3.16, without the
        Lambda-trace observable extension of Eq. 3.21.
        """
        beta = sp.Symbol("beta", positive=True)
        R = Reaction(reactants=(("A", 1),), products=(("A", 2),), rate=beta)
        crn = CRN(name=name, species=("A",), reactions=(R,))
        return crn.with_doi_vertices()

    @staticmethod
    def phi4_doi_peliti(name: str = "phi^4 DP") -> "CRN":
        """The phi^4-style Doi-Peliti CRN: A + A -> 0 (annihilation).

        Reactions:
            2A -> 0  at rate lambda
            (no creation; quartic vertex from the Doi shift on 2A)
        """
        lam = sp.Symbol("lambda", positive=True)
        R = Reaction(reactants=(("A", 2),), products=(), rate=lam)
        crn = CRN(name=name, species=("A",), reactions=(R,))
        return crn.with_doi_vertices()

    @staticmethod
    def from_reactions(name: str, species: Tuple[str, ...],
                       reactions: Tuple[Reaction, ...]) -> "CRN":
        """Build a CRN from an explicit list of reactions."""
        crn = CRN(name=name, species=species, reactions=reactions)
        return crn.with_doi_vertices()

    @staticmethod
    def from_vertices(name: str, species: Tuple[str, ...],
                      vertices: Tuple[Vertex, ...]) -> "CRN":
        """Build a CRN by specifying its vertices directly (no reactions).

        Useful when the vertex set comes from a derivation (e.g.\ thesis
        Eq.~3.21's Lambda-trace observable for BRW) that does not reduce
        cleanly to a small list of stoichiometries.
        """
        return CRN(name=name, species=species, reactions=(), vertices=vertices)

    @staticmethod
    def brw_thesis(name: str = "BRW (thesis Eqs. 3.16+3.21)") -> "CRN":
        """The 7-vertex BRW set from Amarteifio (2019) PhD thesis,
        Eqs.~(3.16)+(3.21). One pure-branching vertex plus six Lambda-trace
        observable vertices that couple species A to species B.

        This vertex set is specified directly because Eq. 3.21 derives the
        trace-observable couplings from a continuous-q expansion that does not
        reduce to elementary stoichiometric reactions.
        """
        # Sign convention from poc_brw.py / thesis.
        V = lambda nm, ia, ib, oa, ob, sg: Vertex(
            name=nm, in_legs=tuple([("A", ia)] if ia else []) + tuple([("B", ib)] if ib else []),
            out_legs=tuple([("A", oa)] if oa else []) + tuple([("B", ob)] if ob else []),
            sign=sg,
        )
        vertices = (
            V("V_branch", 1, 0, 2, 0, -1),    # -beta psit_a psi_a^2 (Eq. 3.16)
            V("V_QC1",    0, 1, 1, 0, +1),    # +Lambda psit_b psi_a (bilinear, Eq. 3.21)
            V("V_QC2",    0, 1, 1, 1, -1),    # -Lambda psit_b psi_b psi_a
            V("V_QC3",    0, 2, 1, 1, -1),    # -Lambda psit_b^2 psi_b psi_a
            V("V_QC4",    1, 2, 1, 1, -1),    # -Lambda psit_b^2 psit_a psi_b psi_a
            V("V_QC5",    1, 1, 1, 1, -1),    # -Lambda psit_b psit_a psi_b psi_a
            V("V_QC6",    0, 1, 2, 0, +1),    # +Lambda psit_b psi_a^2
        )
        return CRN.from_vertices(name=name, species=("A", "B"), vertices=vertices)
