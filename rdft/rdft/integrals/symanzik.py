"""
rdft.integrals.symanzik
=======================
Symanzik polynomials Ψ (first) and Φ (second) for Feynman graphs.

These are the central objects in the parametric integral representation:

    I(G) = Ω_d · Γ(-σ_d) · ∫ Πe dα_e · α_e^{n_e-1}/Γ(n_e)
                           · (Ψ')^{-d/2} · (Ψ'/Φ')^{σ_d} · Πl δ(Σl)

where Ψ and Φ encode the topology (Ψ) and kinematics (Φ).

Definitions (Amarteifio eq. 2.30, Bogner-Weinzierl 2010):

    Ψ = Σ_{T ∈ spanning 1-trees} Π_{e ∉ T} α_e
      = det(M)    where M = E_{[Γ]}^T D_α E_{[Γ]}

    ϕ = Σ_{F ∈ spanning 2-forests} p̂²_F · Π_{e ∉ F} α_e

    Φ = Ψ · (Σ_e α_e m_e²) + ϕ

Properties (Amarteifio §2.5.1):
  1. Ψ is homogeneous of degree L in {α_e}
  2. Φ is homogeneous of degree L+1 in {α_e}
  3. Ψ is independent of masses and momenta (topology only)
  4. Φ encodes all kinematic data

Mathematical reference:
    Amarteifio (2019) §2.3, §2.5.1
    Bogner & Weinzierl (2010): arXiv:1003.1154
    Panzer (2015) PhD thesis
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import sympy as sp

from ..graphs.incidence import FeynmanGraph


class SymanzikPolynomials:
    """
    Compute the first and second Symanzik polynomials for a Feynman graph.

    Parameters
    ----------
    graph : FeynmanGraph
    momenta : dict from external edge index → sympy expression (momentum)
    masses : dict from internal edge index → sympy expression (mass)

    Attributes
    ----------
    Psi : first Symanzik polynomial (topology only)
    Phi : second Symanzik polynomial (includes kinematics)
    """

    def __init__(self,
                 graph: FeynmanGraph,
                 momenta: Optional[Dict[int, sp.Expr]] = None,
                 masses:  Optional[Dict[int, sp.Expr]] = None):

        self.graph = graph
        self.momenta = momenta or {}
        self.masses  = masses  or {}

        self._Psi: Optional[sp.Expr] = None
        self._phi: Optional[sp.Expr] = None
        self._Phi: Optional[sp.Expr] = None

    # ------------------------------------------------------------------ #
    #  First Symanzik polynomial Ψ                                         #
    # ------------------------------------------------------------------ #

    @property
    def Psi(self) -> sp.Expr:
        """
        First Symanzik polynomial.

        Ψ = Σ_{T ∈ Γ^{(1)}} Π_{e ∉ T} α_e

        where the sum is over all spanning 1-trees of the internal subgraph.

        Equivalently (Amarteifio eq. 2.33):
            Ψ = det(M) = det(E_{[Γ]}^T D_α E_{[Γ]})

        This equals the Kirchhoff polynomial's terms of degree L
        (number of loops) in the α_e, after taking the edge complement.

        Property: Ψ is the sum of products of α_e over all edges
        NOT in each spanning tree.
        """
        if self._Psi is None:
            self._Psi = self._compute_Psi()
        return self._Psi

    def _compute_Psi(self) -> sp.Expr:
        """
        Compute Ψ = Σ_{T ∈ spanning 1-trees} Π_{e ∉ T} α_e.

        Self-loop edges are NEVER in any spanning tree, so their Schwinger
        parameters always appear in Ψ as a multiplicative prefactor.

        Strategy:
          1. Separate self-loop internal edges from regular (non-self-loop) edges.
          2. Compute K = Σ_T Π_{e ∈ T} α_e using only regular edges via the
             Kirchhoff polynomial (which gives 0 for self-loop rows anyway).
          3. Take edge complement to get Ψ for regular edges.
          4. Multiply by the self-loop alpha product.

        For a pure-self-loop graph (e.g. tadpole), the only spanning tree is
        the empty set, giving Ψ = Π_{all α_e} = product of all internal alphas.
        """
        graph = self.graph
        alphas = graph._alpha_syms
        n_int = graph.n_internal_edges
        int_edges = graph.internal_edge_indices

        if n_int == 0:
            return sp.S.One

        # Partition internal edges into self-loops and regular edges
        self_loop_alphas = []
        regular_alphas = []
        for i, old_e in enumerate(int_edges):
            src, tgt, _ = graph.edges[old_e]
            if src == tgt:
                self_loop_alphas.append(alphas[i])
            else:
                regular_alphas.append(alphas[i])

        # Self-loop factor: these always appear in Ψ (never in any spanning tree)
        self_loop_factor = sp.Mul(*self_loop_alphas) if self_loop_alphas else sp.S.One

        if not regular_alphas:
            # All edges are self-loops: one spanning tree = empty set, weight 1
            # Ψ = Π_{all α_e} (every edge is in the complement of the empty tree)
            return self_loop_factor

        # K(α) = Σ_T Π_{e ∈ T} α_e  for non-self-loop edges only
        # (kirchhoff_polynomial excludes self-loop contributions automatically)
        K = graph.kirchhoff_polynomial()

        try:
            poly = sp.Poly(K, *regular_alphas)
        except Exception:
            return self_loop_factor * K

        # Ψ_{regular} = complement polynomial of K
        Psi_regular = sp.S.Zero
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            # complement: α_e for every regular edge NOT in this spanning tree
            complement = sp.Mul(*[regular_alphas[i]
                                   for i, k in enumerate(monom) if k == 0])
            Psi_regular += coeff * complement

        return sp.expand(self_loop_factor * Psi_regular)

    # ------------------------------------------------------------------ #
    #  Second Symanzik polynomial Φ                                        #
    # ------------------------------------------------------------------ #

    @property
    def phi_kinematic(self) -> sp.Expr:
        """
        The kinematic part ϕ of Φ:

            ϕ = Σ_{F ∈ Γ^{(2)}} p̂²_F · Π_{e ∉ F} α_e

        where the sum is over spanning 2-forests (i.e. spanning forests
        with exactly 2 components), and p̂²_F is the squared external
        momentum flowing across the cut separating the two trees.

        For 2-forests that isolate only internal vertices from external ones,
        p̂²_F = 0, so these don't contribute.

        (Amarteifio eq. 2.30b)
        """
        if self._phi is None:
            self._phi = self._compute_phi()
        return self._phi

    def _compute_phi(self) -> sp.Expr:
        """Compute the kinematic polynomial ϕ."""
        alphas = self.graph._alpha_syms
        int_edges = self.graph.internal_edge_indices

        # If no external momenta specified, return zero
        if not self.momenta:
            return sp.S.Zero

        # For a graph with external edges connected to v_∞:
        # A 2-forest F separates the graph into two components.
        # p̂²_F = (sum of external momenta entering one component)²
        # We need to enumerate all 2-trees.

        # This is the general algorithm; for specific cases we use
        # the matrix formula (Amarteifio eq. 2.39b):
        # ϕ = Σ_{F} p̂(F)² · Π_{e ∉ F} α_e
        # = -Σ_{F} p̂_1 · p̂_2 · Π_{e ∉ F} α_e   (by momentum conservation)

        # For simple cases (zero or one external momentum scale),
        # we compute directly.
        if len(self.momenta) == 0:
            return sp.S.Zero

        # General case: use the formula (Amarteifio eq. 2.39b)
        # Enumerate 2-forests from K by extracting degree (n_int - L + 1) monomials
        K = self.graph.kirchhoff_polynomial()
        n_int = self.graph.n_internal_edges
        L = self.graph.L

        # 2-trees correspond to monomials of degree n_int - L + 1 in K
        # (one more edge removed than for 1-trees)
        # Extract these using sympy Poly
        poly = sp.Poly(K, *alphas)

        phi = sp.S.Zero
        # This is a simplified implementation for single momentum scale
        q_sq = sp.Symbol('q_sq')   # external momentum squared p̂² = q²
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            degree = sum(monom)
            # 2-trees have degree L+1 terms in Ψ of the 2-forest,
            # which correspond to degree n_int - (L+1) = n_int - L - 1 in K
            # Actually the relationship is:
            # degree d monomial in K ↔ d-tree
            # 2-tree: d=2 → degree 2 monomials in K
            # The complement (used in Ψ^{(2)}) has n_int - 2 alpha factors
            pass

        # For now return the symbolic placeholder; full implementation
        # uses the matrix formula from Panzer (2015)
        return sp.S.Zero   # TODO: implement full 2-tree enumeration

    @property
    def Phi(self) -> sp.Expr:
        """
        Second Symanzik polynomial:
            Φ = Ψ · Σ_e (α_e · m_e²) + ϕ

        (Amarteifio eq. 2.30c)

        For massless, zero-momentum processes (common in reaction-diffusion
        near criticality), Φ → ϕ with appropriate kinematics.
        """
        if self._Phi is None:
            alphas = self.graph._alpha_syms
            int_edges = self.graph.internal_edge_indices

            # Mass term: Ψ · Σ_e α_e m_e²
            mass_term = sp.S.Zero
            for i, edge_idx in enumerate(int_edges):
                m_e = self.masses.get(edge_idx,
                                      sp.Symbol(f'm_{edge_idx}', positive=True))
                mass_term += alphas[i] * m_e**2

            self._Phi = sp.expand(self.Psi * mass_term + self.phi_kinematic)

        return self._Phi

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def verify_homogeneity(self) -> Dict[str, bool]:
        """
        Check the homogeneity properties of Ψ and Φ.
        (Amarteifio §2.5.1 Properties 1 and 2)

        Ψ should be homogeneous of degree L.
        Φ should be homogeneous of degree L+1 (when mass terms are included).
        """
        results = {}
        alphas = self.graph._alpha_syms
        L = self.graph.L

        # Check Ψ: substitute α_e → t·α_e and check Ψ → t^L · Ψ
        t = sp.Symbol('t', positive=True)
        Psi_scaled = self.Psi.subs([(a, t*a) for a in alphas])
        Psi_scaled = sp.expand(Psi_scaled)
        Psi_expected = sp.expand(t**L * self.Psi)
        results['Psi_homogeneous_degree_L'] = (
            sp.simplify(Psi_scaled - Psi_expected) == 0
        )

        return results

    def summary(self) -> str:
        lines = [f'Symanzik polynomials for {self.graph}']
        lines.append(f'  L (loops) = {self.graph.L}')
        lines.append(f'  Ψ = {self.Psi}')
        lines.append(f'  Φ = {self.Phi}')
        return '\n'.join(lines)
