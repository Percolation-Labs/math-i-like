"""
rdft.integrals.parametric
=========================
Parametric integral representation and ω-integration.

The core integral is (Amarteifio eq. 2.76):

    I(G) = Ω_d · Γ(-σ_d(G')) · ∫ Π_e [α_e^{n_e-1}/Γ(n_e)] dα_e
                               · [1/(Ψ')^{d/2}] · [Ψ'/Φ']^{σ_d(G')} · Π_l δ(Σ_l)

where:
  - Ω_d = 2π^{d/2} / (2π)^d = 1/(4π)^{d/2} is the angular factor
  - σ_d(G) = |E_int| - (d/2)·L is the superficial degree of divergence
  - G' = G after ω-integration (L circuits contracted)
  - Ψ', Φ' = Symanzik polynomials with diffusion constants D_e absorbed
  - Σ_l = Σ_e c_{le} α_e is the l-th circuit constraint (from E_f matrix)

The ω-integration is the key new result (Theorem 5.1 of the companion paper).
For each circuit l, the delta function δ(Σ_l) contracts the integration
domain by one dimension:

  ∫ dα_{e_max} δ(Σ_l) f(α) = f(α)|_{α_{e_max} = Σ_l - Σ_{e≠e_max} c_{le} α_e}

Algorithm:
  1. Read edge-basis matrix E_f (each row = one circuit)
  2. For each circuit l:
     a. Identify α_{e_max} = leading edge in Σ_l
     b. Express α_{e_max} in terms of remaining α's
     c. Substitute into Ψ', Φ', reduce integral dimension by 1
  3. After L substitutions: Ψ', Φ' are polynomials in (n_int - L) variables
  4. Perform remaining α-integrations via Euler-Beta formula

Mathematical reference:
    Amarteifio (2019) §2.5.3.2, §2.5.3.3
    Bogner-Weinzierl (2010), Panzer (2015) PhD thesis
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import sympy as sp

from ..graphs.incidence import FeynmanGraph
from .symanzik import SymanzikPolynomials


class OmegaIntegration:
    """
    Perform ω-integration via δ(Σ_l) substitutions.

    This reduces the L-dimensional ω-integration to a product of
    delta functions, each contracting one alpha-parameter integration.

    For a graph G with L loops:
      - Before ω-integration: Ψ and Φ depend on all n_internal_edges α_e
      - After ω-integration: Ψ' and Φ' depend on only n_internal_edges - L
        independent α parameters

    The edge-basis matrix E_f encodes which edges carry loop momenta.
    Each row of E_f gives the circuit constraint Σ_l.

    Parameters
    ----------
    graph : FeynmanGraph
    """

    def __init__(self, graph: FeynmanGraph):
        self.graph = graph
        self._Ef: Optional[sp.Matrix] = None
        self._circuit_constraints: Optional[List[sp.Expr]] = None

    @property
    def edge_basis_matrix(self) -> sp.Matrix:
        """
        The edge-basis matrix E_f.

        Each row l corresponds to an independent circuit.
        Entry E_f[l, e] = ±1 if edge e is in circuit l (with sign
        determined by orientation), 0 otherwise.

        For a graph with L loops, E_f has shape L × n_internal_edges.

        This is the matrix from Amarteifio §2.5.3.2, used to determine
        the delta function arguments Σ_l = Σ_e E_f[l,e] · α_e.

        Implementation: find a spanning tree T, then each non-tree edge
        defines a fundamental circuit. The circuit includes the non-tree
        edge plus the unique tree path connecting its endpoints.
        """
        if self._Ef is not None:
            return self._Ef

        int_edges = self.graph.internal_edge_indices
        n_int = len(int_edges)
        L = self.graph.L

        if L == 0:
            self._Ef = sp.zeros(0, n_int)
            return self._Ef

        # Find a spanning tree via greedy DFS
        spanning_tree = self._find_spanning_tree()
        tree_set = set(spanning_tree)
        non_tree_edges = [e for e in int_edges if e not in tree_set]

        # For each non-tree edge (loop edge), find the fundamental circuit
        Ef = sp.zeros(L, n_int)

        for l, loop_edge_idx in enumerate(non_tree_edges):
            src, tgt, _ = self.graph.edges[loop_edge_idx]

            # Find unique tree path from src to tgt (just to identify which
            # tree edges are in this fundamental circuit — direction ignored)
            path = self._tree_path(spanning_tree, src, tgt)

            # Assign signs following the RDFT ω-integration convention
            # (Amarteifio Theorem 5.1): the circuit constraint is
            #   α_{loop} = Σ_{e' in tree path} α_{e'}
            # which means: non-tree (loop) edge → +1, tree edges → -1.
            # This is a topological statement on Schwinger parameters
            # (α encodes "time" / "length", not momentum direction), so
            # edge orientations do NOT affect the sign.
            alpha_pos = int_edges.index(loop_edge_idx)
            Ef[l, alpha_pos] = sp.S.One  # loop edge itself: +1

            for edge_idx, _sign in path:
                if edge_idx in int_edges:
                    alpha_pos = int_edges.index(edge_idx)
                    Ef[l, alpha_pos] = sp.Integer(-1)  # tree edges: always -1

        self._Ef = Ef
        return self._Ef

    def _find_spanning_tree(self) -> List[int]:
        """Find a spanning tree of the internal subgraph using DFS."""
        int_edges = self.graph.internal_edge_indices
        n_verts = self.graph.n_vertices_int

        visited = set()
        tree = []

        # Start from vertex 0
        stack = [0]
        visited.add(0)

        while stack and len(tree) < n_verts - 1:
            v = stack[-1]
            added = False
            for e_idx in int_edges:
                src, tgt, _ = self.graph.edges[e_idx]
                if e_idx not in tree:
                    if src == v and tgt not in visited:
                        tree.append(e_idx)
                        visited.add(tgt)
                        stack.append(tgt)
                        added = True
                        break
                    elif tgt == v and src not in visited:
                        tree.append(e_idx)
                        visited.add(src)
                        stack.append(src)
                        added = True
                        break
            if not added:
                stack.pop()

        return tree

    def _tree_path(self,
                   spanning_tree: List[int],
                   start: int,
                   end: int) -> List[Tuple[int, int]]:
        """Find the path in the spanning tree from start to end."""
        # Build adjacency list for spanning tree
        adj: Dict[int, List[Tuple[int, int, int]]] = {}
        for e_idx in spanning_tree:
            src, tgt, _ = self.graph.edges[e_idx]
            adj.setdefault(src, []).append((tgt, e_idx, +1))
            adj.setdefault(tgt, []).append((src, e_idx, -1))

        # BFS to find path
        from collections import deque
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            v, path = queue.popleft()
            if v == end:
                return path
            for next_v, e_idx, sign in adj.get(v, []):
                if next_v not in visited:
                    visited.add(next_v)
                    queue.append((next_v, path + [(e_idx, sign)]))

        return []  # No path (disconnected)

    @property
    def circuit_constraints(self) -> List[sp.Expr]:
        """
        List of circuit constraint polynomials Σ_l = Σ_e E_f[l,e] · α_e.

        Each δ(Σ_l) in the parametric integral contracts one α integration.
        """
        if self._circuit_constraints is not None:
            return self._circuit_constraints

        Ef = self.edge_basis_matrix
        alphas = self.graph._alpha_syms
        L = self.graph.L

        constraints = []
        for l in range(L):
            Sigma_l = sum(Ef[l, e] * alphas[e] for e in range(len(alphas)))
            constraints.append(sp.expand(Sigma_l))

        self._circuit_constraints = constraints
        return self._circuit_constraints

    def reduce(self, Psi: sp.Expr, Phi: sp.Expr) -> Tuple[sp.Expr, sp.Expr, Dict]:
        """
        Apply all L delta-function substitutions to Ψ and Φ.

        For each circuit l:
          1. Identify the 'leading' alpha in Σ_l (highest-indexed for
             canonical ordering)
          2. Express it as α_{lead} = Σ_l - Σ_{other}
          3. Substitute into Ψ, Φ, removing that degree of freedom

        Returns
        -------
        Psi_reduced : Ψ after all L substitutions
        Phi_reduced : Φ after all L substitutions
        substitutions : dict of alpha → expression (for record-keeping)
        """
        Psi_r = Psi
        Phi_r = Phi
        subs_record = {}
        alphas = list(self.graph._alpha_syms)

        for l, Sigma_l in enumerate(self.circuit_constraints):
            # Find the alpha to eliminate: last one with nonzero coefficient
            lead_alpha = None
            lead_coeff = None
            for alpha in reversed(alphas):
                if alpha not in subs_record:
                    coeff = Sigma_l.coeff(alpha)
                    if coeff != 0:
                        lead_alpha = alpha
                        lead_coeff = coeff
                        break

            if lead_alpha is None:
                continue  # Degenerate; skip

            # Express: lead_alpha = (Sigma_l - rest) / lead_coeff
            rest = sp.expand(Sigma_l - lead_coeff * lead_alpha)
            alpha_val = sp.simplify(-rest / lead_coeff)

            # Record and substitute
            subs_record[lead_alpha] = alpha_val
            Psi_r = sp.expand(Psi_r.subs(lead_alpha, alpha_val))
            Phi_r = sp.expand(Phi_r.subs(lead_alpha, alpha_val))

        return Psi_r, Phi_r, subs_record


class ParametricIntegral:
    """
    Full parametric integral I(G; d) for a Feynman graph G.

    Computes the result as a function of d (or ε = d_c - d).

    For reaction-diffusion processes:
      - Masses m_e relate to distance from criticality
      - Diffusion constants D_e scale the alpha parameters: α_e → α_e/D_e
      - After ω-integration: d-dependent rational functions of ε

    Parameters
    ----------
    graph : FeynmanGraph
    symanzik : SymanzikPolynomials (precomputed or computed here)
    d : symbolic dimension (default: symbolic 'd')
    diffusion_constants : dict edge_idx → D_e value
    """

    def __init__(self,
                 graph: FeynmanGraph,
                 symanzik: Optional[SymanzikPolynomials] = None,
                 d: Optional[sp.Expr] = None,
                 diffusion_constants: Optional[Dict[int, sp.Expr]] = None):

        self.graph = graph
        self.symanzik = symanzik or SymanzikPolynomials(graph)
        self.d = d or sp.Symbol('d', positive=True)
        self.D = diffusion_constants or {}

        self._omega_integrator = OmegaIntegration(graph)

    def angular_factor(self) -> sp.Expr:
        """
        Ω_d = 2π^{d/2} / (2π)^d = (4π)^{-d/2}

        This is the angular part of the d-dimensional Gaussian integral.
        (Amarteifio eq. 2.100)
        """
        return (4 * sp.pi) ** (-self.d / 2)

    def degree_of_divergence(self) -> sp.Expr:
        """σ_d(G) = |E_int| - d/2 · L (Amarteifio eq. 2.101)"""
        return self.graph.degree_of_divergence(self.d)

    def apply_diffusion_scaling(self, Psi: sp.Expr, Phi: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
        """
        Scale alpha parameters by diffusion constants.

        For species with diffusion D_e, replace α_e → α_e / D_e.
        This produces the primed polynomials Ψ', Φ'.

        (Amarteifio §2.5.3, examples 2.5.15-2.5.17)
        """
        alphas = self.graph._alpha_syms
        int_edges = self.graph.internal_edge_indices

        if not self.D:
            return Psi, Phi  # No scaling

        subs_dict = {}
        for i, edge_idx in enumerate(int_edges):
            D_e = self.D.get(edge_idx)
            if D_e is not None:
                subs_dict[alphas[i]] = alphas[i] / D_e

        Psi_p = sp.expand(Psi.subs(subs_dict))
        Phi_p = sp.expand(Phi.subs(subs_dict))

        return Psi_p, Phi_p

    def compute(self,
                as_epsilon_expansion: bool = True,
                n_terms: int = 2) -> sp.Expr:
        """
        Compute I(G; d) as a symbolic expression.

        Steps:
          1. Compute Ψ, Φ
          2. Apply diffusion scaling → Ψ', Φ'
          3. Apply ω-integration (δ(Σ_l) reductions) → reduced Ψ'', Φ''
          4. Evaluate remaining alpha integrals via Euler-Beta formula
          5. Optionally expand in ε = d_c - d

        For one-loop graphs, this is analytic.
        For multi-loop graphs, the result involves iterated Beta functions.

        Returns a sympy expression.
        """
        Psi = self.symanzik.Psi
        Phi = self.symanzik.Phi

        # Apply diffusion scaling
        Psi_p, Phi_p = self.apply_diffusion_scaling(Psi, Phi)

        sigma = self.degree_of_divergence()
        Omega = self.angular_factor()

        # Degenerate case: n_int == L means ω-integration would eliminate ALL
        # Schwinger parameters (leaving Ψ_r = 0).  Instead, evaluate the
        # Schwinger integral directly on the unduced Ψ and Φ.
        # Example: tadpole (self-loop, n_int=1, L=1) has n_free=0 after ω-int.
        # The correct amplitude is Ω × ∫ dα Ψ(α)^{-d/2} exp(-Φ(α)/Ψ(α)).
        if self.graph.n_internal_edges == self.graph.L:
            return self._evaluate_alpha_integrals(Psi_p, Phi_p, sigma, Omega)

        # Standard path: apply ω-integration first
        Psi_r, Phi_r, _ = self._omega_integrator.reduce(Psi_p, Phi_p)

        result = self._evaluate_alpha_integrals(Psi_r, Phi_r, sigma, Omega)

        return result

    def _evaluate_alpha_integrals(self,
                                   Psi_r: sp.Expr,
                                   Phi_r: sp.Expr,
                                   sigma: sp.Expr,
                                   Omega: sp.Expr) -> sp.Expr:
        """
        Evaluate ∫ Π dα Ψ_r^{-d/2} exp(-Φ_r/Ψ_r) analytically.

        Uses the Schwinger-parameter Gamma formula for monomial integrands.

        Key result: for Ψ_r = C·α^p and Q = Φ_r/Ψ_r = M·α^k,

            ∫_0^∞ dα (C·α^p)^{-d/2} exp(-M·α^k)
                = C^{-d/2} / k · Γ((1 - p·d/2) / k) · M^{-(1-p·d/2)/k}

        For n_free ≥ 2: if Q = Σ_i Q_i(α_i) (sum-separable) and Ψ_r is
        a product Π_i Ψ_i(α_i), the multi-dimensional integral factorises
        into independent 1D integrals and the same formula applies to each.

        The angular factor Ω_d^L = (4π)^{-Ld/2} is supplied as Omega^L
        (caller passes Omega = (4π)^{-d/2}, so Omega^L = (4π)^{-Ld/2}).
        """
        alphas_free = [a for a in self.graph._alpha_syms
                       if Psi_r.has(a) or Phi_r.has(a)]

        if len(alphas_free) == 0:
            # Tree level: no remaining integrals
            return Omega * sp.gamma(-sigma)

        L = self.graph.L
        Omega_L = Omega ** L

        if len(alphas_free) == 1:
            result = self._schwinger_1d(alphas_free[0], Psi_r, Phi_r)
            if result is not None:
                return sp.simplify(Omega_L * result)

        elif len(alphas_free) >= 2:
            # Attempt factorised evaluation: Ψ_r = Π_i f_i(α_i)
            # and Q = Σ_i Q_i(α_i).
            result = self._schwinger_factorised(alphas_free, Psi_r, Phi_r)
            if result is not None:
                return sp.simplify(Omega_L * result)

        # Could not evaluate analytically
        return (Omega_L * sp.gamma(-sigma) *
                sp.Symbol('I_alpha_unsolved', real=True))

    def _schwinger_1d(self,
                      alpha: sp.Symbol,
                      Psi_r: sp.Expr,
                      Phi_r: sp.Expr) -> Optional[sp.Expr]:
        """
        Evaluate ∫_0^∞ dα Ψ_r^{-d/2} exp(-Q)  with Q = Phi_r/Psi_r
        for the case Ψ_r = C·α^p and Q = M·α^k (both pure monomials).

        Returns the result as a sympy expression, or None if the form
        is not recognised.

        Formula:
            C^{-d/2} / k · Γ((1 - p·d/2) / k) · M^{-(1-p·d/2)/k}
        """
        d = self.d

        psi_poly = sp.Poly(Psi_r, alpha)
        p = psi_poly.degree()
        C = psi_poly.nth(p)

        # Ψ_r must be a pure monomial (zero constant and all intermediate terms)
        if any(psi_poly.nth(i) != 0 for i in range(p)):
            return None  # not a monomial

        if sp.simplify(Phi_r) == 0:
            # Massless case: no exp factor.  Use projective representation:
            # ∫_0^∞ dα (C·α^p)^{-d/2}  is regularised by dim-reg to
            # C^{-d/2} · δ(exponent)  → gives a pole structure via Γ.
            # Return the projective result (Cheng-Wu with α=1):
            exponent = -p * d / 2
            # ∫_0^∞ dα α^{exponent} regulated = Γ(1+exponent) / (−exponent) ...
            # Standard result: = 0 by dimensional regularisation for massless
            # self-energy tadpoles with no external scale.
            return sp.S.Zero

        Q = sp.simplify(sp.cancel(Phi_r / Psi_r))
        q_poly = sp.Poly(Q, alpha)
        k = q_poly.degree()
        M = q_poly.nth(k)

        # Q must be a pure monomial
        if any(q_poly.nth(i) != 0 for i in range(k)):
            return None

        # Γ formula: ∫_0^∞ dα (C·α^p)^{-d/2} exp(-M·α^k)
        #          = C^{-d/2} / k · Γ((1-p·d/2)/k) · M^{-(1-p·d/2)/k}
        exponent_arg = (1 - p * d / 2) / k
        return (C ** (-d / 2)
                / sp.Integer(k)
                * sp.gamma(exponent_arg)
                * M ** (-exponent_arg))

    def _schwinger_factorised(self,
                               alphas: List[sp.Symbol],
                               Psi_r: sp.Expr,
                               Phi_r: sp.Expr) -> Optional[sp.Expr]:
        """
        Multi-dimensional Schwinger integral for factorisable integrands.

        Requires:
          Ψ_r = Π_i Ψ_i(α_i)       (product over free parameters)
          Q   = Σ_i Q_i(α_i)       (sum over free parameters)

        where each Ψ_i and Q_i depend on only one α_i.

        Under these conditions exp(-Q) = Π_i exp(-Q_i) and the
        multi-dimensional integral = Π_i [1D Schwinger integral for α_i].

        Returns the product of 1D results, or None if factorisation fails.
        """
        # --- Check Ψ_r factorises as a product of univariate pieces ---
        # Strategy: for each α_i, collect the factor of Ψ_r that depends on α_i.
        psi_factors: Dict[sp.Symbol, sp.Expr] = {}

        # Try: Ψ_r = f₀(α₀) × f₁(α₁) × ...
        remaining = sp.expand(Psi_r)
        for alpha in alphas:
            others = [a for a in alphas if a is not alpha]
            # Factor out the α_i-dependent part
            factor = sp.collect(remaining, alpha, evaluate=False)
            # Check if remaining / factor(alpha→1) is independent of alpha
            try:
                psi_at_1 = remaining.subs(alpha, sp.S.One)
                ratio = sp.simplify(sp.cancel(remaining / psi_at_1))
                if not ratio.has(*others):
                    # ratio depends only on alpha; psi_at_1 on the rest
                    psi_factors[alpha] = ratio
                    remaining = psi_at_1
                else:
                    return None  # not factorisable at this α
            except Exception:
                return None

        # --- Check Q = Φ_r/Ψ_r decomposes as a sum ---
        Q_full = sp.simplify(sp.cancel(Phi_r / Psi_r))
        Q_parts: Dict[sp.Symbol, sp.Expr] = {}

        q_remaining = sp.expand(Q_full)
        for alpha in alphas:
            others = [a for a in alphas if a is not alpha]
            # Extract terms of Q that involve alpha
            q_terms = sp.collect(q_remaining, alpha, evaluate=False)
            q_alpha = sum(
                (coeff * alpha ** n)
                for n, coeff in q_terms.items()
                if isinstance(n, int) and n > 0
                and not sp.sympify(coeff).has(*others)
            ) if isinstance(q_terms, dict) else sp.S.Zero

            if q_alpha == sp.S.Zero:
                # Try directly: part depending on alpha only
                q_alpha = sp.S.Zero
                for term in sp.Add.make_args(q_remaining):
                    if term.has(alpha) and not term.has(*others):
                        q_alpha += term

            Q_parts[alpha] = q_alpha
            q_remaining = sp.expand(q_remaining - q_alpha)

        # Any remainder must be zero (or we can't factorise)
        if sp.simplify(q_remaining) != 0:
            return None

        # --- Evaluate each 1D factor ---
        # `remaining` now holds the constant prefactor C of Ψ_r = C × Π_i ψ_i(α_i).
        # This constant contributes C^{-d/2} to the integral.
        psi_constant = remaining  # scalar; no free alphas remain
        d = self.d

        product = psi_constant ** (-d / 2) if psi_constant != sp.S.One else sp.S.One

        for alpha in alphas:
            Psi_i = sp.expand(psi_factors[alpha])
            Q_i   = sp.expand(Q_parts[alpha])
            Phi_i = sp.expand(Q_i * Psi_i)  # reconstruct Phi_i = Q_i × Ψ_i

            result_i = self._schwinger_1d(alpha, Psi_i, Phi_i)
            if result_i is None:
                return None
            product *= result_i

        return product

    def epsilon_expansion(self, d_c: sp.Expr, n_terms: int = 2) -> sp.Expr:
        """
        Expand I(G; d) in ε = d_c - d around the upper critical dimension.

        The UV poles appear as 1/ε and 1/ε² terms from Γ(ε/2) ~ 2/ε.

        Returns Laurent series in ε.
        """
        eps = sp.Symbol('epsilon', positive=True)
        d_val = d_c - eps
        result = self.compute().subs(self.d, d_val)
        return sp.series(result, eps, 0, n_terms)


# ------------------------------------------------------------------ #
#  Convenience: reproduce thesis examples                              #
# ------------------------------------------------------------------ #

def thesis_example_2515(D_A=None, m_A=None) -> sp.Expr:
    """
    Reproduce Amarteifio (2019) Example 2.5.15:
    One-loop self-energy with two A-species propagators.

    Expected result (eq. 2.103c):
        I = A_d · Γ(1 - d/2) · [2m_A]^{2/d - 1}
    where A_d = (4πD_A)^{-d/2}
    """
    d = sp.Symbol('d', positive=True)
    D = D_A or sp.Symbol('D_A', positive=True)
    m = m_A or sp.Symbol('m_A', positive=True)

    A_d = (4 * sp.pi * D) ** (-d / 2)
    result = A_d * sp.gamma(1 - d/2) * (2*m) ** (sp.Integer(2)/d - 1)
    return result


def thesis_example_2516(D_A=None, D_B=None, m_A=None, m_B=None) -> sp.Expr:
    """
    Reproduce Amarteifio (2019) Example 2.5.16:
    One-loop with two different species A, B (D_B = 0 case).

    Expected (eq. 2.105):
        I = A_d · Γ(1 - d/2) · [m_A + m_B]^{2/d - 1}
    where A_d = (1/2)(4π)^{-d/2}  (no D_B diffusion, A_d modified)
    """
    d = sp.Symbol('d', positive=True)
    m_a = m_A or sp.Symbol('m_A', positive=True)
    m_b = m_B or sp.Symbol('m_B', positive=True)

    A_d = sp.Rational(1, 2) * (4 * sp.pi) ** (-d / 2)
    result = A_d * sp.gamma(1 - d/2) * (m_a + m_b) ** (sp.Integer(2)/d - 1)
    return result
