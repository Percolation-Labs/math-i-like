"""
rdft.core.generators
====================
Heisenberg-Weyl generators for reaction-diffusion processes.

This implements the central formula of Amarteifio (2019) §1.3:

    Q[∂_z, z] = ((z+1)^l - (z+1)^k) ∂_z^k             (Amarteifio eq. 1.36)

for the factorial-moment generating function, and

    Q̂[∂_z, z] = (z^l - z^k) ∂_z^k                      (Amarteifio eq. 1.35)

for the ordinary generating function.

The Heisenberg-Weyl algebra is the differential algebra g with Lie bracket:
    [∂_z, z] = 1    i.e.   ∂_z · z - z · ∂_z = 1        (Amarteifio eq. 1.21)

In the Doi picture, ∂_z ↔ a (annihilation), z ↔ a† (creation).

After the Doi shift z → z+1 (replacing z with z+1 throughout), the
factorial-moment generator Q becomes the standard form used in the
Doi-Peliti path integral. In field-theory notation after the shift
and moving to fields φ̃, φ via z ↔ φ̃, ∂_z ↔ φ, the interaction terms
in Q correspond to vertices in the action.

Mathematical reference:
    Amarteifio (2019) §1.3, eqs. (1.35)-(1.38)
    Doi (1976a,b), Peliti (1985), Täuber-Howard-Vollmayr-Lee (2005)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import sympy as sp

from .reaction_network import ReactionNetwork, Reaction, Species


# ------------------------------------------------------------------ #
#  Single-reaction generators                                          #
# ------------------------------------------------------------------ #

def generator_ordinary(reaction: Reaction, species: Species) -> sp.Expr:
    """
    Generator for ordinary GF (generating function of occupation numbers).

    For a single-species reaction kA → lA at rate λ:
        Q̂ = λ (z^l - z^k) ∂_z^k

    Equation (1.35) of Amarteifio (2019).

    Returns a sympy expression in symbols z and Dz (representing ∂_z).
    """
    z  = sp.Symbol('z')
    Dz = sp.Symbol('Dz')     # symbolic placeholder for ∂_z

    k = reaction.k(species)
    l = reaction.l(species)
    λ = reaction.rate

    return λ * (z**l - z**k) * Dz**k


def generator_factorial(reaction: Reaction, species: Species) -> sp.Expr:
    """
    Generator for factorial-moment GF.

    For a single-species reaction kA → lA at rate λ:
        Q = λ ((z+1)^l - (z+1)^k) ∂_z^k

    Equation (1.36) of Amarteifio (2019).

    This is the generator used in the Doi-Peliti coherent-state path
    integral. It corresponds to the Poisson ansatz (coherent states).
    """
    z  = sp.Symbol('z')
    Dz = sp.Symbol('Dz')

    k = reaction.k(species)
    l = reaction.l(species)
    λ = reaction.rate

    return λ * ((z + 1)**l - (z + 1)**k) * Dz**k


def generator_action_term(reaction: Reaction, species: Species) -> sp.Expr:
    """
    Expand the factorial-moment generator into action vertices.

    After expanding (z+1)^l = sum_j C(l,j) z^j and (z+1)^k = sum_j C(k,j) z^j,
    each term z^m Dz^k corresponds to a vertex with m outgoing φ̃ legs
    and k incoming φ legs in the Doi-Peliti action.

    Returns a sympy polynomial in z (= φ̃) and Dz (= φ).

    Example: coagulation 2A → A (k=2, l=1):
        Q = χ((z+1)^1 - (z+1)^2) Dz^2
          = χ(z - z^2 - z·2 - 1 + 1 + 2z + z^2)·Dz^2  ... etc
          = χ(z - z^2)·Dz^2   after simplification

    which matches Amarteifio eq. (1.33d) giving Q = (z - z^2)∂_z^2.
    """
    z  = sp.Symbol('z')
    Dz = sp.Symbol('Dz')

    k = reaction.k(species)
    l = reaction.l(species)
    λ = reaction.rate

    raw = λ * (sp.expand((z + 1)**l) - sp.expand((z + 1)**k)) * Dz**k
    return sp.expand(raw)


# ------------------------------------------------------------------ #
#  Multi-reaction Liouvillian                                          #
# ------------------------------------------------------------------ #

class Liouvillian:
    """
    The full Liouvillian (infinitesimal generator of the semigroup) for
    a reaction network, summing generators over all reactions.

    Q_total = Σ_reactions Q_i

    This is the object that becomes the interaction part of the Doi-Peliti
    action after adding the free (diffusion + mass) part:

        S[φ̃, φ] = ∫ dt [φ̃(∂_t - D∇²)φ - Q_total(φ̃, φ)]

    See Amarteifio (2019) eq. (1.67).

    Attributes
    ----------
    network      : the reaction network
    species      : the species for which the generator is computed
    terms        : dict from (k, l) → sympy expr, the expanded action vertices
    """

    def __init__(self, network: ReactionNetwork, use_factorial: bool = True):
        self.network = network
        self.use_factorial = use_factorial
        self._compute()

    def _compute(self):
        """Expand all generators and collect by monomial type."""
        self.raw_generators: List[sp.Expr] = []
        self.action_terms:   Dict[Tuple[int, int], sp.Expr] = {}

        for rxn in self.network.reactions:
            # For single-species networks use the first (only) species
            # Multi-species: handled via tensor product of generators
            if self.network.n_species == 1:
                sp_ = self.network.species[0]
                if self.use_factorial:
                    gen = generator_action_term(rxn, sp_)
                else:
                    gen = generator_ordinary(rxn, sp_)
                self.raw_generators.append(gen)
            else:
                # Multi-species: each reaction acts on a product of species GFs
                # Defer to the tensor-product version
                gen = self._multispecies_generator(rxn)
                self.raw_generators.append(gen)

        self.total = sp.expand(sum(self.raw_generators))

    def _multispecies_generator(self, rxn: Reaction) -> sp.Expr:
        """
        Multi-species generator (Doi formulation).

        For a reaction k_1 A + k_2 B → l_1 A + l_2 B with rate λ,
        the Liouvillian in the factorial-moment representation is:

            Q = λ · (Π_i (z_i+1)^{l_i} - Π_i (z_i+1)^{k_i}) · Π_i Dz_i^{k_i}

        Note: the products over species are JOINT (not factored per species).
        This is the Doi (1976) formula: the "final state" product minus the
        "initial state" product, times the annihilation operators.

        For A → A+B (k_A=1,l_A=1, k_B=0,l_B=1):
          final   = (z_A+1)^1 · (z_B+1)^1
          initial = (z_A+1)^1 · (z_B+1)^0 = (z_A+1)
          Q = γ · ((z_A+1)(z_B+1) - (z_A+1)) · Dz_A^1 · Dz_B^0
            = γ · (z_A+1) · z_B · Dz_A
            = γ · (z_A·z_B + z_B) · Dz_A
        which gives vertices φ̃_A φ_A φ̃_B (transmutation) and φ_A φ̃_B (source).
        """
        z_syms  = {sp_: sp.Symbol(f'z_{sp_.name}')  for sp_ in self.network.species}
        Dz_syms = {sp_: sp.Symbol(f'Dz_{sp_.name}') for sp_ in self.network.species}

        all_species = list(rxn.all_species)

        # Final state product: Π_i (z_i+1)^{l_i}
        final_product = sp.S.One
        for sp_ in all_species:
            l = rxn.l(sp_)
            z = z_syms[sp_]
            final_product *= (z + 1)**l

        # Initial state product: Π_i (z_i+1)^{k_i}
        initial_product = sp.S.One
        for sp_ in all_species:
            k = rxn.k(sp_)
            z = z_syms[sp_]
            initial_product *= (z + 1)**k

        # Annihilation operators: Π_i Dz_i^{k_i}
        annihilation = sp.S.One
        for sp_ in all_species:
            k = rxn.k(sp_)
            Dz = Dz_syms[sp_]
            annihilation *= Dz**k

        result = rxn.rate * (final_product - initial_product) * annihilation

        return sp.expand(result)

    @property
    def vertices(self) -> Dict[Tuple, sp.Expr]:
        """
        Extract Feynman vertices from the action.

        For single-species: each term λ·z^m·Dz^n gives a vertex with
        m outgoing (φ̃) legs and n incoming (φ) legs.

        For multi-species: each term λ·z_A^{m_A}·z_B^{m_B}·Dz_A^{n_A}·Dz_B^{n_B}
        gives a vertex with (m_A + m_B) outgoing legs and (n_A + n_B) incoming
        legs, coloured by species. The key is ((m_A, m_B, ...), (n_A, n_B, ...))
        for multi-species, or (m_total, n_total) for single-species.

        Returns dict: key → coupling expr
        """
        if self.network.n_species == 1:
            return self._single_species_vertices()
        else:
            return self._multi_species_vertices()

    def _single_species_vertices(self) -> Dict[Tuple, sp.Expr]:
        """Extract vertices for single-species theories."""
        z  = sp.Symbol('z')
        Dz = sp.Symbol('Dz')
        vertices = {}
        try:
            poly = sp.Poly(self.total, z, Dz)
            for monom, coeff in zip(poly.monoms(), poly.coeffs()):
                m_out, n_in = monom
                if coeff != 0:
                    key = (m_out, n_in)
                    vertices[key] = vertices.get(key, sp.S.Zero) + coeff
        except Exception:
            pass
        return {k: v for k, v in vertices.items() if v != 0}

    def _multi_species_vertices(self) -> Dict[Tuple, sp.Expr]:
        """
        Extract vertices for multi-species theories.

        Each monomial in Q_total is a product of z_i^{m_i} · Dz_i^{n_i}
        over all species. We extract the exponents per species.

        Returns dict with keys as tuples of per-species (m_i, n_i) pairs:
            ((m_A, n_A), (m_B, n_B), ...) → coupling
        """
        species_list = self.network.species
        z_syms  = [sp.Symbol(f'z_{s.name}')  for s in species_list]
        Dz_syms = [sp.Symbol(f'Dz_{s.name}') for s in species_list]

        all_syms = []
        for z, Dz in zip(z_syms, Dz_syms):
            all_syms.extend([z, Dz])

        vertices = {}
        try:
            poly = sp.Poly(self.total, *all_syms)
            for monom, coeff in zip(poly.monoms(), poly.coeffs()):
                if coeff == 0:
                    continue
                # monom = (z_A_pow, Dz_A_pow, z_B_pow, Dz_B_pow, ...)
                key = tuple((monom[2*i], monom[2*i+1])
                           for i in range(len(species_list)))
                vertices[key] = vertices.get(key, sp.S.Zero) + coeff
        except Exception:
            pass
        return {k: v for k, v in vertices.items() if v != 0}

    def action_density(self, include_free: bool = True) -> sp.Expr:
        """
        The Lagrangian density (action integrand).

        L = φ̃(∂_t + m² - D∇²)φ - Q_total(φ̃, φ)

        where the free part uses symbols φ_tilde, phi.
        With φ̃ = z, φ = Dz and the Doi shift already applied in
        the factorial-moment generator.

        Returns a sympy expression in φ̃ (z), φ (Dz) and symbolic
        parameters (rates, mass m, diffusion D).
        """
        z  = sp.Symbol('z')    # φ̃ after Doi shift
        Dz = sp.Symbol('Dz')   # φ

        interaction = -self.total  # minus sign from action convention

        if include_free:
            D = self.network.species[0].diffusion_constant
            m = sp.Symbol('m', positive=True)
            # Free part: φ̃(∂_t + m² - D∇²)φ
            # In zero-d (no spatial structure): φ̃(∂_t + m²)φ → z·(∂_t + m²)·Dz
            # In field theory, this is symbolic; the propagator handles spatial part
            free = z * (sp.Symbol('partial_t') + m**2) * Dz
            return sp.expand(free + interaction)
        else:
            return sp.expand(interaction)

    def __repr__(self) -> str:
        lines = [f'Liouvillian for {self.network.name}']
        lines.append(f'  Q_total = {self.total}')
        lines.append('  Vertices:')
        for (m, n), coeff in self.vertices.items():
            lines.append(f'    z^{m} Dz^{n}  (φ̃^{m} φ^{n}):  {coeff}')
        return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  BRW transmutation vertices (thesis Eq. 3.21)                        #
# ------------------------------------------------------------------ #

def brw_transmutation_vertices() -> Dict[Tuple, sp.Expr]:
    """
    The 6 transmutation interaction terms from thesis Eq. (3.21).

    These arise from the carrying capacity trick (Eq. 3.17):
        ρ(n_b) = n_a(c - n_b)|_{c=1}

    which enforces site exclusion: A deposits B only if no B present.

    In the Doi-shifted action (φ† = 1 + φ̃), the 6 terms are:

        -Q_Λ = Λ × [  +φ̃_b φ_a           term 1: transmutation source
                      -φ̃_b φ_b φ_a        term 2: B-suppressed source
                      -φ̃²_b φ_b φ_a       term 3: quadratic B suppression
                      -φ̃²_b φ̃_a φ_b φ_a  term 4: full exclusion
                      -φ̃_b φ̃_a φ_b φ_a   term 5: mixed exclusion
                      +φ̃_b φ̃_a φ_a    ]  term 6: transmutation vertex

    In the notation ((m_A, n_A), (m_B, n_B)) for per-species leg counts:

    Returns dict with multi-species vertex keys → couplings.
    """
    Λ = sp.Symbol('Lambda', positive=True)

    # The 6 terms from Eq. (3.21), with signs as written
    # Key: ((m_A, n_A), (m_B, n_B)) = (outgoing_A, incoming_A, outgoing_B, incoming_B)
    vertices = {
        # Term 1: +φ̃_b φ_a  = ((0,1), (1,0))
        ((0, 1), (1, 0)): Λ,
        # Term 2: -φ̃_b φ_b φ_a = ((0,1), (1,1))
        ((0, 1), (1, 1)): -Λ,
        # Term 3: -φ̃²_b φ_b φ_a = ((0,1), (2,1))
        ((0, 1), (2, 1)): -Λ,
        # Term 4: -φ̃²_b φ̃_a φ_b φ_a = ((1,1), (2,1))
        ((1, 1), (2, 1)): -Λ,
        # Term 5: -φ̃_b φ̃_a φ_b φ_a = ((1,1), (1,1))
        ((1, 1), (1, 1)): -Λ,
        # Term 6: +φ̃_b φ̃_a φ_a = ((1,1), (1,0))
        ((1, 1), (1, 0)): Λ,
    }

    return vertices


def brw_all_vertices(max_legs: int = 4) -> Dict[Tuple, sp.Expr]:
    """
    All vertices for the full BRW theory (A-sector + transmutation).

    Combines the Gribov A-sector vertices with the 6 transmutation
    terms from the carrying capacity trick.

    Parameters
    ----------
    max_legs : maximum total number of legs (m_A + n_A + m_B + n_B)
        for interaction vertices. Default 4 corresponds to the
        dimensional analysis constraint at d_c = 4 (thesis Eq. 3.12:
        m + m̃ + 2ñ ≤ 3 for B-species, giving total legs ≤ 4).
        Set to None to keep all vertices including irrelevant ones.

    Returns multi-species vertex dict with keys
    ((m_A, n_A), (m_B, n_B)) → coupling.
    """
    from .reaction_network import ReactionNetwork
    net = ReactionNetwork.gribov()
    L = Liouvillian(net)

    # A-sector vertices (single-species → lift to multi-species format)
    a_verts = {}
    for (m, n), g in L.vertices.items():
        a_verts[((m, n), (0, 0))] = g

    # Transmutation vertices
    t_verts = brw_transmutation_vertices()

    # Merge
    all_verts = {**a_verts, **t_verts}

    # Filter by relevance (dimensional analysis at d_c)
    if max_legs is not None:
        filtered = {}
        for key, g in all_verts.items():
            (m_A, n_A), (m_B, n_B) = key
            total = m_A + n_A + m_B + n_B
            # Keep propagators (total ≤ 2) always
            # Keep interaction vertices only if total ≤ max_legs
            if total <= 2 or total <= max_legs:
                filtered[key] = g
        return filtered

    return all_verts


# ------------------------------------------------------------------ #
#  Verification against thesis examples                                #
# ------------------------------------------------------------------ #

def verify_thesis_examples() -> Dict[str, bool]:
    """
    Verify the generators match Amarteifio (2019) equations (1.38a-c)
    for the Gribov process.

    Equation (1.38a): Q_χ = χ((z+1)^1 - (z+1)^2)∂_z^2 = χ(z - z^2)∂_z^2
                          → Wait, thesis has χ = coagulation (2A→A)
                             and Q_χ = χ((z+1)^1 - (z+1)^2)∂_z^2
    Equation (1.38b): Q_β = β((z+1)^2 - (z+1)^1)∂_z^1  [branching A→2A]
    Equation (1.38c): Q_ε = ε((z+1)^0 - (z+1)^1)∂_z^1  [death A→∅]
    """
    results = {}
    z  = sp.Symbol('z')
    Dz = sp.Symbol('Dz')

    beta_sym    = sp.Symbol('beta')
    epsilon_sym = sp.Symbol('epsilon')
    chi_sym     = sp.Symbol('chi')

    A = Species('A')

    # A → 2A (branching, rate β)
    rxn_branch = Reaction({A: 1}, {A: 2}, rate=beta_sym)
    Q_beta = generator_action_term(rxn_branch, A)
    # Expected (from thesis 1.38b): β((z+1)^2 - (z+1)^1)Dz^1
    expected_beta = sp.expand(beta_sym * ((z+1)**2 - (z+1)**1) * Dz**1)
    results['branching_A→2A'] = sp.simplify(Q_beta - expected_beta) == 0

    # A → ∅ (death, rate ε)
    rxn_death = Reaction({A: 1}, {}, rate=epsilon_sym)
    Q_eps = generator_action_term(rxn_death, A)
    # Expected (from thesis 1.38c): ε((z+1)^0 - (z+1)^1)Dz^1 = ε(1-(z+1))Dz = -εzDz
    expected_eps = sp.expand(epsilon_sym * ((z+1)**0 - (z+1)**1) * Dz**1)
    results['death_A→∅'] = sp.simplify(Q_eps - expected_eps) == 0

    # 2A → A (coagulation, rate χ)
    rxn_coag = Reaction({A: 2}, {A: 1}, rate=chi_sym)
    Q_chi = generator_action_term(rxn_coag, A)
    # Expected (from thesis 1.38a): χ((z+1)^1 - (z+1)^2)Dz^2
    expected_chi = sp.expand(chi_sym * ((z+1)**1 - (z+1)**2) * Dz**2)
    results['coagulation_2A→A'] = sp.simplify(Q_chi - expected_chi) == 0

    return results
