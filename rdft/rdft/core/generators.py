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
        Multi-species generator.

        For a reaction k_1 A + k_2 B → l_1 A + l_2 B with rate λ,
        the generator in the tensor product Weyl algebra is:

            Q = λ · Π_i ((z_i+1)^{l_i} - (z_i+1)^{k_i}) · Dz_i^{k_i}

        This follows from the independence of species in the Poisson ansatz.
        """
        z_syms  = {sp_: sp.Symbol(f'z_{sp_.name}')  for sp_ in self.network.species}
        Dz_syms = {sp_: sp.Symbol(f'Dz_{sp_.name}') for sp_ in self.network.species}

        result = rxn.rate
        for sp_ in rxn.all_species:
            k = rxn.k(sp_)
            l = rxn.l(sp_)
            z  = z_syms[sp_]
            Dz = Dz_syms[sp_]
            result *= ((z + 1)**l - (z + 1)**k) * Dz**k

        return sp.expand(result)

    @property
    def vertices(self) -> Dict[Tuple, sp.Expr]:
        """
        Extract Feynman vertices from the action.

        Each term of the form λ·z^m·Dz^n corresponds to a vertex with:
          - m outgoing lines (φ̃ fields, creator fields)
          - n incoming lines (φ fields, annihilator fields)
          - coupling constant λ

        Returns dict: (m_out, n_in) → coupling expr

        These vertices are the "primitive corollas" that seed the shuffle product.
        """
        z  = sp.Symbol('z')
        Dz = sp.Symbol('Dz')

        vertices = {}
        poly = sp.Poly(self.total, z, Dz) if self.network.n_species == 1 else None

        if poly is not None:
            for monom, coeff in zip(poly.monoms(), poly.coeffs()):
                m_out, n_in = monom   # powers of z, Dz
                if coeff != 0:
                    key = (m_out, n_in)
                    vertices[key] = vertices.get(key, sp.S.Zero) + coeff

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
