"""
rdft.core.reaction_network
==========================
Chemical reaction network (CRN) data structures.

A CRN is a compound Poisson process defined by a set of reactions
kA → lA (generalised to multiple species). This follows directly
from Definition 1 of Amarteifio (2019), with the stoichiometric
matrix S_{kl} as the primary encoding.

Each reaction is specified as:
  - reactants: {species: count} — the k-side
  - products:  {species: count} — the l-side
  - rate:      symbolic or numeric rate parameter
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sympy as sp


@dataclass(frozen=True)
class Species:
    """A particle species in the reaction network."""
    name: str
    diffusion_constant: sp.Expr = sp.Symbol('D')

    def __repr__(self) -> str:
        return self.name

    @property
    def symbol(self) -> sp.Symbol:
        return sp.Symbol(self.name)


@dataclass
class Reaction:
    """
    A single elementary reaction k_1 A + k_2 B + ... → l_1 A + l_2 B + ...

    The stoichiometric vector is:
        s_i = l_i - k_i  (net change in species i)

    For a single-species reaction kA → lA this reduces to Amarteifio (1.8).

    Parameters
    ----------
    reactants : dict mapping Species → stoichiometric count (k side)
    products  : dict mapping Species → stoichiometric count (l side)
    rate      : symbolic rate parameter λ
    name      : optional human-readable label

    Examples
    --------
    Pair annihilation 2A → ∅:
        Reaction({A: 2}, {}, rate=sp.Symbol('lambda'))

    Coagulation 2A → A:
        Reaction({A: 2}, {A: 1}, rate=sp.Symbol('chi'))

    Branching A → 2A:
        Reaction({A: 1}, {A: 2}, rate=sp.Symbol('beta'))
    """
    reactants: Dict[Species, int]
    products:  Dict[Species, int]
    rate:      sp.Expr
    name:      Optional[str] = None

    def __post_init__(self):
        # Fill missing species with 0
        all_species = set(self.reactants) | set(self.products)
        for sp_ in all_species:
            self.reactants.setdefault(sp_, 0)
            self.products.setdefault(sp_, 0)
        if self.name is None:
            lhs = ' + '.join(f'{v}{k}' for k, v in self.reactants.items() if v > 0)
            rhs = ' + '.join(f'{v}{k}' for k, v in self.products.items() if v > 0) or '∅'
            self.name = f'{lhs} → {rhs}'

    @property
    def all_species(self) -> List[Species]:
        return sorted(set(self.reactants) | set(self.products), key=lambda s: s.name)

    def k(self, species: Species) -> int:
        """Number of input particles of this species (reactant stoichiometry)."""
        return self.reactants.get(species, 0)

    def l(self, species: Species) -> int:
        """Number of output particles of this species (product stoichiometry)."""
        return self.products.get(species, 0)

    def net_change(self, species: Species) -> int:
        """Net stoichiometric change: l - k."""
        return self.l(species) - self.k(species)

    def is_single_species(self) -> bool:
        return len(self.all_species) == 1

    def total_reactants(self) -> int:
        return sum(self.reactants.values())

    def total_products(self) -> int:
        return sum(self.products.values())

    def __repr__(self) -> str:
        return self.name


@dataclass
class ReactionNetwork:
    """
    A chemical reaction-diffusion system S(C, G, W) as in Amarteifio (2019) Def. 2.

    A ReactionNetwork C consists of a set of species and a set of reactions.
    The geometry (graph G and jump distribution W) is handled separately in
    rdft.graphs.

    Parameters
    ----------
    species   : list of Species objects
    reactions : list of Reaction objects
    name      : optional label for the network

    Examples
    --------
    Pair annihilation:
        A = Species('A', D)
        net = ReactionNetwork([A], [Reaction({A:2}, {}, rate=lam)])

    Gribov process (A → 2A, A → ∅):
        net = ReactionNetwork.gribov()
    """
    species:   List[Species]
    reactions: List[Reaction]
    name:      str = 'CRN'

    def __post_init__(self):
        # Validate: every species in reactions must be in the species list
        declared = set(self.species)
        for rxn in self.reactions:
            for sp_ in rxn.all_species:
                if sp_ not in declared:
                    raise ValueError(
                        f"Species {sp_} in reaction '{rxn}' not declared in network"
                    )

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    def stoichiometric_matrix(self) -> sp.Matrix:
        """
        Returns the stoichiometric matrix S of shape (n_reactions, 2, n_species).

        For each reaction r and species i:
            S[r, 0, i] = k_i  (reactant count)
            S[r, 1, i] = l_i  (product count)

        This is the tensor form used in Amarteifio (1.37) for the Gribov process.
        """
        n_r = self.n_reactions
        n_s = self.n_species
        k_mat = sp.zeros(n_r, n_s)
        l_mat = sp.zeros(n_r, n_s)
        for r, rxn in enumerate(self.reactions):
            for s, sp_ in enumerate(self.species):
                k_mat[r, s] = rxn.k(sp_)
                l_mat[r, s] = rxn.l(sp_)
        return k_mat, l_mat

    def summary(self) -> str:
        lines = [f'ReactionNetwork: {self.name}']
        lines.append(f'  Species: {[str(s) for s in self.species]}')
        for rxn in self.reactions:
            lines.append(f'  {rxn}  (rate={rxn.rate})')
        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    #  Generic constructor from stoichiometry matrix                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_stoichiometry(cls, S, species_names: Optional[List[str]] = None,
                            rates: Optional[List[sp.Expr]] = None,
                            name: str = 'Custom CRN') -> 'ReactionNetwork':
        """
        Build a ReactionNetwork from a raw stoichiometry matrix.

        Parameters
        ----------
        S : list of [k_1, ..., k_m, l_1, ..., l_m] rows, or list of (k, l) tuples.
            Each row is one reaction.
            For single-species: S = [[k, l], [k, l], ...]
            For multi-species with m species: each row has 2m entries
            [k_1, ..., k_m, l_1, ..., l_m]
        species_names : list of species name strings (default: A, B, C, ...)
        rates : list of rate symbols (default: auto-generated r_0, r_1, ...)
        name : network name

        Examples
        --------
        Pair annihilation 2A→∅:
            ReactionNetwork.from_stoichiometry([[2, 0]])

        Gribov process (A→2A, A→∅, 2A→A):
            ReactionNetwork.from_stoichiometry([[1,2], [1,0], [2,1]])

        Two-species A+B→∅:
            ReactionNetwork.from_stoichiometry([[1,1,0,0]])
            # k_A=1, k_B=1, l_A=0, l_B=0
        """
        S_list = [list(row) for row in S]
        n_rxn = len(S_list)
        row_len = len(S_list[0])

        # Determine number of species
        if row_len % 2 != 0:
            raise ValueError(f"Row length {row_len} must be even (k_1,...,k_m,l_1,...,l_m)")
        n_species = row_len // 2

        # Species
        if species_names is None:
            species_names = [chr(65 + i) for i in range(n_species)]  # A, B, C, ...
        species_list = [Species(name) for name in species_names]

        # Rates
        if rates is None:
            if n_rxn == 1:
                rates = [sp.Symbol('lambda', positive=True)]
            else:
                rates = [sp.Symbol(f'r_{i}', positive=True) for i in range(n_rxn)]

        # Build reactions
        reactions = []
        for r, row in enumerate(S_list):
            k_vec = row[:n_species]
            l_vec = row[n_species:]

            reactants = {species_list[i]: k_vec[i] for i in range(n_species) if k_vec[i] > 0}
            products  = {species_list[i]: l_vec[i] for i in range(n_species) if l_vec[i] > 0}

            reactions.append(Reaction(reactants, products, rate=rates[r]))

        return cls(species_list, reactions, name=name)

    # ------------------------------------------------------------------ #
    #  Standard networks from the literature                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def pure_death(cls, rate: Optional[sp.Expr] = None) -> 'ReactionNetwork':
        """A → ∅ with rate δ."""
        A = Species('A')
        δ = rate or sp.Symbol('delta', positive=True)
        return cls([A], [Reaction({A: 1}, {}, rate=δ)], name='Pure Death A→∅')

    @classmethod
    def birth_death(cls, birth_rate=None, death_rate=None) -> 'ReactionNetwork':
        """A → 2A (rate β), A → ∅ (rate δ)."""
        A = Species('A')
        β = birth_rate or sp.Symbol('beta', positive=True)
        δ = death_rate or sp.Symbol('delta', positive=True)
        return cls(
            [A],
            [Reaction({A: 1}, {A: 2}, rate=β, name='A→2A'),
             Reaction({A: 1}, {},     rate=δ, name='A→∅')],
            name='Birth-Death'
        )

    @classmethod
    def pair_annihilation(cls, rate=None) -> 'ReactionNetwork':
        """2A → ∅ with rate λ. Lee (1994)."""
        A = Species('A')
        λ = rate or sp.Symbol('lambda', positive=True)
        return cls([A], [Reaction({A: 2}, {}, rate=λ)], name='Pair Annihilation 2A→∅')

    @classmethod
    def coagulation(cls, rate=None) -> 'ReactionNetwork':
        """2A → A with rate χ. Same universality class as pair annihilation."""
        A = Species('A')
        χ = rate or sp.Symbol('chi', positive=True)
        return cls([A], [Reaction({A: 2}, {A: 1}, rate=χ)], name='Coagulation 2A→A')

    @classmethod
    def gribov(cls, branch_rate=None, death_rate=None, annihil_rate=None) -> 'ReactionNetwork':
        """
        Gribov process (branching random walk):
            A → 2A  (rate β)
            A → ∅   (rate ε)
            2A → A  (rate χ)

        This is the model studied in Amarteifio (2019) Chapter 3.
        The stoichiometric matrix is eq. (1.37) of the thesis.
        """
        A = Species('A')
        β = branch_rate  or sp.Symbol('beta',    positive=True)
        ε = death_rate   or sp.Symbol('epsilon', positive=True)
        χ = annihil_rate or sp.Symbol('chi',     positive=True)
        return cls(
            [A],
            [Reaction({A: 1}, {A: 2}, rate=β, name='A→2A'),
             Reaction({A: 1}, {},     rate=ε, name='A→∅'),
             Reaction({A: 2}, {A: 1}, rate=χ, name='2A→A')],
            name='Gribov Process'
        )

    @classmethod
    def brw_full(cls) -> 'ReactionNetwork':
        """
        Full Branching Random Walk with two species (Amarteifio 2019, §3.2.2).

        Species A: active walkers (branching, death, coagulation)
        Species B: immobile tracers (deposited by walkers, site exclusion)

        A-sector reactions (the superprocess, Eq. 3.15):
            A → 2A  (branching, rate β)
            A → ∅   (death, rate ε)
            2A → A  (coagulation, rate χ)

        Transmutation (tracer deposition with carrying capacity, Eq. 3.17-3.21):
            A deposits B only if no B present (site exclusion).
            Rate: Λ (transmutation rate).

        The carrying capacity trick ρ(n_b) = n_a(c - n_b)|_{c=1} generates
        6 interaction terms in the Doi-shifted action (Eq. 3.21).

        The transmutation vertices cannot be generated by the standard
        generator formula Q = λ((z+1)^l - (z+1)^k)∂z^k because the site
        exclusion is a constraint, not a simple reaction. Instead, the
        Liouvillian terms are constructed directly from Eq. (3.19)-(3.21).
        """
        A = Species('A')
        B = Species('B', sp.Symbol('D_B', positive=True))
        β = sp.Symbol('beta', positive=True)
        ε = sp.Symbol('epsilon', positive=True)
        χ = sp.Symbol('chi', positive=True)

        # A-sector reactions (standard generator formula applies)
        rxn_branch = Reaction({A: 1}, {A: 2}, rate=β, name='A→2A')
        rxn_death  = Reaction({A: 1}, {},     rate=ε, name='A→∅')
        rxn_coag   = Reaction({A: 2}, {A: 1}, rate=χ, name='2A→A')

        # Transmutation: A deposits B with site exclusion
        # This is a simplified transmutation without carrying capacity;
        # the full 6 terms from Eq. (3.21) are added as explicit
        # Liouvillian terms via the brw_transmutation_vertices() function
        # in rdft.core.generators.
        Λ = sp.Symbol('Lambda', positive=True)
        rxn_deposit = Reaction({A: 1}, {A: 1, B: 1}, rate=Λ, name='A→A+B')

        return cls(
            [A, B],
            [rxn_branch, rxn_death, rxn_coag, rxn_deposit],
            name='BRW (full two-species)'
        )

    @classmethod
    def two_species_annihilation(cls) -> 'ReactionNetwork':
        """A + B → ∅. Lee-Cardy (1995)."""
        A = Species('A')
        B = Species('B', sp.Symbol('D_B', positive=True))
        λ = sp.Symbol('lambda', positive=True)
        return cls([A, B], [Reaction({A: 1, B: 1}, {}, rate=λ)],
                   name='Two-species A+B→∅')

    @classmethod
    def contact_process(cls) -> 'ReactionNetwork':
        """
        Contact process: A → 2A (rate λ), 2A → ∅ (rate μ).
        Directed percolation universality class.
        """
        A = Species('A')
        λ = sp.Symbol('lambda', positive=True)
        μ = sp.Symbol('mu',     positive=True)
        return cls(
            [A],
            [Reaction({A: 1}, {A: 2}, rate=λ, name='A→2A'),
             Reaction({A: 2}, {},     rate=μ, name='2A→∅')],
            name='Contact Process'
        )

    # ------------------------------------------------------------------ #
    #  Extended CRN library for testing the AC pipeline                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def triplet_annihilation(cls, rate=None) -> 'ReactionNetwork':
        """
        3A → ∅ with rate λ.
        d_c = 2/(3-1) = 1. Mean-field in all physical dimensions d ≥ 1.
        Density decay: n(t) ~ t^{-1/2} (mean-field) for d ≥ 1.
        Below d_c (fractals): n(t) ~ t^{-d/2}.
        Ref: Lee, J. Phys. A 27, 2633 (1994).
        """
        A = Species('A')
        λ = rate or sp.Symbol('lambda', positive=True)
        return cls([A], [Reaction({A: 3}, {}, rate=λ, name='3A→∅')],
                   name='Triplet Annihilation 3A→∅')

    @classmethod
    def k_particle_annihilation(cls, k: int = 2, rate=None) -> 'ReactionNetwork':
        """
        kA → ∅ with rate λ.
        d_c = 2/(k-1). For d < d_c: n(t) ~ t^{-d/2} (exact to all orders).
        Ref: Lee, J. Phys. A 27, 2633 (1994).
        """
        A = Species('A')
        λ = rate or sp.Symbol('lambda', positive=True)
        return cls([A], [Reaction({A: k}, {}, rate=λ, name=f'{k}A→∅')],
                   name=f'{k}-Particle Annihilation {k}A→∅')

    @classmethod
    def barw_even(cls, branch_rate=None, annihil_rate=None) -> 'ReactionNetwork':
        """
        Branching-Annihilating Random Walk with even parity conservation:
            A → 3A (rate σ), 2A → ∅ (rate λ)
        Net offspring change is +2 (even), so parity of particle number is conserved.
        Universality class: Parity-Conserving (PC), NOT directed percolation.
        d_c = 2 (formally), but ε-expansion fails to reach the PC fixed point.
        Exponents in d=1 (numerical only): β ≈ 0.92, ν_⊥ ≈ 1.83, z ≈ 1.77.
        Ref: Cardy & Täuber, PRL 77, 4780 (1996).
        """
        A = Species('A')
        σ = branch_rate  or sp.Symbol('sigma',  positive=True)
        λ = annihil_rate or sp.Symbol('lambda', positive=True)
        return cls(
            [A],
            [Reaction({A: 1}, {A: 3}, rate=σ, name='A→3A'),
             Reaction({A: 2}, {},     rate=λ, name='2A→∅')],
            name='BARW-even (PC class)'
        )

    @classmethod
    def barw_odd(cls, branch_rate=None, annihil_rate=None) -> 'ReactionNetwork':
        """
        Branching-Annihilating Random Walk with odd offspring (DP class):
            A → 2A (rate σ), 2A → ∅ (rate λ)
        Net offspring change is +1 (odd), parity NOT conserved.
        Universality class: Directed Percolation (DP).
        d_c = 4. Same as contact process.
        Ref: Cardy & Täuber, J. Stat. Phys. 90, 1 (1998).
        """
        A = Species('A')
        σ = branch_rate  or sp.Symbol('sigma',  positive=True)
        λ = annihil_rate or sp.Symbol('lambda', positive=True)
        return cls(
            [A],
            [Reaction({A: 1}, {A: 2}, rate=σ, name='A→2A'),
             Reaction({A: 2}, {},     rate=λ, name='2A→∅')],
            name='BARW-odd (DP class)'
        )

    @classmethod
    def pcpd(cls, branch_rate=None, annihil_rate=None) -> 'ReactionNetwork':
        """
        Pair Contact Process with Diffusion (PCPD):
            2A → 3A (rate σ), 2A → ∅ (rate λ)
        Both reactions require a PAIR to meet. Controversial universality class
        — debated for 20+ years. Not clearly DP or PC.
        Numerical exponents d=1: β ≈ 0.58-0.64 (conflicting studies).
        Ref: Hinrichsen, PRE 63, 016109 (2001).
        """
        A = Species('A')
        σ = branch_rate  or sp.Symbol('sigma',  positive=True)
        λ = annihil_rate or sp.Symbol('lambda', positive=True)
        return cls(
            [A],
            [Reaction({A: 2}, {A: 3}, rate=σ, name='2A→3A'),
             Reaction({A: 2}, {},     rate=λ, name='2A→∅')],
            name='PCPD'
        )

    @classmethod
    def schlogl_second(cls, branch_rate=None, death_rate=None) -> 'ReactionNetwork':
        """
        Second Schlögl model: 2A → 3A (rate σ), A → ∅ (rate μ).
        Universality class: Directed Percolation (DP). d_c = 4.
        Different CRN from Gribov but same universality class.
        Ref: Schlögl, Z. Phys. 253, 147 (1972); Grassberger, Z. Phys. B 47, 365 (1982).
        """
        A = Species('A')
        σ = branch_rate or sp.Symbol('sigma', positive=True)
        μ = death_rate  or sp.Symbol('mu',    positive=True)
        return cls(
            [A],
            [Reaction({A: 2}, {A: 3}, rate=σ, name='2A→3A'),
             Reaction({A: 1}, {},     rate=μ, name='A→∅')],
            name='Schlögl II (DP class)'
        )

    @classmethod
    def schlogl_first(cls) -> 'ReactionNetwork':
        """
        First Schlögl model (simplified, with particle reservoir):
            ∅ → A (rate α), A → ∅ (rate δ),
            2A → 3A (rate σ), 3A → 2A (rate λ)
        At the critical point: Ising universality class. d_c = 4.
        Ref: Schlögl, Z. Phys. 253, 147 (1972).
        """
        A = Species('A')
        α = sp.Symbol('alpha',  positive=True)
        δ = sp.Symbol('delta',  positive=True)
        σ = sp.Symbol('sigma',  positive=True)
        λ = sp.Symbol('lambda', positive=True)
        return cls(
            [A],
            [Reaction({},     {A: 1}, rate=α, name='∅→A'),
             Reaction({A: 1}, {},     rate=δ, name='A→∅'),
             Reaction({A: 2}, {A: 3}, rate=σ, name='2A→3A'),
             Reaction({A: 3}, {A: 2}, rate=λ, name='3A→2A')],
            name='Schlögl I (Ising class)'
        )

    @classmethod
    def lotka_volterra(cls) -> 'ReactionNetwork':
        """
        Lotka-Volterra predator-prey:
            A → 2A (prey reproduction, rate σ)
            A + B → 2B (predation, rate λ)
            B → ∅ (predator death, rate μ)
        Prey extinction transition: DP universality class, d_c = 4.
        Ref: Mobilia, Georgiev, Täuber, J. Stat. Phys. 128, 447 (2007).
        """
        A = Species('A')  # prey
        B = Species('B', sp.Symbol('D_B', positive=True))  # predator
        σ = sp.Symbol('sigma',  positive=True)
        λ = sp.Symbol('lambda', positive=True)
        μ = sp.Symbol('mu',     positive=True)
        return cls(
            [A, B],
            [Reaction({A: 1}, {A: 2},         rate=σ, name='A→2A'),
             Reaction({A: 1, B: 1}, {B: 2},   rate=λ, name='A+B→2B'),
             Reaction({B: 1}, {},              rate=μ, name='B→∅')],
            name='Lotka-Volterra'
        )

    @classmethod
    def reversible_annihilation(cls, forward_rate=None,
                                 backward_rate=None) -> 'ReactionNetwork':
        """
        Reversible pair annihilation: 2A → C (rate λ), C → 2A (rate σ).
        Relaxation to equilibrium: δn(t) ~ t^{-d/2}, same exponent as
        irreversible 2A → ∅. d_c = 2.
        Ref: Rey & Cardy, J. Phys. A 32, 1585 (1999).
        """
        A = Species('A')
        C = Species('C', sp.Symbol('D_C', positive=True))
        λ = forward_rate  or sp.Symbol('lambda', positive=True)
        σ = backward_rate or sp.Symbol('sigma',  positive=True)
        return cls(
            [A, C],
            [Reaction({A: 2}, {C: 1},   rate=λ, name='2A→C'),
             Reaction({C: 1}, {A: 2},   rate=σ, name='C→2A')],
            name='Reversible 2A⇌C'
        )
