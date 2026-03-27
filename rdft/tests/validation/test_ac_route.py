"""
tests.validation.test_ac_route
==============================
Validate that the Analytic Combinatorics route reproduces known
critical exponents independently of the Doi-Peliti / RG calculation.

The central claim: singularity type of the generating function
determines the universality class. Both routes (RG and AC) arrive
at the same answer because they detect the same singularity.

Reference: Amarteifio (2026), Tutorial Parts III-IV
"""

import sympy as sp
from sympy import Rational, Symbol, exp, sqrt, pi, gamma
import pytest


# ================================================================== #
#  AC Layer: Lagrange equations                                       #
# ================================================================== #

class TestLagrangeEquations:
    """Test Lagrange inversion and singularity detection."""

    def test_sir_branch_point(self):
        """
        SIR epidemic: T* = 1/R₀, z* = e^{1-1/R₀}/R₀
        (Amarteifio Tutorial eq. 4.7-4.8)
        """
        from rdft.ac.lagrange import sir_epidemic
        R0 = Symbol('R_0', positive=True)
        sir = sir_epidemic(R0)
        T_star, z_star = sir.branch_point()

        assert sp.simplify(T_star - 1/R0) == 0, f"T* = {T_star}, expected 1/R₀"
        # z* = T*/φ(T*) = (1/R₀)/exp(R₀(1/R₀ - 1)) = (1/R₀)·exp(R₀-1)
        # At R₀=1 (criticality): z* = 1 (branch point on unit circle)
        z_at_1 = z_star.subs(R0, 1)
        assert abs(float(z_at_1) - 1.0) < 1e-10, \
            f"z*(R₀=1) = {z_at_1}, expected 1"
        # At R₀=2: z* = e/2 ≈ 1.359
        z_at_2 = float(z_star.subs(R0, 2))
        assert abs(z_at_2 - sp.E/2) < 1e-10, f"z*(R₀=2) = {z_at_2}"

    def test_sir_borel_distribution(self):
        """
        SIR coefficients should be the Borel distribution:
        [z^n]T = e^{-R₀n}(R₀n)^{n-1}/n!
        """
        from rdft.ac.lagrange import sir_epidemic
        R0 = Symbol('R_0', positive=True)
        sir = sir_epidemic(R0)
        coeffs = sir.coefficients(4)

        # n=1: e^{-R₀} · 1 / 1! = e^{-R₀}
        assert sp.simplify(coeffs[0] - exp(-R0)) == 0

        # n=2: e^{-2R₀} · (2R₀)^1 / 2! = R₀·e^{-2R₀}
        expected_2 = R0 * exp(-2*R0)
        assert sp.simplify(coeffs[1] - expected_2) == 0

    def test_sir_singularity_type(self):
        """SIR has a square-root branch point (→ n^{-3/2})."""
        from rdft.ac.lagrange import sir_epidemic
        sir = sir_epidemic()
        sing = sir.singularity_type()
        assert sing['type'] == 'square_root_branch'

    def test_first_passage_singularity(self):
        """
        First-passage GF for symmetric 1D walk has square-root branch.
        This gives the 2A→∅ exponent via AC.
        """
        from rdft.ac.lagrange import first_passage_1d
        fp = first_passage_1d()
        sing = fp.singularity_type()
        assert sing['type'] == 'square_root_branch'

    def test_first_passage_coefficients(self):
        """
        First-passage: F(z) = z(1+F²)/2
        Odd coefficients only (walk must take odd number of steps).
        [z^1]F = 1/2, [z^3]F = 1/8, [z^5]F = 1/16
        """
        from rdft.ac.lagrange import first_passage_1d
        fp = first_passage_1d()
        coeffs = fp.coefficients(5)

        assert coeffs[0] == Rational(1, 2)    # [z^1]
        assert coeffs[1] == 0                  # [z^2] = 0
        assert coeffs[2] == Rational(1, 8)     # [z^3]
        assert coeffs[3] == 0                  # [z^4] = 0
        assert coeffs[4] == Rational(1, 16)    # [z^5]

    def test_pair_annihilation_dse_branch(self):
        """
        DSE for 2A→∅: G = G₀(1 + λG²) has square-root branch.
        """
        from rdft.ac.lagrange import pair_annihilation_dse
        pa = pair_annihilation_dse()
        sing = pa.singularity_type()
        assert sing['type'] == 'square_root_branch'


# ================================================================== #
#  Transfer Theorem                                                    #
# ================================================================== #

class TestTransferTheorem:
    """Test the Flajolet-Odlyzko transfer theorem."""

    def test_square_root_gives_minus_three_halves(self):
        """Square-root branch (α=1/2) → [z^n] ~ n^{-3/2}."""
        from rdft.ac.transfer import Singularity
        sing = Singularity(z_star=sp.S.One, alpha=Rational(1, 2))
        asymp = sing.coefficient_asymptotics_simplified()
        assert asymp['power_law_exponent'] == Rational(-3, 2)

    def test_simple_pole_gives_exponential(self):
        """Simple pole (α=-1) → [z^n] ~ z*^{-n} (no power law)."""
        from rdft.ac.transfer import Singularity
        sing = Singularity(z_star=sp.S(2), alpha=-1)
        asymp = sing.coefficient_asymptotics_simplified()
        assert asymp['power_law_exponent'] == 0  # n^0 = constant

    def test_inverse_sqrt_gives_minus_half(self):
        """Inverse square-root (α=-1/2) → [z^n] ~ n^{-1/2}."""
        from rdft.ac.transfer import Singularity
        sing = Singularity(z_star=sp.S.One, alpha=Rational(-1, 2))
        asymp = sing.coefficient_asymptotics_simplified()
        assert asymp['power_law_exponent'] == Rational(-1, 2)

    def test_density_exponent_from_square_root(self):
        """
        Square-root branch → n^{-3/2} → P_surv ~ t^{-1/2} → ρ ~ t^{-1/2}
        This is the AC derivation of the 2A→∅ exponent.
        """
        from rdft.ac.transfer import Singularity
        sing = Singularity(z_star=sp.S.One, alpha=Rational(1, 2))
        dens = sing.density_exponent()
        assert dens['density_exponent_1d'] == Rational(-1, 2)
        assert dens['upper_critical_dimension'] == 2


# ================================================================== #
#  AC ↔ QFT Agreement                                                 #
# ================================================================== #

class TestACvsQFT:
    """
    Verify that the AC route and the RG route give the same exponents.
    This is the key validation: both routes detect the same singularity.
    """

    def test_pair_annihilation_ac_equals_rg(self):
        """
        2A→∅: AC gives α = -1/2 (from n^{-3/2} first-passage tail)
               RG gives α = d/2 with d=1 → α = 1/2
        The density decay is ρ ~ t^{-α} where α = 1/2 = d/2.
        """
        from rdft.ac.transfer import Singularity
        from rdft.rg.rg_functions import KnownResults

        # AC route
        sing = Singularity(z_star=sp.S.One, alpha=Rational(1, 2))
        dens = sing.density_exponent()
        ac_exponent = -dens['density_exponent_1d']  # positive exponent

        # RG route
        d = Symbol('d', positive=True)
        rg_result = KnownResults.pair_annihilation()
        rg_exponent = rg_result['alpha'].subs(d, 1)  # d=1

        assert ac_exponent == rg_exponent, \
            f"AC gives α={ac_exponent}, RG gives α={rg_exponent}"

    def test_sir_universality_class(self):
        """
        SIR at R₀=1 is in the same universality class as mean-field
        percolation: P(size=n) ~ n^{-3/2} (Borel distribution tail).
        """
        from rdft.ac.lagrange import sir_epidemic
        R0 = Symbol('R_0', positive=True)
        sir = sir_epidemic(R0)
        sing = sir.singularity_type()

        # At R₀=1, z*=1, pure power law
        T_star = sing['T_star']
        z_star = sing['z_star']
        assert sp.simplify(T_star.subs(R0, 1) - 1) == 0
        assert sp.simplify(z_star.subs(R0, 1) - 1) == 0


# ================================================================== #
#  Spectral Dimension                                                  #
# ================================================================== #

class TestSpectralDimension:
    """Test spectral dimension computation and substitution."""

    def test_1d_lattice(self):
        """1D periodic lattice should give d_s ≈ 1."""
        from rdft.graphs.spectral import SpectralDimension, hypercubic_lattice
        adj = hypercubic_lattice(1, 63)
        sd = SpectralDimension(adj)
        d_s = sd.spectral_dimension()
        assert abs(d_s - 1.0) < 0.2, f"1D lattice: d_s = {d_s:.2f}, expected ~1.0"

    def test_2d_lattice(self):
        """2D periodic lattice should give d_s ≈ 2."""
        from rdft.graphs.spectral import SpectralDimension, hypercubic_lattice
        adj = hypercubic_lattice(2, 10)
        sd = SpectralDimension(adj)
        d_s = sd.spectral_dimension()
        assert abs(d_s - 2.0) < 0.3, f"2D lattice: d_s = {d_s:.2f}, expected ~2.0"

    def test_d_substitution(self):
        """
        Replacing d → d_s in ρ ~ t^{-d/2} gives ρ ~ t^{-d_s/2}
        on arbitrary graphs.
        """
        from rdft.graphs.spectral import substitute_spectral_dimension
        d = Symbol('d', positive=True)
        d_s = Symbol('d_s', positive=True)

        expr = -d / 2   # density exponent
        result = substitute_spectral_dimension(expr, d_s, d)
        assert result == -d_s / 2

    def test_brw_scaling_below_dc(self):
        """BRW scaling: ⟨a^p⟩ ~ t^{(pd_s-2)/2} for d_s < 4."""
        from rdft.graphs.spectral import brw_scaling_exponents
        d_s = Symbol('d_s', positive=True)
        p = Symbol('p', positive=True, integer=True)

        exponents = brw_scaling_exponents(d_s)
        time_exp = exponents['below_dc']['time_exponent']
        assert sp.simplify(time_exp - (p * d_s - 2) / 2) == 0


# ================================================================== #
#  Diagram Generation                                                  #
# ================================================================== #

class TestDiagramGeneration:
    """Test Feynman diagram generation from Wick contractions."""

    def test_pair_annihilation_has_one_loop_diagrams(self):
        """2A→∅ should produce at least one 1PI diagram at one loop."""
        from rdft.core.expansion import FeynmanExpansion
        from rdft.core.reaction_network import ReactionNetwork

        net = ReactionNetwork.pair_annihilation()
        exp = FeynmanExpansion(net, max_loops=1)
        results = exp.expand()

        assert 1 in results, "No one-loop diagrams found"
        assert len(results[1]) > 0, "Empty one-loop diagram list"

    def test_gribov_has_one_loop_diagrams(self):
        """Gribov process should produce 1PI diagrams at one loop."""
        from rdft.core.expansion import FeynmanExpansion
        from rdft.core.reaction_network import ReactionNetwork

        net = ReactionNetwork.gribov()
        exp = FeynmanExpansion(net, max_loops=1)
        results = exp.expand()

        assert 1 in results, "No one-loop diagrams for Gribov"
        # Thesis states ~7 distinct loop integrals for BWS
        print(f"Gribov one-loop: {len(results[1])} distinct diagrams")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
