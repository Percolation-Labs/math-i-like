"""
rdft.rg.rg_functions
====================
Renormalization group functions β(λ) and extraction of critical exponents.

The Callan-Symanzik β-function controls how the coupling λ flows under
a change of renormalization scale μ:

    β(λ) = μ ∂λ/∂μ|_{λ_0 fixed}
          = -ελ + b_1 λ² + b_2 λ³ + ...

where ε = d_c - d is the deviation from upper critical dimension.

The coefficients b_n are extracted from the poles of the Z-factor:
    Z_λ = 1 + Σ_n (z_n/ε^n) λ^n
    b_n = n · z_1(n)   (one-loop: b_1 from residue of 1/ε pole)

Critical exponents at the non-trivial fixed point β(λ*) = 0:
    λ* = ε/b_1 + O(ε²)   (Wilson-Fisher fixed point)

At the fixed point:
    ν = -1/(β'(λ*))       correlation length exponent
    η = η(λ*)              anomalous dimension
    α = d·ν/2 - ...        density decay exponent (process-specific)

For reaction-diffusion:
    ρ(t) ~ t^{-α}
    α = d/2 for 2A→∅ (exact, all orders)

Mathematical reference:
    Lee (1994) J. Phys. A 27:2633
    Täuber-Howard-Vollmayr-Lee (2005) J. Phys. A 38:R79
    Amarteifio (2019) §3.1
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import sympy as sp
import numpy as np


class RGFunctions:
    """
    Renormalization group functions for a reaction-diffusion process.

    Computes β(λ), η(λ), and other Wilson functions from the Z-factors.

    Parameters
    ----------
    z_lambda : sympy expr for Z_λ as a series in λ (and possibly 1/ε)
    z_phi    : sympy expr for Z_φ (field renormalization)
    d_c      : upper critical dimension (sympy expr or number)
    coupling : the coupling symbol λ
    """

    def __init__(self,
                 z_lambda: sp.Expr,
                 z_phi:    Optional[sp.Expr] = None,
                 d_c:      sp.Expr = sp.Integer(2),
                 coupling: Optional[sp.Symbol] = None):

        self.Z_lambda = z_lambda
        self.Z_phi    = z_phi or sp.Integer(1)
        self.d_c      = d_c
        self.lam      = coupling or sp.Symbol('lambda', positive=True)

        self.eps = sp.Symbol('epsilon', positive=True)
        self._beta: Optional[sp.Expr] = None
        self._eta:  Optional[sp.Expr] = None

    # ------------------------------------------------------------------ #
    #  Beta function                                                        #
    # ------------------------------------------------------------------ #

    @property
    def beta(self) -> sp.Expr:
        """
        β(λ) = μ ∂λ/∂μ = -ελ + β_int(λ)

        Extracted from Z_λ via the standard Callan-Symanzik relation:
            β(λ) = -ελ · ∂(ln Z_λ)/∂λ · 1/(1 + λ ∂(ln Z_λ)/∂λ)
            (At one loop: β ≈ -ελ - λ² ∂(ln Z_λ)/∂λ at leading order)

        For Z_λ = 1 + z_1 λ/ε + O(λ²):
            β = -ελ + z_1 λ²  + O(ε, λ³)

        This is the standard result used in Lee (1994) and Täuber+ (2005).
        """
        if self._beta is not None:
            return self._beta

        lam = self.lam
        eps = self.eps

        # Expand Z_λ as series in λ
        Z = sp.series(self.Z_lambda, lam, 0, 3)

        # Extract pole coefficients of Z_λ
        Z_expanded = sp.expand(Z)

        # β = -ελ + Σ_n b_n λ^{n+1}
        # At one loop: b_1 = residue of 1/ε pole in Z_λ at O(λ)
        # β(λ) ≈ -ελ + b_1 λ²

        # Method: β = μ ∂μ λ, where λ_0 = μ^ε Z_λ^{-1} λ
        # Taking μ ∂μ at fixed λ_0:
        # 0 = ε + β(λ) ∂/∂λ [ln(Z_λ)] + β(λ)/λ
        # → β(λ)[1/λ + ∂_λ ln Z] = -ε
        # → β(λ) = -ελ / (1 + λ ∂_λ ln Z_λ)

        # Compute ∂_λ ln Z_λ at leading order in λ
        ln_Z = sp.series(sp.log(Z), lam, 0, 3)
        d_ln_Z = sp.diff(ln_Z, lam)

        # Denominator: 1 + λ ∂_λ ln Z
        denom = sp.series(1 + lam * d_ln_Z, lam, 0, 3)

        # β = -ελ / denom
        beta_raw = sp.series(-eps * lam / denom, lam, 0, 4)

        # Extract leading terms (keep only ε^0 and ε^1)
        # In practice: β = -ελ + b_1(ε=0)λ² + O(ελ², λ³)
        self._beta = sp.expand(beta_raw.removeO())
        return self._beta

    def beta_coefficients(self, n_loops: int = 2) -> List[sp.Expr]:
        """
        Return [b_1, b_2, ..., b_{n_loops}] where β = -ελ + Σ b_n λ^{n+1}.
        """
        lam = self.lam
        beta_series = sp.series(self.beta + self.eps * lam, lam, 0, n_loops + 2)
        coeffs = []
        for n in range(1, n_loops + 1):
            c = beta_series.coeff(lam, n + 1)
            coeffs.append(sp.simplify(c))
        return coeffs

    # ------------------------------------------------------------------ #
    #  Anomalous dimension                                                  #
    # ------------------------------------------------------------------ #

    @property
    def eta(self) -> sp.Expr:
        """
        Anomalous dimension η(λ) = μ ∂(ln Z_φ)/∂μ.

        At one loop: η = γ_φ(λ) = -b_φ λ for some coefficient b_φ.
        """
        if self._eta is not None:
            return self._eta

        lam = self.lam
        ln_Zphi = sp.series(sp.log(self.Z_phi), lam, 0, 3)
        # η = β(λ) · ∂_λ ln Z_φ
        eta_raw = sp.series(self.beta * sp.diff(ln_Zphi, lam), lam, 0, 3)
        self._eta = sp.expand(eta_raw.removeO())
        return self._eta

    # ------------------------------------------------------------------ #
    #  Fixed points                                                         #
    # ------------------------------------------------------------------ #

    def fixed_points(self) -> Dict[str, sp.Expr]:
        """
        Find fixed points β(λ*) = 0.

        Returns dict:
          'gaussian'    : λ* = 0 (always a fixed point)
          'wilson_fisher': λ* at first order in ε
        """
        lam = self.lam
        eps = self.eps

        beta = self.beta

        # Gaussian: λ* = 0
        fps = {'gaussian': sp.S.Zero}

        # Non-trivial: solve β = 0 perturbatively in ε
        # β = -ελ + b_1 λ² + ... = 0
        # → λ* = ε/b_1 + O(ε²)
        b_coeffs = self.beta_coefficients(n_loops=1)
        if b_coeffs:
            b1 = b_coeffs[0]
            if b1 != 0:
                lam_star_1loop = sp.simplify(eps / b1)
                fps['wilson_fisher_1loop'] = lam_star_1loop

        # Two-loop correction if available
        if len(b_coeffs) >= 2:
            b1, b2 = b_coeffs[:2]
            if b1 != 0 and b2 != 0:
                # β = -ελ + b1λ² + b2λ³ = 0
                # λ(1 - b1λ/ε - b2λ²/ε) = 0 ... solve perturbatively
                lam_star_2loop = sp.simplify(
                    eps/b1 * (1 - b2/b1**2 * eps)
                )
                fps['wilson_fisher_2loop'] = lam_star_2loop

        return fps

    def critical_exponents(self, fixed_point_key: str = 'wilson_fisher_1loop') -> Dict[str, sp.Expr]:
        """
        Compute critical exponents at a fixed point.

        Returns:
          nu    : correlation length exponent ν = -1/β'(λ*)
          eta   : anomalous dimension η(λ*)
          z_dyn : dynamical exponent z = 2 - η (for diffusive dynamics)
        """
        fps = self.fixed_points()
        if fixed_point_key not in fps:
            return {}

        lam = self.lam
        lam_star = fps[fixed_point_key]

        # ν = -1/β'(λ*)
        beta_prime = sp.diff(self.beta, lam)
        beta_prime_at_star = sp.simplify(beta_prime.subs(lam, lam_star))

        if beta_prime_at_star == 0:
            nu = sp.oo
        else:
            nu = sp.simplify(-1 / beta_prime_at_star)

        # η at fixed point
        eta_at_star = sp.simplify(self.eta.subs(lam, lam_star))

        # Dynamic exponent for diffusive systems: z = 2 - η
        z_dyn = 2 - eta_at_star

        return {
            'lambda_star': lam_star,
            'nu': nu,
            'eta': eta_at_star,
            'z': z_dyn,
        }

    def density_exponent(self, process: str = 'annihilation') -> sp.Expr:
        """
        Compute the density decay exponent α where ρ(t) ~ t^{-α}.

        For pair annihilation 2A → ∅ (Lee 1994):
            α = d/2 (exact to all orders)
            This follows because the Ward identity cancels the quartic
            vertex, making the exponent exact.

        For coagulation 2A → A:
            Same universality class: α = d/2

        For branching-annihilation (DP class):
            α computed from ν, η, z via scaling relations.
        """
        d = sp.Symbol('d', positive=True)
        eps = self.eps

        if process in ('annihilation', 'coagulation', '2A_zero', '2A_A'):
            # Exact result: Lee (1994)
            return d / 2

        elif process in ('directed_percolation', 'DP', 'contact'):
            # DP scaling: α = d/(2ν_⊥ z) via hyperscaling
            exps = self.critical_exponents()
            if 'nu' in exps and 'z' in exps:
                return sp.simplify(d / (2 * exps['nu'] * exps['z']))
            return sp.Symbol('alpha_DP')

        elif process == 'BWS':
            # Branching Wiener Sausage (Amarteifio Ch.3)
            # Scaling of distinct sites visited: V(t) ~ t^{d/4} for d < d_c = 4
            return d / 4

        else:
            return sp.Symbol('alpha')

    def upper_critical_dimension(self) -> sp.Expr:
        """
        The upper critical dimension d_c where mean-field breaks down.

        Determined by: σ_{d_c}(G) = 0, i.e. [coupling] = 0 at d_c.

        For kA → ∅: d_c = 2/(k-1)
        For BWS:     d_c = 4
        For A+B→∅:   d_c = 4
        """
        return self.d_c

    def summary(self, n_loops: int = 1) -> str:
        lines = ['RG Functions Summary']
        lines.append(f'  d_c = {self.d_c}')
        lines.append(f'  β(λ) = {sp.simplify(self.beta)}')
        lines.append(f'  η(λ) = {sp.simplify(self.eta)}')

        fps = self.fixed_points()
        for name, lam_star in fps.items():
            lines.append(f'  Fixed point ({name}): λ* = {lam_star}')

        exps = self.critical_exponents()
        for name, val in exps.items():
            lines.append(f'  {name} = {sp.simplify(val)}')

        return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Known results for validation                                         #
# ------------------------------------------------------------------ #

class KnownResults:
    """
    Reference values from the literature for validation.

    All results are expressed symbolically in ε = d_c - d.
    """

    @staticmethod
    def pair_annihilation(d: sp.Expr = None) -> Dict[str, sp.Expr]:
        """
        2A → ∅: Lee (1994) J. Phys. A 27:2633.

        - d_c = 2
        - ρ(t) ~ t^{-d/2} (exact, all orders)
        - β(λ) = -ελ + (1/8πD) λ² (one loop)
        - Exponent α = d/2 is exact (Ward identity)
        """
        if d is None:
            d = sp.Symbol('d', positive=True)
        D = sp.Symbol('D', positive=True)
        eps = sp.Symbol('epsilon', positive=True)

        return {
            'd_c': sp.Integer(2),
            'alpha': d / 2,
            'beta_1loop_coeff': sp.Rational(1, 8) / (sp.pi * D),
            'lambda_star': 8 * sp.pi * D * eps,
            'exact': True,
            'reference': 'Lee (1994) J. Phys. A 27:2633',
        }

    @staticmethod
    def coagulation(d: sp.Expr = None) -> Dict[str, sp.Expr]:
        """
        2A → A: same universality class as pair annihilation.
        Lee (1994), Peliti (1986).
        """
        result = KnownResults.pair_annihilation(d)
        result['reference'] = 'Lee (1994), Peliti (1986)'
        result['process'] = '2A → A'
        return result

    @staticmethod
    def two_species_annihilation(d: sp.Expr = None) -> Dict[str, sp.Expr]:
        """
        A+B → ∅: Lee & Cardy (1995) J. Stat. Phys. 80:971.

        - d_c = 4
        - ρ(t) ~ t^{-d/4} for d < 4 (fluctuation-dominated)
        - Amplitude C ~ Δ^{1/2} (Δ = initial density difference)
        """
        if d is None:
            d = sp.Symbol('d', positive=True)

        return {
            'd_c': sp.Integer(4),
            'alpha': d / 4,
            'reference': 'Lee & Cardy (1995) J. Stat. Phys. 80:971',
        }

    @staticmethod
    def bws_hypercubic(d: sp.Expr = None) -> Dict[str, sp.Expr]:
        """
        Branching Wiener Sausage on hypercubic lattice.
        Bordeu, Amarteifio et al. (2019).

        - d_c = 4
        - V(t) ~ t^{d/4} for d < 4  (distinct sites visited)
        - V(t) ~ t/ln(t) at d = d_c = 4 (logarithmic correction)
        """
        if d is None:
            d = sp.Symbol('d', positive=True)

        return {
            'd_c': sp.Integer(4),
            'V_exponent': d / 4,
            'reference': 'Bordeu, Amarteifio+ (2019)',
        }

    @staticmethod
    def directed_percolation(eps: sp.Expr = None) -> Dict[str, sp.Expr]:
        """
        Directed percolation universality class.
        A → 2A, 2A → ∅ (branching-annihilation).

        One-loop RG results:
            ν_⊥ = 1/2 + 3ε/16 + O(ε²)
            η = 0 + O(ε²)

        Reference: Täuber-Howard-Vollmayr-Lee (2005) Table 1.
        """
        if eps is None:
            eps = sp.Symbol('epsilon', positive=True)

        return {
            'd_c': sp.Integer(4),
            'nu_perp_1loop': sp.Rational(1, 2) + sp.Rational(3, 16) * eps,
            'eta_1loop': sp.S.Zero,
            'reference': 'Täuber-Howard-Vollmayr-Lee (2005) J. Phys. A 38:R79',
        }
