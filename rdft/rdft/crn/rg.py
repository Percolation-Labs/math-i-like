"""
rdft.crn.rg
===========

``RGProgram`` orchestrates the full RG pipeline for a CRN.

Mechanical end-to-end. The only inputs are:
  (a) the CRN and a ``RenormalisationScheme`` (user's physical ansatz);
  (b) master integral values (gap (b) -- the bridge integrals).

Everything else -- 1-loop algebra factors, beta_1, BPHZ counterterms, Hopf
antipode double poles, primitive 2-loop residues, IBP simple poles, the four
RG functions via the Tauber relation, the Wilson-Fisher fixed point, and
the critical exponents -- is derived in code.

No hard-coded ``a1 = {...}`` rationals, no hard-coded ``prim = {...}``,
no hard-coded JT05 RG functions. If a number appears, it comes from either
the CRN (via Doi shift), the scheme (via the propagator + sub-point + IBP
structure), or the master values.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sympy as sp

from rdft.crn.crn import CRN
from rdft.crn.diagram import Diagram, Provenance
from rdft.crn.scheme import RenormalisationScheme, Schemes
from rdft.crn.algebra import (all_one_loop_algebra_factors, beta_one,
                                hopf_antipode_double_pole)
from rdft.crn.enumerator import (enumerate_phi_trees, diagram_from_phi_tree,
                                  enumerate_bubbles, enumerate_tadpoles)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ZFactor:
    """A renormalisation-constant Z-factor with double pole, simple pole,
    and a provenance entry."""
    name: str
    a_one_loop: sp.Rational              # 1-loop algebra factor
    double_pole: sp.Expr
    simple_pole_rat: sp.Expr = sp.Integer(0)
    simple_pole_L: sp.Expr = sp.Integer(0)
    provenance: List[Provenance] = field(default_factory=list)

    def display(self) -> str:
        L = sp.Symbol("L")
        return (f"Z_{self.name}: a^(1)={self.a_one_loop}, "
                f"double pole = {self.double_pole}, "
                f"simple pole = {self.simple_pole_rat} + {self.simple_pole_L}*L")


@dataclass
class Exponents:
    """Critical exponents with comparison to JT05 Eq. (60)."""
    eta: sp.Expr
    z: sp.Expr
    nu: sp.Expr
    beta_DP: sp.Expr
    residuals: Dict[str, sp.Expr] = field(default_factory=dict)

    def compare_to_jt05(self) -> Dict[str, bool]:
        return {k: v == 0 for k, v in self.residuals.items()}


# ---------------------------------------------------------------------------
# RGProgram
# ---------------------------------------------------------------------------

class RGProgram:
    """End-to-end RG pipeline for a CRN. Mechanical from CRN + scheme + masters.

    Usage:
        rg = RGProgram(CRN.reggeon_dp(), scheme=Schemes.jt05_reggeon_dp())
        rg.run()
        rg.exponents.compare_to_jt05()
        rg.audit()

    If no scheme is passed, defaults to JT05 Reggeon-DP.
    """

    def __init__(self, crn: CRN, *,
                 scheme: Optional[RenormalisationScheme] = None,
                 loop_order: int = 2):
        self.crn = crn
        self.scheme = scheme or Schemes.jt05_reggeon_dp()
        self.loop_order = loop_order
        self.history: List[Provenance] = []
        self.diagrams: List[Diagram] = []
        self.phi_polynomial: Optional[sp.Expr] = None
        self.lagrange_counts: Dict[int, int] = {}
        self.zfactors: Dict[str, ZFactor] = {}
        self.beta_1: Optional[sp.Rational] = None
        self.exponents: Optional[Exponents] = None
        self._is_run = False

    # ----- driver -----
    def run(self) -> "RGProgram":
        self.layer1_doi_shift()
        self.layer2_lagrange_counts()
        self.layer2_diagram_enumeration()
        self.layer2_one_loop_algebra()
        self.layer2_hopf_antipode()
        self.layer2_simple_poles()
        self.final_rg_functions_and_exponents()
        self._is_run = True
        return self

    # ----- Layer 1 -----
    def layer1_doi_shift(self) -> None:
        crn = self.crn.with_doi_vertices() if not self.crn.vertices else self.crn
        self.crn = crn
        self.phi_polynomial = crn.phi_polynomial(max_legs=3)
        self.history.append(Provenance(
            layer="Layer 1", rule="Doi shift",
            detail=(f"{len(crn.vertices)} Doi-shifted vertices "
                    f"({len(crn.interaction_vertices(max_legs=3))} cubic interactions); "
                    f"phi(G) = {self.phi_polynomial}"),
            reference="rdft/crn/crn.py: CRN.doi_shift",
        ))

    # ----- Layer 2: Lagrange counts -----
    def layer2_lagrange_counts(self, max_n: int = 7) -> None:
        for n in range(1, max_n + 1):
            shapes = (enumerate_phi_trees(n)
                      if (self.phi_polynomial == 1 + sp.Symbol("G")**2) else [])
            self.lagrange_counts[n] = len(shapes)
        self.history.append(Provenance(
            layer="Layer 2", rule="Lagrange inversion",
            detail=("[z^n]G for n=1,3,5,7 = "
                    + ", ".join(str(self.lagrange_counts.get(n, "?"))
                                for n in (1, 3, 5, 7))),
            reference="rdft/crn/enumerator.py: enumerate_phi_trees",
        ))

    # ----- Layer 2: diagram enumeration -----
    def layer2_diagram_enumeration(self) -> None:
        if self.phi_polynomial == 1 + sp.Symbol("G")**2:
            for n in (3, 5, 7):
                if 2 * (self.loop_order + 1) - 1 < n:
                    continue
                for shape in enumerate_phi_trees(n):
                    d = diagram_from_phi_tree(shape, crn_name=self.crn.name)
                    self.diagrams.append(d)
        bubbles = enumerate_bubbles(self.crn.interaction_vertices(max_legs=4))
        tadpoles = enumerate_tadpoles(self.crn.interaction_vertices(max_legs=4))
        self.diagrams.extend(bubbles)
        self.diagrams.extend(tadpoles)
        self.history.append(Provenance(
            layer="Layer 2", rule="diagram enumeration",
            detail=(f"{len(self.diagrams)} diagrams: phi-tree shapes + "
                    f"{len(bubbles)} bubbles + {len(tadpoles)} tadpoles"),
        ))

    # ----- Layer 2: 1-loop algebra factors a_X^(1) (DERIVED) -----
    def layer2_one_loop_algebra(self) -> None:
        """Derive a_X^(1) for every Z-factor in the scheme, and beta_1.

        Mechanical: a_X^(1) = c_X^(1)(crn) * K_X(scheme).
        """
        a_factors = all_one_loop_algebra_factors(self.crn, self.scheme)
        self.beta_1 = beta_one(a_factors)
        for name, a_val in a_factors.items():
            zf = ZFactor(name=name, a_one_loop=a_val, double_pole=sp.Integer(0))
            zf.provenance.append(Provenance(
                layer="Layer 2", rule="1-loop algebra factor",
                detail=f"a_{name}^(1) = c (CRN) * K (scheme) = {a_val}",
                reference="rdft/crn/algebra.py: one_loop_algebra_factor",
            ))
            self.zfactors[name] = zf
        self.history.append(Provenance(
            layer="Layer 2", rule="1-loop algebra + beta_1",
            detail=(f"a^(1) = {dict(a_factors)}; beta_1 = a_u - 2 a_psi = {self.beta_1}"),
            reference="rdft/crn/algebra.py",
        ))

    # ----- Layer 2: Hopf antipode -> double poles (DERIVED) -----
    def layer2_hopf_antipode(self) -> None:
        """Apply the Connes-Kreimer Hopf-antipode formula
        Z_X^(2,2) = (1/2) a_X^(1) (beta_1 + a_X^(1)) for every Z-factor.
        Universal identity for cubic theories.
        """
        for name, zf in self.zfactors.items():
            z22 = hopf_antipode_double_pole(zf.a_one_loop, self.beta_1)
            zf.double_pole = z22
            zf.provenance.append(Provenance(
                layer="Layer 2", rule="Hopf antipode (Connes-Kreimer)",
                detail=(f"Z_{name}^(2,2) = (1/2) a_{name}^(1) (beta_1 + a_{name}^(1))"
                        f" = {z22}"),
                reference="rdft/crn/algebra.py: hopf_antipode_double_pole",
            ))
        z22_dict = {n: self.zfactors[n].double_pole for n in self.zfactors}
        self.history.append(Provenance(
            layer="Layer 2", rule="Hopf antipode -> Z^(2,2)",
            detail=f"Z^(2,2) = {z22_dict}",
            reference="rdft/crn/algebra.py",
        ))

    # ----- Layer 2: simple poles via IBP closure (DERIVED from scheme) -----
    def layer2_simple_poles(self) -> None:
        """Derive Z_X^(2,1) for every Z-factor from:
          - BPHZ counterterm   (universal: (1/2) a_X^(1) (a_X^(1) - beta_1))
          - Primitive residue  (scheme.kinematic_kernels_2loop x master values)
        """
        masters = self.scheme.master_values or {}
        kernels = self.scheme.kinematic_kernels_2loop or {}

        for name, zf in self.zfactors.items():
            a_X = zf.a_one_loop
            bphz = sp.Rational(1, 2) * a_X * (a_X - self.beta_1)
            prim_rat = sp.Integer(0)
            prim_L = sp.Integer(0)
            for basis_name, master_vals in masters.items():
                K = kernels.get((name, basis_name), sp.Integer(0))
                prim_rat += K * master_vals.get("rat", sp.Integer(0))
                prim_L   += K * master_vals.get("L",   sp.Integer(0))
            Z21_rat = sp.simplify(bphz + prim_rat)
            Z21_L   = sp.simplify(prim_L)
            zf.simple_pole_rat = Z21_rat
            zf.simple_pole_L = Z21_L
            zf.provenance.append(Provenance(
                layer="Layer 2", rule="IBP closure -> Z^(2,1)",
                detail=(f"Z_{name}^(2,1) = BPHZ + sum_basis K * m = "
                        f"{Z21_rat} + {Z21_L}*L"),
                reference="rdft/crn/algebra.py + rdft/crn/scheme.py",
            ))
        self.history.append(Provenance(
            layer="Layer 2", rule="IBP closure -> Z^(2,1)",
            detail=(f"Simple poles: " +
                    "; ".join(f"{n}: {z.simple_pole_rat}+{z.simple_pole_L}L"
                              for n, z in self.zfactors.items())),
            reference="rdft/crn/scheme.py: scheme.kinematic_kernels_2loop",
        ))

    # ----- Final: Tauber relation + RG functions + exponents (DERIVED) -----
    def final_rg_functions_and_exponents(self) -> None:
        """Derive RG functions via the Tauber relation
            gamma_X(u) = -u * d/du [simple-pole residue of ln Z_X(u)],
        construct gamma, zeta, kappa, beta as standard combinations,
        solve beta(u*) = 0 for the Wilson-Fisher fixed point, and assemble
        the critical exponents.
        """
        eps = sp.Symbol("varepsilon", positive=True, real=True)
        u = self.scheme.coupling
        L = sp.log(sp.Rational(4, 3))

        # Tauber relation (JT05 / Tauber 2014 textbook §4):
        #   gamma_X(u) = -u * d/du [ Z_X^{simple-pole}(u) ]
        # where Z_X^{simple-pole}(u) = a_X^(1) u + Z_X^(2,1) u^2 + ...
        # is the 1/eps coefficient of Z_X(u) viewed as a Laurent series in eps.
        # The Z_X^(2,2)/eps^2 piece does NOT enter gamma_X directly -- it's
        # tied to beta(u) by the BPHZ consistency requirement that 1/eps^2
        # poles cancel order-by-order in renormalised quantities.
        gamma_X = {}
        for name, zf in self.zfactors.items():
            simple_pole = (zf.a_one_loop * u
                           + (zf.simple_pole_rat + zf.simple_pole_L * L) * u**2)
            gamma_X[name] = sp.simplify(-u * sp.diff(simple_pole, u))

        # Standard RG-function combinations (Reggeon-DP convention)
        gamma  = gamma_X['psi']
        zeta   = gamma_X['psi'] - gamma_X['lambda']
        kappa  = gamma_X['lambda'] - gamma_X['tau']

        # beta(u) from MSbar pole-cancellation on Z_combined.  The combination
        # is determined by the scheme: Z_combined = prod_X Z_X^{coupling_z_exp[X]}.
        # With u_bare = mu^eps * u * Z_combined, the requirement
        # mu * du_bare/dmu = 0 gives beta(u) finite as eps -> 0 with the form
        #   beta(u) = -eps*u + u * (d/du)[Z_combined^(simple-pole)(u)]
        # to 2 loops.  This follows from the standard MSbar pole-cancellation
        # (Vasilev §1.114; Zinn-Justin §11.4).
        coupling_z_exp = self.scheme.coupling_z_exponents or {}
        if not coupling_z_exp:
            # Fallback: use the Z_u alone (won't be right for theories where
            # the coupling has nontrivial field-rescaling exponents).
            coupling_z_exp = {"u": sp.Rational(1)}

        # Z_combined^{simple-pole}(u) at order u, u^2:
        #   coefficient of 1/eps:
        #     u^1: sum_X exponent_X * a_X
        #     u^2: sum_X exponent_X * Z_X^(2,1)
        Z_c_1 = sp.Integer(0)
        Z_c_2_rat = sp.Integer(0)
        Z_c_2_L = sp.Integer(0)
        for X, exponent in coupling_z_exp.items():
            if X not in self.zfactors:
                continue
            zf = self.zfactors[X]
            Z_c_1 += exponent * zf.a_one_loop
            Z_c_2_rat += exponent * zf.simple_pole_rat
            Z_c_2_L += exponent * zf.simple_pole_L

        # MSbar pole-cancellation:  beta(u) = u * (-eps + B(u))  where
        # B(u) = u * d/du[Z_c^(1)(u)]  is the finite (eps-independent) part.
        # With Z_c^(1)(u) = Z_c_1*u + Z_c^(2,1)*u^2,
        #   B(u) = u * (Z_c_1 + 2*Z_c^(2,1)*u) = Z_c_1*u + 2*Z_c^(2,1)*u^2.
        # Hence beta(u) = u^2*Z_c_1 + 2*u^3*Z_c^(2,1) - eps*u.
        Z_c_simple_pole = Z_c_1 * u + (Z_c_2_rat + Z_c_2_L * L) * u**2
        B = u * sp.diff(Z_c_simple_pole, u)        # = Z_c_1 u + 2 Z_c^(2,1) u^2
        beta_u = sp.expand(u * (-eps + B))
        beta_u = sp.simplify(beta_u)

        # Solve beta(u*) = 0 to O(eps^2)
        a1, a2 = sp.symbols("a1 a2", real=True)
        u_ansatz = a1 * eps + a2 * eps**2
        beta_at_us = sp.expand(beta_u.subs(u, u_ansatz))
        c2 = sp.Poly(beta_at_us, eps).coeff_monomial(eps**2)
        a1_solutions = [s for s in sp.solve(c2, a1) if s != 0]
        if not a1_solutions:
            self.exponents = None
            self.history.append(Provenance(
                layer="final", rule="Wilson-Fisher fixed point",
                detail="beta(u*) = 0 has no nontrivial solution at O(eps^2)",
            ))
            return
        a1_sol = a1_solutions[0]
        beta_at_us = sp.expand(beta_at_us.subs(a1, a1_sol))
        c3 = sp.Poly(beta_at_us, eps).coeff_monomial(eps**3)
        a2_sol = sp.solve(c3, a2)[0]
        u_star = sp.simplify(a1_sol * eps + a2_sol * eps**2)

        def trunc2(expr):
            return sp.series(expr, eps, 0, 3).removeO()

        eta_ours = trunc2(gamma.subs(u, u_star))
        z_ours = 2 + trunc2(zeta.subs(u, u_star))
        nu_inv_ours = 2 - trunc2(kappa.subs(u, u_star))
        nu_ours = trunc2(1 / nu_inv_ours)
        beta_ours = trunc2(nu_ours * (4 - eps + eta_ours) / 2)

        # =====================================================================
        # COMPARISON ONLY -- not part of the derivation.
        # The exponents above (eta_ours, z_ours, nu_ours, beta_ours) are the
        # mechanically derived results.  Below we compare against JT05
        # Eq. (60) for verification.  The JT_* values are JT05's published
        # numbers; they are NOT used to derive anything, only to compute
        # the residual.  For schemes other than JT05 Reggeon-DP, the
        # comparison would be against a different reference (or omitted).
        # =====================================================================
        JT_eta = trunc2(-(eps / 6) * (1 + (sp.Rational(25, 288)
                                            + sp.Rational(161, 144) * L) * eps))
        JT_z = trunc2(2 - (eps / 12) * (1 + (sp.Rational(67, 288)
                                              + sp.Rational(59, 144) * L) * eps))
        JT_nu = trunc2(sp.Rational(1, 2) + (eps / 16) * (1 + (sp.Rational(107, 288)
                                                               - sp.Rational(17, 144) * L) * eps))
        JT_beta = trunc2(1 - (eps / 6) * (1 - (sp.Rational(11, 288)
                                                - sp.Rational(53, 144) * L) * eps))

        # log(4/3) vs log(2)-log(3): rewrite both sides into log(4/3) so
        # sp.expand sees them as identical.  sp.simplify can hang on this.
        log43 = sp.log(sp.Rational(4, 3))
        def _to_log43(e):
            e = e.rewrite(sp.log)
            e = e.subs(sp.log(2), (log43 + sp.log(3)) / 2)
            return sp.expand(e)
        residuals = {}
        for k, ours, jt in [("eta", eta_ours, JT_eta),
                             ("z", z_ours, JT_z),
                             ("nu", nu_ours, JT_nu),
                             ("beta_DP", beta_ours, JT_beta)]:
            diff = sp.expand(_to_log43(ours - jt))
            residuals[k] = diff
        self.exponents = Exponents(eta=eta_ours, z=z_ours, nu=nu_ours,
                                    beta_DP=beta_ours, residuals=residuals)
        self.history.append(Provenance(
            layer="final", rule="Tauber relation + Wilson-Fisher",
            detail=("RG functions derived via gamma_X = -u * d/du(simple-pole "
                    "of ln Z_X); beta(u*)=0 at O(eps^2); exponents from "
                    "eta=gamma(u*), z=2+zeta(u*), nu=1/(2-kappa(u*)), "
                    "beta_DP=nu(d+eta)/2"),
            reference="rdft/crn/rg.py: final_rg_functions_and_exponents",
        ))

    # ----- audit -----
    def audit(self) -> str:
        if not self._is_run:
            return "RGProgram has not been run."
        from rdft.crn.audit import format_audit
        return format_audit(self)
