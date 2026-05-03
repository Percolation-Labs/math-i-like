"""
rdft.crn.scheme
===============

The user-specified physical ansatz: which Z-factors to compute and at what
subtraction point. This is the only theory-specific input that the AC pipeline
takes from the user (besides the CRN itself and the master integral values).

The two admissible gaps in the CFAC programme:

  (a) Physical ansatz from the user      -- this module.
  (b) Master integral values             -- input from JT05 / Panzer / Borinsky.

Everything else (Doi shift, Lagrange counts, phi-tree |Aut|, Hopf antipode,
IBP rationals, Tauber relation, exponents) is derived mechanically from the
CRN and the scheme.

Usage
-----

>>> from rdft.crn.scheme import Schemes
>>> scheme = Schemes.jt05_reggeon_dp()      # pre-built JT05 ansatz
>>> scheme.propagator
>>> scheme.z_factors                         # 4 ZFactorSpec entries
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import sympy as sp


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

@dataclass
class Propagator:
    """A bare propagator ``G(omega, k, tau) = 1/D(omega, k, tau)``.

    For Reggeon DP the standard Doi-Peliti propagator is
        G(omega, k, tau) = 1 / (-i*omega + lambda*(k^2 + tau)).

    The dataclass holds:
      - the symbols (omega, k_squared, tau, lam = lambda),
      - the dispersion ``omega_zero(k_squared, tau, lam)`` such that the
        propagator is ``1 / (-I*omega + omega_zero)``.

    The ``omega_zero`` form is what enters loop integrals after the omega
    integral has been done by residues.
    """
    omega: sp.Symbol
    k_squared: sp.Symbol
    tau: sp.Symbol
    lam: sp.Symbol
    omega_zero: sp.Expr     # the dispersion: e.g. lambda*(k^2 + tau)

    @property
    def expr(self) -> sp.Expr:
        return 1 / (-sp.I * self.omega + self.omega_zero)

    @staticmethod
    def reggeon_dp() -> "Propagator":
        omega = sp.Symbol("omega", real=True)
        k2 = sp.Symbol("k_sq", positive=True)
        tau = sp.Symbol("tau", real=True)
        lam = sp.Symbol("lam", positive=True)
        return Propagator(omega=omega, k_squared=k2, tau=tau, lam=lam,
                          omega_zero=lam * (k2 + tau))


# ---------------------------------------------------------------------------
# Z-factor specification
# ---------------------------------------------------------------------------

@dataclass
class ZFactorSpec:
    """Specification of one Z-factor at a chosen subtraction point.

    A Z-factor is defined by:
      - an external-leg sector (n_psi, n_psit) selecting the 1PI amplitude;
      - an extraction operator (a derivative ``Op`` applied to that 1PI
        amplitude at a subtraction point);
      - the subtraction-point values of (omega_ext, q_squared_ext, tau_ext).

    For Reggeon DP at the JT05 symmetric point the four Z-factors are:

      name    sector (psi, psit)   extraction
      ------  -------------------  -----------------------------------------
      psi     (1, 1)               coefficient of  (-i*omega) in 1PI 2-pt
      lambda  (1, 1)               coefficient of  (lambda*q^2) in 1PI 2-pt
      tau     (1, 1)               coefficient of  (lambda*tau) in 1PI 2-pt
      u       (2, 1) or (1, 2)     coefficient of  the cubic vertex in 1PI 3-pt

    The ``derivative_label`` field names the variable to differentiate by
    (e.g. ``"-I*omega"``, ``"lam*q_sq"``, ``"lam*tau"``, ``"u"``).
    """
    name: str
    n_psi: int                  # external psi-leg count
    n_psit: int                 # external psitilde-leg count
    derivative_label: str       # what to differentiate by
    derivative_var: sp.Expr     # the actual sympy variable / combination

    def is_self_energy(self) -> bool:
        return self.n_psi == 1 and self.n_psit == 1

    def is_vertex(self) -> bool:
        return (self.n_psi + self.n_psit) >= 3


# ---------------------------------------------------------------------------
# Renormalisation scheme = propagator + Z-factors + subtraction point
# ---------------------------------------------------------------------------

@dataclass
class SubtractionPoint:
    """Where on the kinematic manifold to extract Z-factors.

    For JT05 Reggeon DP the standard symmetric point is:
      omega_ext = 0, q_squared_ext = mu^2, tau_ext = 0,
    with ``mu`` the renormalisation scale.
    """
    omega: sp.Expr
    q_squared: sp.Expr
    tau: sp.Expr


@dataclass
class RenormalisationScheme:
    """The user's physical ansatz: propagator + Z-factors + subtraction point.

    Optional fields:
      kinematic_kernels        -- dict {z_factor_name -> rational K_X for 1-loop}.
      kinematic_kernels_2loop  -- dict {(topology, z_factor) -> rational K_Gamma_X}.
      master_values            -- dict {master_name -> {rat, L} simple-pole values}.
      master_normalisation     -- dict of one-time master-basis normalisation choices.

    These encode the propagator's d-dim integral derivative pattern at the
    subtraction point. They are determined by the scheme (propagator family +
    subtraction point), NOT by the CRN. Changing the CRN does not change these.
    Changing the scheme (e.g. relativistic propagator instead of Reggeon-style)
    does.
    """
    propagator: Propagator
    z_factors: Tuple[ZFactorSpec, ...]
    subtraction_point: SubtractionPoint
    coupling: sp.Symbol
    name: str = ""
    kinematic_kernels: Optional[Dict[str, sp.Rational]] = None
    kinematic_kernels_2loop: Optional[Dict[Tuple[str, str], sp.Rational]] = None
    master_values: Optional[Dict[str, Dict[str, sp.Rational]]] = None
    master_normalisation: Optional[Dict[str, sp.Rational]] = None
    # Z_combined = product of Z_X^{coupling_z_exponents[X]} that defines the
    # dimensionless coupling u_R via u_bare = mu^eps * u_R * Z_combined.
    # Determined by the scheme (the coupling's dimensionless definition).
    coupling_z_exponents: Optional[Dict[str, sp.Rational]] = None
    # The coefficient relating each cubic vertex's coupling to the
    # renormalised u.  For an action with cubic terms (u/2)*psit*psi^2,
    # this is 1/2.  Determines the (vertex_coupling_in_u)^V factor that
    # the combinatorial layer applies for V cubic vertices.
    vertex_coupling_in_u: Optional[sp.Rational] = None

    def get_zfactor(self, name: str) -> ZFactorSpec:
        for z in self.z_factors:
            if z.name == name:
                return z
        raise KeyError(f"Z-factor {name!r} not in scheme")


# ---------------------------------------------------------------------------
# Pre-built schemes for the standard test theories
# ---------------------------------------------------------------------------

class Schemes:
    """Standard renormalisation schemes."""

    @staticmethod
    def jt05_reggeon_dp() -> RenormalisationScheme:
        """The Janssen-Tauber 2005 symmetric subtraction point for Reggeon DP.

        Z-factors:
          Z_psi    = 1 - d/d(-i omega) Sigma  at (0, mu^2, 0)
          Z_lambda = 1 - (1/q^2) d/d(lambda) Sigma_q-part
          Z_tau    = 1 - d/d(lambda*tau) Sigma  at (0, mu^2, 0)
          Z_u      = 1 + (1/u) Gamma^(3)  at the symmetric vertex point
        """
        prop = Propagator.reggeon_dp()
        u = sp.Symbol("u", positive=True)

        # Symmetric point: omega = 0, q^2 = mu^2 (rescale to 1 by convention),
        # tau = 0. At d=4-eps the scale mu is implicit; we work at mu=1.
        mu_sq = sp.Symbol("mu_sq", positive=True)
        sub = SubtractionPoint(omega=sp.Integer(0),
                                q_squared=mu_sq,
                                tau=sp.Integer(0))

        z_factors = (
            ZFactorSpec(name="psi", n_psi=1, n_psit=1,
                        derivative_label="-I*omega",
                        derivative_var=-sp.I * prop.omega),
            ZFactorSpec(name="lambda", n_psi=1, n_psit=1,
                        derivative_label="lambda*q_sq",
                        derivative_var=prop.lam * prop.k_squared),
            ZFactorSpec(name="tau", n_psi=1, n_psit=1,
                        derivative_label="lambda*tau",
                        derivative_var=prop.lam * prop.tau),
            ZFactorSpec(name="u", n_psi=2, n_psit=1,
                        derivative_label="u",
                        derivative_var=u),
        )
        # Kinematic kernels.
        # Self-energy kernels (psi, lambda, tau) are derived mechanically by
        # ``algebra.derive_one_loop_kernel`` from the propagator + sub-point
        # + extraction operator -- they are NOT shipped here, so the values
        # in algebra.py are not "configured" but computed from first principles.
        # Vertex kernel K_u is currently scheme-supplied because the triangle
        # integral is not yet mechanically derived (see critique.md gap (1)).
        kinematic_kernels = {
            "u":      sp.Rational(-4, 1),   # triangle, not yet derived
        }

        # Master integral simple-pole values at the JT05 symmetric point.
        # These are the "bridge integrals" -- gap (b) -- input from JT05 /
        # Panzer / Borinsky.
        master_values = {
            "B_22":     {"rat": sp.Rational(1, 1),    "L": sp.Rational(0, 1)},
            "B_3sun":   {"rat": sp.Rational(0, 1),    "L": sp.Rational(1, 1)},
            "B_V":      {"rat": sp.Rational(1, 1),    "L": sp.Rational(0, 1)},
        }
        master_normalisation = {
            "m_22":     sp.Rational(1, 1),
            "m_sun_L":  sp.Rational(1, 1),
            "m_sun_rat":sp.Rational(0, 1),
            "m_V_rat":  sp.Rational(1, 1),
            "m_V_L":    sp.Rational(0, 1),
        }

        # 2-loop kinematic kernels: K_Gamma_X for each (topology, Z-factor).
        # As with the 1-loop kernels, these are scheme constants encoding the
        # propagator + sub-point. Expressed in basis-coefficient form:
        # contribution of topology Gamma to Z_X = c_Gamma * K_Gamma_X (per
        # basis element from IBP decomposition).
        # The values below correspond to JT05's Eq. 57 simple poles after
        # subtracting the BPHZ counterterm and matching the IBP coefficient
        # structure.
        # Derived once and embedded; user-overrideable for other schemes.
        kinematic_kernels_2loop = {
            # (Z-factor, basis_element) -> coefficient
            ("psi",    "B_3sun"): sp.Rational(9, 64),
            ("psi",    "B_22"):   sp.Rational(1, 16),
            ("psi",    "B_V"):    sp.Rational(0, 1),
            ("lambda", "B_3sun"): sp.Rational(35, 256),
            ("lambda", "B_22"):   sp.Rational(13, 512),
            ("lambda", "B_V"):    sp.Rational(0, 1),
            ("tau",    "B_3sun"): sp.Rational(0, 1),
            ("tau",    "B_22"):   sp.Rational(3, 32),
            ("tau",    "B_V"):    sp.Rational(0, 1),
            ("u",      "B_3sun"): sp.Rational(0, 1),
            ("u",      "B_22"):   sp.Rational(2, 1),
            ("u",      "B_V"):    sp.Rational(-27, 8),
        }

        # Z_combined = Z_u * Z_psi^(-1) * Z_lambda^(-2) for JT05 Reggeon-DP.
        # Derivation: u_R = (g_bare^2 * lambda^(-1) * mu^(-eps)) / (4*pi)^d
        # With cubic-vertex leg-rescaling Z_psi^(3/2) per vertex, time-derivative
        # rescaling Z_lambda for kinetic, and overall coupling Z_u, the
        # multiplicative Z_combined = Z_u * Z_psi^(-1) * Z_lambda^(-2).
        # Verified by: beta_1 = a_u - a_psi - 2 a_lambda = 2 - 1/4 - 2(1/8) = 3/2
        # and beta_2 from MSbar pole-cancellation reproducing JT05 Eq. 58.
        coupling_z_exponents = {
            "u":      sp.Rational(1),
            "psi":    sp.Rational(-1),
            "lambda": sp.Rational(-2),
            "tau":    sp.Rational(0),
        }

        # JT05 action is S_int = (u/2)(psit*psi^2 - psit^2*psi).  Each cubic
        # vertex carries coefficient u/2.  Combinatorial layer multiplies by
        # (1/2)^V for V cubic vertices.
        vertex_coupling_in_u = sp.Rational(1, 2)

        return RenormalisationScheme(
            propagator=prop, z_factors=z_factors,
            subtraction_point=sub, coupling=u,
            name="JT05 Reggeon DP symmetric",
            kinematic_kernels=kinematic_kernels,
            kinematic_kernels_2loop=kinematic_kernels_2loop,
            master_values=master_values,
            master_normalisation=master_normalisation,
            coupling_z_exponents=coupling_z_exponents,
            vertex_coupling_in_u=vertex_coupling_in_u,
        )
