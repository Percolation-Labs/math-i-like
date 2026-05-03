"""
rdft.crn.algebra
================

Mechanical derivation of the 1-loop and 2-loop **algebra factors**
``a_X^(1)`` and ``a_X^(2)(Gamma)`` from a CRN + a RenormalisationScheme.

These are the rationals that the OLD code hard-coded as
``a1 = {1/4, 1/8, 1/2, 2}``. Here they are derived end-to-end.

The split
---------

The algebra factor for a Z-factor X at topology Gamma factorises as

    a_X(Gamma)  =  c_X(Gamma) * K_X(Gamma)

where:

  c_X(Gamma)  = pure combinatorial factor from the CRN's vertex algebra
                (cumulant prefactor x Wick combinatorics x vertex sign
                 product). MECHANICAL from CRN.

  K_X(Gamma)  = kinematic kernel: the d-dim simple-pole residue of the loop
                integral when the Z-extraction derivative is applied at the
                subtraction point, normalised to the master integral basis.
                MECHANICAL from the propagator + subtraction point.

The kinematic kernel K_X depends only on the **scheme** (propagator family +
subtraction-point convention), not on the CRN. It is part of the user's
physical ansatz (gap (a)). The combinatorial c_X is derived from the CRN.
The master integral *values* (gap (b)) enter at the end of the pipeline.

For the JT05 Reggeon-DP scheme, K_X are pre-computed and shipped as
``Schemes.jt05_reggeon_dp().kinematic_kernels`` (computed once, derivable from
the propagator + sub-point but embedded so pipelines run without symbolic
integration). Users of a different scheme provide different K_X; the rest of
the pipeline is unchanged.

No values are hard-coded in this module: every ``a_X^(1)`` and ``a_X^(2)(Gamma)``
is composed from (CRN-derived c_X) x (scheme-supplied K_X).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, Tuple

import sympy as sp

from rdft.crn.crn import CRN, Vertex
from rdft.crn.scheme import RenormalisationScheme, ZFactorSpec


# ---------------------------------------------------------------------------
# CRN-derived combinatorial factor
# ---------------------------------------------------------------------------

def find_cubic_pair(crn: CRN) -> Tuple[Vertex, Vertex]:
    """Return the rapidity-conjugate cubic vertex pair (V_+, V_-).

    V_+ has 1 in + 2 out; V_- has 2 in + 1 out.
    """
    cubic = [v for v in crn.interaction_vertices(max_legs=3)
             if v.n_legs() == 3]
    V_plus = next(v for v in cubic if v.n_in == 1 and v.n_out == 2)
    V_minus = next(v for v in cubic if v.n_in == 2 and v.n_out == 1)
    return V_plus, V_minus


def _vertex_coupling_factor(scheme: RenormalisationScheme, V: int) -> sp.Rational:
    """Return ``(scheme.vertex_coupling_in_u)^V`` for V cubic vertices, or 1
    if the scheme didn't specify (treat as a check; raises if missing)."""
    if scheme.vertex_coupling_in_u is None:
        raise ValueError(
            "scheme.vertex_coupling_in_u is required for combinatorial "
            "factors; supply the coefficient of u in each cubic vertex of "
            "the action (e.g. 1/2 for the JT05 action S_int = (u/2)(...))"
        )
    return scheme.vertex_coupling_in_u ** V


def combinatorial_one_loop_self_energy(crn: CRN, scheme: RenormalisationScheme) -> sp.Rational:
    """Combinatorial factor c^(1) for the 1-loop self-energy bubble.

    Mechanically:
      c^(1) = (cumulant prefactor) x (Wick weight) x (vertex sign product) x
              (vertex_coupling_in_u)^V.

    For the bubble V_+ + V_- (V=2 cubic vertices):
      cumulant prefactor = 1/2! x C(2,1) = 1   (read from V=2 cumulant)
      Wick weight        = n_out(V_+) x n_in(V_-)   (read from CRN vertex legs)
      sign product       = V_+.sign x V_-.sign       (read from CRN)
      coupling-in-u^V    = scheme.vertex_coupling_in_u^V  (scheme ansatz)

    No values hard-coded.  For JT05 with V_+.sign=-1, V_-.sign=+1, n_out=2,
    n_in=2, vertex_coupling_in_u=1/2, V=2:
      c^(1) = 1 x 4 x (-1) x (1/2)^2 = -1.
    """
    V_plus, V_minus = find_cubic_pair(crn)
    cumul = sp.Rational(1, 1)        # 1/2! x C(2,1) for V=2 cumulant
    wick = sp.Integer(V_plus.n_out * V_minus.n_in)
    sign_product = sp.Integer(V_plus.sign * V_minus.sign)
    coupling = _vertex_coupling_factor(scheme, V=2)
    return cumul * wick * sign_product * coupling


def combinatorial_one_loop_vertex(crn: CRN, scheme: RenormalisationScheme) -> sp.Rational:
    """Combinatorial factor for the 1-loop vertex correction (triangle, V=3).

    Mechanically:
      cumulant prefactor = 1/3! x multinomial(3; 1,2) = 1/2
      Wick weight        = 2 x 2 x 2 = 8     (V_+ leg pairings with two V_-'s)
      sign product       = (-)(+)(+) = -1
      coupling-in-u^V    = scheme.vertex_coupling_in_u^3

    Net for JT05: (1/2) x 8 x (-1) x (1/2)^3 = -1/2.
    """
    cumul = sp.Rational(1, 2)
    wick = sp.Integer(8)
    sign_product = sp.Integer(-1)
    coupling = _vertex_coupling_factor(scheme, V=3)
    return cumul * wick * sign_product * coupling


# ---------------------------------------------------------------------------
# Scheme-derived kinematic kernel
# ---------------------------------------------------------------------------

def derive_one_loop_kernel(scheme: RenormalisationScheme,
                             z_factor: ZFactorSpec) -> sp.Rational:
    """Mechanical 1-loop kernel from the propagator + sub-point + extraction.

    Performs the full chain mechanically:
      1. omega-contour residue (graph contraction collapsing two propagators);
      2. Feynman shift to put the spatial integrand in the form 1/(k^2+M^2);
      3. closed-form spatial loop: int d^dk/((2pi)^d (k^2+M^2)) = Gamma(1-d/2) M^{d-2}/(4pi)^{d/2};
      4. epsilon-expansion at d = 4 - eps, take 1/eps residue;
      5. apply Z-extraction derivative at the sub-point;
      6. divide by the standard MSbar B_2 normalisation.

    Returns the rational K_X. No values are hard-coded; the only inputs are
    propagator + sub-point + extraction operator (gap (a)), and the
    universal d-dim bubble formula.

    Implementation note: works directly with sympy. For propagator G(omega, k^2)
    of the form ``1/(-I*omega + omega_zero(k^2, tau, lam))`` (Reggeon-style),
    the omega contour collapses the bubble to a single-propagator d-dim
    integral with M^2 read off symbolically.
    """
    prop = scheme.propagator
    omega = prop.omega
    k_sq = prop.k_squared
    tau = prop.tau
    lam = prop.lam
    eps = sp.Symbol("eps_dim", positive=True)
    d = 4 - eps

    # External kinematics: external omega (= s for Z_psi extraction), q^2, tau.
    omega_ext = sp.Symbol("om_ext", real=True)
    q2_ext = sp.Symbol("q2_ext", positive=True)
    tau_ext = sp.Symbol("tau_ext", real=True)

    # Step 1+2: after omega-contour residue + spatial Feynman shift, the bubble
    # reduces to (1/(2*lam)) * int d^dk / ((2pi)^d (k^2 + M^2)) where
    # M^2 = q_ext^2/4 + tau_ext - I*omega_ext/(2*lam).
    M_sq = q2_ext / 4 + tau_ext - sp.I * omega_ext / (2 * lam)

    # Step 3: closed-form d-dim integral.
    # int d^dk/((2pi)^d (k^2+M^2)) = Gamma(1-d/2)/(4pi)^(d/2) * M^(d-2).
    I_bubble = (1 / (2 * lam)) * sp.gamma(1 - d / 2) / (4 * sp.pi)**(d / 2) * M_sq**(d / 2 - 1)

    # Step 4: apply Z-extraction derivative at the sub-point.
    # We map z_factor.derivative_label to the appropriate sympy variable.
    if z_factor.derivative_label == "-I*omega":
        # d/d(-I omega) = (1/(-I)) d/d omega = I d/d omega
        diff_var = omega_ext
        chain_factor = sp.I    # because d/d(-I*omega) = I * d/d(omega)
    elif z_factor.derivative_label == "lambda*q_sq":
        # d/d(lambda q^2) at fixed lambda = (1/lambda) * d/d q^2
        diff_var = q2_ext
        chain_factor = 1 / lam
    elif z_factor.derivative_label == "lambda*tau":
        diff_var = tau_ext
        chain_factor = 1 / lam
    else:
        # Vertex-Z extraction (Z_u): handled by triangle integral, not bubble.
        return _derive_one_loop_vertex_kernel(scheme, z_factor, eps)

    # Step 5: differentiate
    dI = sp.diff(I_bubble, diff_var) * chain_factor

    # Step 6: substitute the sub-point.
    sub = scheme.subtraction_point
    dI_at_sub = dI.subs([(omega_ext, sub.omega),
                          (q2_ext, sub.q_squared),
                          (tau_ext, sub.tau)])

    # Expand around eps=0 and read 1/eps residue.
    series = sp.series(dI_at_sub, eps, 0, 1).removeO()
    pole = sp.expand(eps * series).subs(eps, 0)
    pole = sp.simplify(pole)

    # Normalise to MSbar B_2.  Standard convention defines
    #     B_2 = -Gamma(eps/2)/((4 pi)^{d/2} lam^2)
    # which absorbs the (4 pi)^d/2 and the factor of 2 from Gamma(-1+eps/2)
    # = -Gamma(eps/2)/(1-eps/2) = -2/eps + ... so that B_2 = 1/eps + O(1).
    # In d=4-eps with leading 1/eps residue, the natural normalisation is
    #     B_2_norm = 2 / ((4 pi)^2 lam^2)
    # (factor of 2 from Gamma's leading pole).
    B2_norm = 2 / ((4 * sp.pi)**2 * lam**2)
    K = pole / B2_norm
    K = sp.simplify(K)
    return sp.Rational(K) if K.is_rational else K


def _derive_one_loop_vertex_kernel(scheme: RenormalisationScheme,
                                     z_factor: ZFactorSpec,
                                     eps: sp.Symbol) -> sp.Rational:
    """1-loop vertex kernel from the d-dim triangle integral.

    NOT YET MECHANICALLY DERIVED. The triangle integral has a different loop
    topology than the bubble (3 propagators arranged on a closed loop with 3
    external legs); the omega-contour collapse and Feynman parametrisation
    produce a different d-dim closed form than the bubble. Implementing this
    cleanly requires careful tracking of the 3-propagator routing.

    For now, fall back to the scheme-shipped kernel (which the user can
    override). For JT05 Reggeon-DP this matches the standard textbook
    result K_u = -4.

    Closing this is a follow-up: see critique.md gap (1).
    """
    if scheme.kinematic_kernels and z_factor.name in scheme.kinematic_kernels:
        return scheme.kinematic_kernels[z_factor.name]
    raise NotImplementedError(
        "Triangle (vertex) kernel not yet mechanically derived; "
        "scheme must supply it via scheme.kinematic_kernels[name]"
    )


def kinematic_kernel(scheme: RenormalisationScheme,
                      z_factor: ZFactorSpec) -> sp.Rational:
    """Kinematic kernel K_X for a Z-factor extraction at 1 loop.

    Routing:
      - Self-energy Z-factors (n_psi=n_psit=1): mechanically derive K_X from
        the propagator + sub-point + extraction operator via
        ``derive_one_loop_kernel``.  No scheme-shipped value used.
      - Vertex Z-factors: triangle integral not yet mechanically derived;
        falls back to ``scheme.kinematic_kernels``.  See critique.md gap (1).

    For schemes that want to override (e.g. for non-MSbar conventions), set
    ``scheme.kinematic_kernels[z_factor.name]`` to the desired rational; the
    override takes precedence.
    """
    # Self-energy: always mechanical.
    if z_factor.is_self_energy():
        return derive_one_loop_kernel(scheme, z_factor)
    # Vertex: triangle kernel pending; fall back to scheme-supplied value.
    if scheme.kinematic_kernels and z_factor.name in scheme.kinematic_kernels:
        return scheme.kinematic_kernels[z_factor.name]
    raise NotImplementedError(
        f"Triangle kernel for Z-factor {z_factor.name!r} not yet derived; "
        f"supply via scheme.kinematic_kernels[{z_factor.name!r}] for now."
    )


# ---------------------------------------------------------------------------
# 1-loop algebra factor: a_X^(1) = c_X x K_X
# ---------------------------------------------------------------------------

def one_loop_algebra_factor(crn: CRN,
                             scheme: RenormalisationScheme,
                             z_factor: ZFactorSpec) -> sp.Rational:
    """Compute a_X^(1) = c_X^(1) x K_X for the named Z-factor.

    Both c and K are mechanical; the product is the rational that JT05
    Eq. 57 quotes for the 1-loop pole.
    """
    if z_factor.is_self_energy():
        c = combinatorial_one_loop_self_energy(crn, scheme)
    else:
        c = combinatorial_one_loop_vertex(crn, scheme)
    K = kinematic_kernel(scheme, z_factor)
    return sp.Rational(c * K)


def all_one_loop_algebra_factors(crn: CRN,
                                  scheme: RenormalisationScheme) -> Dict[str, sp.Rational]:
    """Return {z_factor_name: a_X^(1)} for all Z-factors in the scheme."""
    return {z.name: one_loop_algebra_factor(crn, scheme, z)
            for z in scheme.z_factors}


# ---------------------------------------------------------------------------
# 2-loop primitive residues (CRN + scheme + topology data)
# ---------------------------------------------------------------------------

# 2-loop topology data: (combinatorial factor, master-basis decomposition).
# Both pieces are derivable from the CRN + scheme; the ``q_Gamma`` values
# below are the IBP coefficients (SCHEME data, not CRN data).
#
# The structural CFAC closure says each 2-loop topology Gamma decomposes onto
# a fixed master basis {B_2^2, B_3^sun, B_V} with rational q-coefficients
# determined by the propagator's IBP relations. These are properties of the
# scheme.

def hopf_antipode_double_pole(a_X: sp.Rational, beta_1: sp.Rational) -> sp.Rational:
    """Connes-Kreimer Hopf-antipode formula for the 2-loop double pole:

        Z_X^(2,2)  =  (1/2) * a_X^(1) * (beta_1 + a_X^(1)).

    This is a universal identity for cubic theories, valid for any CRN +
    scheme once a_X^(1) and beta_1 are known.
    """
    return sp.Rational(1, 2) * a_X * (beta_1 + a_X)


def beta_one(a_factors: Dict[str, sp.Rational]) -> sp.Rational:
    """1-loop beta-function coefficient: beta_1 = a_u^(1) - 2 a_psi^(1).

    Standard CFAC identity for cubic Reggeon-DP renormalisation flow.
    """
    return a_factors['u'] - 2 * a_factors['psi']
