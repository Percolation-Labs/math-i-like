"""
rdft.ac.conserved
=================
Conservation-law-aware coupled DSEs for the CDP / depinning universality.

The conserved directed percolation (CDP) class — to which Manna sandpile,
fixed-energy sandpiles, and depinning interfaces all belong — has a
2-field structure:
    psi  = activity field (DP-like)
    rho  = conserved density field (long-range diffusive)

The activity follows a DP-style action with absorption (psi^2 vertex);
the density is enforced to be conserved (∂_t rho is a divergence).  The
coupling is a "kinetic" term psi-tilde psi rho that converts active
particles to background and back.

Field-theoretic action (Vespignani-Munoz-Dickman 1998-2000, Le Doussal-Wiese
2002, 2016):

    S = int dt d^d x [
          psi-tilde (∂_t - D ∇^2) psi
        + lambda psi-tilde psi (psi - psi-tilde)      <- DP-like absorbing
        + chi   psi-tilde psi rho                      <- conservation coupling
        + rho-tilde (∂_t - D_rho ∇^2) rho
        + chi'  rho-tilde psi-tilde psi               <- back-coupling
        ]

The long-range nature of conservation appears via the SOFT MODE: rho's
Green's function is 1/(i omega + D_rho k^2), purely diffusive (no mass).
At zero external momentum and frequency, this propagator DIVERGES (k=0
is a singularity), and naive elimination via substitution G_rho = z(...)
fails.  The correct treatment requires an EFFECTIVE coupling that
absorbs the soft-mode contribution.

This module provides:
    coupled_dse_conserved(...) — the two-field DSE with conservation projector.
    cdp_one_loop_dressing(...) — one-loop spatial dressing in eps = 4 - d
                                  for the CDP class, computed from the
                                  conservation-modified rank-2 bridge.

Honest scope: this is a SCOPING implementation.  It captures the
algebraic structure of the conservation projection.  For numerical
agreement with FRG (Le Doussal-Wiese 2-loop) requires functional RG
treatment that is outside the algebraic-discriminant scope of CFAC.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Sequence, Optional


def cdp_action_vertices() -> dict:
    """Return the vertex content of the CDP action.

    Vertices are labelled by (m_psi_dual, m_psi, m_rho_dual, m_rho).  For
    each, the engineering dimension determines its relevance.
    """
    return {
        # mass term: (1, 1, 0, 0) — psi-tilde psi, kinetic
        # absorbing: (2, 1, 0, 0) — psi-tilde^2 psi, the DP cubic vertex
        ('DP_cubic', (2, 1, 0, 0)): {'engineered_dim': 4, 'role': 'DP-relevant'},
        # conservation coupling: (1, 1, 0, 1) — psi-tilde psi rho
        ('cons_coupling', (1, 1, 0, 1)): {'engineered_dim': 4, 'role': 'conservation-relevant'},
        # back-coupling: (1, 1, 1, 0) — psi-tilde psi rho-tilde
        ('back_coupling', (1, 1, 1, 0)): {'engineered_dim': 4, 'role': 'conservation-back'},
    }


def cdp_one_loop_dressing(d: float, lambda_DP: float = 1.0,
                          chi_cons: float = 1.0) -> dict:
    """One-loop spatial dressing at the CDP fixed point.

    For CDP at d_c = 4, the standard one-loop computation (Vespignani-
    Munoz 2000) gives:
      beta(g_DP) = -eps g_DP + b_DP g_DP^2 / (16 pi^2)
      beta(g_chi) = -eps g_chi + b_chi g_chi^2 / (16 pi^2)
    with conservation modifying b_chi by an extra "soft mode" contribution.

    The eta exponent (anomalous dimension of psi):
      eta = -g* * (counting) / (8 pi^2)
    At one loop, conservation REDUCES eta compared to plain DP by a factor
    related to the soft-mode integral.

    For the cluster size exponent tau via hyperscaling:
      tau_CDP = 1 + d nu / (d nu + beta_OP)
    with nu, beta_OP from CDP-modified Wilson-Fisher.

    HONEST SCOPE: this function returns the structural setup with one-loop
    coefficients estimated from the Vespignani-Munoz framework; it is NOT
    a substitute for the Le Doussal-Wiese 2-loop FRG.

    For Manna at d=1 (eps=3), perturbative one-loop is unreliable; we
    report the formula but mark it as non-perturbative.
    """
    eps = 4 - d
    if eps <= 0:
        return {'d': d, 'tau_CDP': 1.5, 'eta': 0, 'note': 'mean-field d>=4'}

    # PLACEHOLDER one-loop CDP coefficients.
    # The PROPER values come from Le Doussal-Wiese 2-loop FRG:
    #   zeta(d) = eps/3 (1 + 0.14331 eps), eps = 4 - d
    # which then maps to tau_CDP via depinning hyperscaling.  Implementing
    # that requires the FRG infrastructure outside CFAC.  Here we record
    # the PUBLISHED LDW zeta for reference.
    zeta_LDW_2loop = (eps / 3) * (1 + 0.14331 * eps)
    return {
        'd': d,
        'eps': eps,
        'zeta_LDW': zeta_LDW_2loop,
        'tau_CDP': float('nan'),  # NOT computed from CFAC one-loop
        'note': (
            'CFAC one-loop CDP not implemented; Le Doussal-Wiese 2-loop FRG '
            'is the reference for this class.  Zeta value above is from LDW; '
            'tau requires depinning hyperscaling (beyond current scope).'
        ),
    }


def coupled_dse_conserved(
        dp_vertices: dict[tuple[int, int], float],
        chi_coupling: float,
        d: float = 4.0,
        z: Optional[sp.Symbol] = None,
        G_psi: Optional[sp.Symbol] = None,
) -> dict:
    """CDP coupled DSE with conservation projector.

    For the conserved-density field rho with diffusive Green's function
    1/(i omega + D_rho k^2), at zero external momentum the propagator is
    formally infinite (k=0 singularity).  The CONSERVATION PROJECTOR sets
    the rho zero-mode to its NESS value; loop integrals over rho are
    cut off by the IR diffusive scale.

    For an algebraic DSE reduction at zero external momentum:
      G_psi = z (1 + sum vertex_psi G_psi^... + chi rho_eff G_psi)
    where rho_eff = (NESS density) is treated as an effective coupling
    for the algebraic curve.

    This is the simplest scoping implementation.  For full CDP physics
    one needs the soft-mode expansion that we leave for the next
    library extension.
    """
    if z is None:
        z = sp.Symbol('z', positive=True)
    if G_psi is None:
        G_psi = sp.Symbol('G_psi')

    # Build psi-sector kernel
    phi_psi = sp.S.One
    for (m, n), g in dp_vertices.items():
        if m + n >= 3:
            phi_psi += g * G_psi ** (m + n - 2)

    # Conservation effective coupling at NESS density rho_eff = 1 (canonical)
    # The chi G_psi rho coupling becomes chi G_psi after rho -> rho_eff = 1
    rho_eff = 1.0
    full_kernel = phi_psi + chi_coupling * G_psi * rho_eff

    F = sp.expand(G_psi - z * full_kernel)
    F_poly = sp.Poly(F, G_psi)
    return {
        'F': F,
        'degree_in_G_psi': F_poly.degree(),
        'phi_psi_alone': phi_psi,
        'phi_with_conservation': sp.expand(full_kernel),
        'comment': (
            'Scoping implementation: rho integrated out at NESS density rho_eff=1.'
            'Full CDP needs soft-mode expansion of conservation Green function.'
        ),
    }
