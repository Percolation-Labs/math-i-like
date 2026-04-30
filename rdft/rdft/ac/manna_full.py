"""
rdft.ac.manna_full
==================
Tier: 2 (extension)

Manna / CDP exponents from the FULL CFAC pipeline (counting × bridge ×
algebra), mirroring one_loop_KS for chemotaxis and one_loop_On for O(N)
phi^4.

This is the PROPER application of CFAC to Manna: write the CDP action
as a coupled DP-MSR system, identify the 1-loop self-energy diagrams,
and for each diagram apply the factorisation
    contribution = counting * bridge * algebra
where:
  - counting = vertex-coupling product (combinatorial)
  - bridge   = mass-independent residue of the loop integral
              (from rdft/ac/bridge.py — handles soft-mode IR)
  - algebra  = Gamma-function ratios (standard)

Action (Vespignani-Munoz-Dickman / Rossi-Pastor-Satorras-Vespignani):
  S = int dt d^d x [
        psi-tilde (d_t - D_psi grad^2) psi      // activity kinetic
      - sigma psi-tilde^2 psi                    // branching A -> 2A
      + lambda psi-tilde psi^2                   // coalescence 2A -> A
      + chi psi-tilde psi rho                    // density boosts activity
      + rho-tilde (d_t - D_rho grad^2) rho       // BACKGROUND DIFFUSIVE
      - chi' rho-tilde psi-tilde psi             // activity drains density
      ]

Two 1-loop self-energy diagrams contribute to psi:
  (A) DP loop: sigma-lambda vertex pair, internal G_psi G_psi
       counting = sigma * lambda
       bridge   = bridge_scalar() = 1
       algebra  = standard DP 1-loop pole
       gives the DP-like contribution.

  (B) Soft-mode loop: chi-chi' vertex pair, internal G_psi G_rho
       counting = chi * chi'
       bridge   = bridge_gradient_mass(D_psi, D_rho) = ln(r)/(r-1)
                  with r = D_psi / D_rho
       algebra  = standard 1-loop pole at d_c = 4
       gives the EXTRA CDP contribution beyond DP.

The activity anomalous dimension at one loop:
   eta_psi = (sigma lambda) * 1 + (chi chi') * f(r) all over (8 pi^2)
        = counting_DP / (8 pi^2) + counting_chi * f(r) / (8 pi^2)

The cluster-size exponent tau follows from scaling.  In d_c = 4:
   tau = 1 + d_f / (d - d_f)  where d_f involves eta_psi, z

For Manna in 1+1d, observed tau ~ 1.30.  CFAC at one loop gives a
specific prediction.

What this module computes
-------------------------
- manna_one_loop_self_energy: the two-diagram contribution to Sigma_psi
- manna_eta_one_loop: the anomalous dimension at the WF fixed point
- manna_tau_prediction: the cluster-size exponent via scaling
- comparison_DP_vs_CDP: the SHIFT from DP to CDP at one loop
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from .bridge import (
    bridge_scalar, bridge_gradient_mass, bridge_gradient_diffusion,
    one_loop_pole, omega_d, bridge_rank_k
)


def manna_action_couplings(sigma: float, lambda_dp: float,
                            chi: float, chi_prime: float,
                            D_psi: float, D_rho: float) -> Dict:
    """The CDP/Manna action couplings (standard RPV normalisation)."""
    return {
        'sigma': sigma,            # branching A -> 2A
        'lambda_dp': lambda_dp,    # coalescence 2A -> A
        'chi': chi,                # activity-density coupling (forward)
        'chi_prime': chi_prime,    # activity-density coupling (back)
        'D_psi': D_psi,            # activity diffusion
        'D_rho': D_rho,            # background diffusion (CONSERVED, massless)
        'd_c': 4.0,                # upper critical dimension (DP-like)
    }


def manna_one_loop_self_energy(sigma: float, lambda_dp: float,
                                 chi: float, chi_prime: float,
                                 D_psi: float = 1.0, D_rho: float = 1.0,
                                 d_c: float = 4.0) -> Dict:
    """One-loop self-energy of the activity field psi.

    Two diagrams:
      (A) DP loop: counting = sigma * lambda, bridge = 1
      (B) Soft-mode loop: counting = chi * chi_prime,
          bridge = bridge_gradient_mass(D_psi, D_rho)

    Each contributes a 1/eps pole.  Total residue:
       Sigma_psi(p=0) = (sigma*lambda * 1 + chi*chi_prime * f(r)) /
                       (8 pi^2 D_psi^2 eps)

    Returns the breakdown.
    """
    # Bridges (mass-independent, computed at d_c)
    bridge_DP = bridge_scalar()                                # = 1
    bridge_soft = bridge_gradient_mass(D_psi, D_rho)            # = ln(r)/(r-1)

    # Loop measure at d_c=4: 1/(8 pi^2)
    loop_factor = 1.0 / (8 * np.pi**2)

    # Counting
    counting_DP = sigma * lambda_dp
    counting_soft = chi * chi_prime

    # Pole residues (coefficient of 1/eps)
    pole_DP = counting_DP * bridge_DP * loop_factor / D_psi**2
    pole_soft = counting_soft * bridge_soft * loop_factor / (D_psi * D_rho)

    return {
        'bridge_DP': bridge_DP,
        'bridge_soft': bridge_soft,
        'counting_DP': counting_DP,
        'counting_soft': counting_soft,
        'pole_DP': pole_DP,
        'pole_soft': pole_soft,
        'pole_total': pole_DP + pole_soft,
        'cdp_minus_dp_pole': pole_soft,  # the EXTRA contribution
        'diffusion_ratio': D_psi / D_rho,
        'derivation': (
            'Sigma_psi(p=0) = counting_DP * bridge_scalar / (8 pi^2 D_psi^2) '
            '+ counting_soft * bridge_gradient_mass(D_psi, D_rho) / '
            '(8 pi^2 D_psi D_rho) -- all per 1/eps.'
        ),
    }


def manna_one_loop_beta_functions(d: float = 1.0,
                                    D_psi: float = 1.0, D_rho: float = 1.0
                                    ) -> Dict:
    """One-loop beta functions for the CDP/Manna couplings.

    At d_c = 4, eps = 4 - d.  The beta functions at one loop:
       beta(sigma) = -eps/2 * sigma + B_DP * sigma^2 lambda
       beta(lambda) = -eps/2 * lambda + B_DP * sigma lambda^2
       beta(chi) = -eps/2 * chi + B_chi * chi^2 chi_prime * f(r)
       beta(chi_prime) = -eps/2 * chi_prime + B_chi * chi chi_prime^2 * f(r)

    where:
       B_DP = bridge_scalar() / (8 pi^2)
       B_chi = bridge_gradient_mass(D_psi, D_rho) / (8 pi^2)
       f(r) = ln(r)/(r-1) with r = D_psi/D_rho

    The Wilson-Fisher fixed point in eps:
       sigma* = lambda* = sqrt(eps / (2 B_DP))      (DP-like fixed point)
       chi* = chi_prime* = sqrt(eps / (2 B_chi))    (CDP-like fixed point)

    For Manna at d=1: eps = 3 (large).  Perturbative one-loop is
    QUANTITATIVELY UNRELIABLE in d=1 but gives the right STRUCTURE.

    Returns the fixed-point couplings and the predicted anomalous
    dimensions.
    """
    eps = 4.0 - d
    if eps <= 0:
        return {'eps': eps, 'note': 'mean-field d >= 4'}

    f_r = bridge_gradient_mass(D_psi, D_rho)
    B_DP = bridge_scalar() / (8 * np.pi**2)
    B_chi = f_r / (8 * np.pi**2)

    # Wilson-Fisher fixed points (at one loop, schematic)
    sigma_star_sq = eps / (2 * B_DP)
    chi_star_sq = eps / (2 * B_chi) if B_chi > 0 else float('inf')

    # Anomalous dimensions of psi at the fixed point
    # eta_psi = sigma* lambda* B_DP + chi* chi'* B_chi
    eta_DP_only = sigma_star_sq * B_DP   # if chi=0 (pure DP)
    eta_CDP = sigma_star_sq * B_DP + chi_star_sq * B_chi

    return {
        'eps': eps,
        'd': d,
        'd_c': 4.0,
        'bridge_DP': bridge_scalar(),
        'bridge_soft': f_r,
        'B_DP_norm': B_DP,
        'B_chi_norm': B_chi,
        'sigma_lambda_star_sq': sigma_star_sq,
        'chi_chi_prime_star_sq': chi_star_sq,
        'eta_psi_DP': eta_DP_only,
        'eta_psi_CDP': eta_CDP,
        'eta_shift_CDP_minus_DP': eta_CDP - eta_DP_only,
    }


def manna_tau_prediction(d: float = 1.0,
                          D_psi: float = 1.0, D_rho: float = 1.0
                          ) -> Dict:
    """Predict the cluster-size exponent tau for CDP/Manna at one loop.

    The cluster-size exponent for an absorbing-state transition is
        tau = 1 + d / (d_f)
    where d_f = (d + 2 - eta) / 2 is the cluster fractal dimension at
    the WF fixed point (one-loop scaling).

    For DP at d_c = 4, eta_DP ~ -eps/12 (one-loop, Janssen).
    For CDP at d_c = 4, eta_CDP gets the extra soft-mode contribution.

    At d=1 (eps = 3), one-loop is non-perturbative but we report the
    schematic prediction.
    """
    bf = manna_one_loop_beta_functions(d, D_psi, D_rho)
    if 'note' in bf and bf['note'] == 'mean-field d >= 4':
        return {'d': d, 'tau': 1.5, 'note': 'mean-field'}

    eta_CDP = bf['eta_psi_CDP']
    eta_DP = bf['eta_psi_DP']

    # Scaling: tau = 1 + d / d_f where d_f = (d + 2 - eta)/2
    # For Manna (1+1d): want tau ~ 1.30 -> d_f ~ 3.33 -> eta ~ -3.66 (large!)
    # In d=1, perturbative results are not directly comparable because
    # eps = 3 is large.  We report the SHIFT eta_CDP - eta_DP as the
    # cleaner quantity.

    d_f_DP = (d + 2 - eta_DP) / 2
    d_f_CDP = (d + 2 - eta_CDP) / 2
    tau_DP_pred = 1 + d / d_f_DP if d_f_DP > 0 else float('nan')
    tau_CDP_pred = 1 + d / d_f_CDP if d_f_CDP > 0 else float('nan')

    return {
        'd': d,
        'eps': bf['eps'],
        'eta_DP_oneloop': eta_DP,
        'eta_CDP_oneloop': eta_CDP,
        'eta_shift': bf['eta_shift_CDP_minus_DP'],
        'd_f_DP': d_f_DP,
        'd_f_CDP': d_f_CDP,
        'tau_DP_pred_oneloop': tau_DP_pred,
        'tau_CDP_pred_oneloop': tau_CDP_pred,
        'tau_shift': tau_CDP_pred - tau_DP_pred,
        'tau_DP_observed': 1.108,
        'tau_CDP_observed': 1.30,
        'tau_shift_observed': 1.30 - 1.108,
        'note': (
            'd=1 means eps=3 (large).  One-loop is non-perturbative; '
            'absolute tau values not reliable.  The SHIFT '
            'tau_CDP - tau_DP is more meaningful as it cancels '
            'common one-loop overcorrections.  Compare predicted '
            'shift to observed shift = 0.19.'
        ),
    }


def comparison_DP_vs_CDP(D_psi: float = 1.0, D_rho: float = 1.0,
                          d_values: list = None) -> Dict:
    """Compare DP and CDP exponent shift across dimensions.

    The CFAC machinery predicts that the difference between CDP and DP
    exponents is dominated by the soft-mode bridge contribution:

       (eta_CDP - eta_DP) / eta_DP = bridge_soft / bridge_DP * (chi*/sigma*)^2

    For equal diffusion (D_psi = D_rho = 1), bridge_soft = 1 (limit of
    bridge_gradient_mass) so the shift is set entirely by the WF
    fixed-point ratio.
    """
    if d_values is None:
        d_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    rows = []
    for d in d_values:
        bf = manna_one_loop_beta_functions(d, D_psi, D_rho)
        if 'eta_shift_CDP_minus_DP' in bf:
            rows.append({
                'd': d,
                'eps': bf['eps'],
                'eta_DP': bf['eta_psi_DP'],
                'eta_CDP': bf['eta_psi_CDP'],
                'shift': bf['eta_shift_CDP_minus_DP'],
            })
    return {
        'rows': rows,
        'D_psi': D_psi,
        'D_rho': D_rho,
        'bridge_at_equal_D': bridge_gradient_mass(1.0, 1.0),
    }
