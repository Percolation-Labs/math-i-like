"""
rdft.ac — Analytic Combinatorics for Doi-Peliti DSEs (CFAC framework).

Pillar 2 (CRN → DSE mapping).

Tier structure (see rdft/ac/README.md for the full map):

  Tier 1 (core)      — fundamental to the DSE pipeline; re-exported here.
                       dse, lagrange, transfer, algebraic, stratification,
                       bridge.
  Tier 2 (extension) — documented extensions with proofs / tests / papers.
                       admissible, multivariate, log_corrections,
                       signed_projector, tilted, hopf_flow, ope,
                       eft_matching, multipoint, manna_full, manna_2loop,
                       manna_depinning, conserved{,_2,_3}.  Importable as
                       `from rdft.ac.<name> import ...`.
  Tier 3 (research)  — speculative / experimental / failure-mode modules.
                       lerw_*, replica_*, tap_complexity, matrix_model,
                       network_percolation, sandpile_group, correspondence,
                       topology_sketch, manna_dp_anchor_DEPRECATED.  Not
                       re-exported at any package level.

Only Tier 1 symbols are re-exported at the `rdft.ac` level.  Tier 2 and 3
modules are importable via their explicit module paths.
"""
# Tier 1 — core DSE pipeline -----------------------------------------
from .lagrange import (
    LagrangeEquation,
    sir_epidemic,
    first_passage_1d,
    pair_annihilation_dse,
    general_reaction_dse,
)
from .transfer import Singularity, from_lagrange
from .algebraic import NewtonPolygon, AlgebraicSingularity
from .dse import (
    classify_vertices,
    combinatorial_dse_kernel,
    dse_polynomial,
    dse_from_liouvillian,
    upper_critical_dimension,
    dse_summary,
    diagnose_singularity,
    weighted_dse_from_liouvillian,
    ac_scaling_exponent,
    ac_full_derivation,
    upper_critical_dimension_general,
    coupled_dse,
)
from .stratification import (
    puiseux_order,
    canonical_family,
    canonical_z_star,
    on_stratum_C_k,
    phi_from_reactions,
    mass_shift_from_reactions,
    is_dyadic,
    banderier_drmota_status,
    discriminant_in_z,
    discriminant_roots,
    lambda_scgf,
)
from .bridge import (
    bridge_scalar,
    bridge_rank_k,
    bridge_rank_k_numeric,
    bridge_gradient_mass,
    bridge_gradient_diffusion,
    gamma_k_anomalous,
    tau_dressed,
    one_loop_KS,
    one_loop_On,
    one_loop_multicritical,
    upper_critical_dim_for_cusp,
    bubble_critical_dim,
)

__all__ = [
    # lagrange
    'LagrangeEquation',
    'sir_epidemic', 'first_passage_1d',
    'pair_annihilation_dse', 'general_reaction_dse',
    # transfer
    'Singularity', 'from_lagrange',
    # algebraic
    'NewtonPolygon', 'AlgebraicSingularity',
    # dse
    'classify_vertices', 'combinatorial_dse_kernel',
    'dse_polynomial', 'dse_from_liouvillian',
    'upper_critical_dimension', 'dse_summary',
    'diagnose_singularity', 'weighted_dse_from_liouvillian',
    'ac_scaling_exponent', 'ac_full_derivation',
    'upper_critical_dimension_general', 'coupled_dse',
    # stratification
    'puiseux_order', 'canonical_family', 'canonical_z_star',
    'on_stratum_C_k', 'phi_from_reactions',
    'mass_shift_from_reactions',
    'is_dyadic', 'banderier_drmota_status',
    'discriminant_in_z', 'discriminant_roots', 'lambda_scgf',
    # bridge
    'bridge_scalar', 'bridge_rank_k', 'bridge_rank_k_numeric',
    'bridge_gradient_mass', 'bridge_gradient_diffusion',
    'gamma_k_anomalous', 'tau_dressed',
    'one_loop_KS', 'one_loop_On',
    'one_loop_multicritical',
    'upper_critical_dim_for_cusp', 'bubble_critical_dim',
]
