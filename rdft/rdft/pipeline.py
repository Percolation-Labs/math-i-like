"""
rdft.pipeline
=============
End-to-end pipeline: CRN → critical exponents via two routes.

Route 1 (Doi-Peliti): CRN → Liouvillian → diagrams → Symanzik → integral → RG → exponents
Route 2 (Analytic Combinatorics): CRN → DSE → Lagrange eq → branch point → transfer → exponents

Both routes arrive at the same answer because both detect the same singularity.

Usage:
    from rdft.pipeline import analyze
    results = analyze('pair_annihilation')
    results = analyze('gribov')          # BRW process
"""

from __future__ import annotations
from typing import Dict, Optional
import sympy as sp

from .core.reaction_network import ReactionNetwork
from .core.generators import Liouvillian
from .core.expansion import FeynmanExpansion, extract_vertices
from .graphs.incidence import FeynmanGraph
from .graphs.render import render_ascii
from .integrals.symanzik import SymanzikPolynomials
from .rg.rg_functions import KnownResults
from .ac.lagrange import LagrangeEquation, first_passage_1d, pair_annihilation_dse
from .ac.transfer import Singularity, from_lagrange
from .ac.correspondence import CorrespondenceTable


# ================================================================== #
#  Standard process registry                                          #
# ================================================================== #

PROCESSES = {
    'pair_annihilation': {
        'factory': ReactionNetwork.pair_annihilation,
        'known': KnownResults.pair_annihilation,
        'ac_lagrange': pair_annihilation_dse,
        'description': '2A → ∅ (pair annihilation)',
        'd_c': 2,
    },
    'coagulation': {
        'factory': ReactionNetwork.coagulation,
        'known': KnownResults.coagulation,
        'description': '2A → A (coagulation)',
        'd_c': 2,
    },
    'gribov': {
        'factory': ReactionNetwork.gribov,
        'known': KnownResults.bws_hypercubic,
        'description': 'A→2A, A→∅, 2A→A (Gribov / BRW process)',
        'd_c': 4,
    },
    'contact_process': {
        'factory': ReactionNetwork.contact_process,
        'known': KnownResults.directed_percolation,
        'description': 'A→2A, 2A→∅ (contact process / DP class)',
        'd_c': 4,
    },
}


def analyze(process_name: str,
            max_loops: int = 1,
            d: sp.Symbol = None,
            verbose: bool = True) -> Dict:
    """
    Full analysis pipeline for a reaction-diffusion process.

    Parameters
    ----------
    process_name : key in PROCESSES dict, or 'custom' with network kwarg
    max_loops : maximum loop order for diagram generation
    d : symbolic dimension variable
    verbose : print progress

    Returns
    -------
    Dict with all results from both routes.
    """
    if d is None:
        d = sp.Symbol('d', positive=True)

    proc = PROCESSES[process_name]
    network = proc['factory']()
    known = proc['known']() if proc.get('known') else {}

    results = {
        'process': proc['description'],
        'network': network,
        'd_c': proc.get('d_c'),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"RDFT Analysis: {proc['description']}")
        print(f"{'='*60}")

    # ---- Route 1: Doi-Peliti ----
    if verbose:
        print(f"\n--- Route 1: Doi-Peliti Field Theory ---")

    # Step 1: Liouvillian
    L = Liouvillian(network)
    results['liouvillian'] = L
    results['vertices'] = L.vertices

    if verbose:
        print(f"Liouvillian Q = {L.total}")
        print(f"Interaction vertices:")
        for (m, n), c in L.vertices.items():
            if m + n >= 3:
                print(f"  φ̃^{m}φ^{n} : coupling = {c}")

    # Step 2: Diagram generation
    exp = FeynmanExpansion(network, max_loops=max_loops)
    diagrams = exp.expand()
    results['diagrams'] = diagrams

    if verbose:
        for loop_order, diag_list in diagrams.items():
            print(f"\nLoop order L={loop_order}: {len(diag_list)} distinct 1PI diagram(s)")
            for i, diag in enumerate(diag_list[:5]):  # show first 5
                g = diag['graph']
                print(f"  [{i+1}] V={g.n_vertices_int}, E_int={g.n_internal_edges}, "
                      f"E_ext={g.n_external_edges}, L={g.L}")

    # Step 3: Graph polynomials (for canonical one-loop diagram)
    if 1 in diagrams and diagrams[1]:
        g0 = diagrams[1][0]['graph']
        sym = SymanzikPolynomials(g0)
        results['Psi'] = sym.Psi
        results['graph_example'] = g0

        if verbose:
            print(f"\nCanonical one-loop diagram: {g0}")
            print(f"  Ψ = {sym.Psi}")
            print(f"  σ_d = {g0.degree_of_divergence()}")
            print(f"  K(α) = {g0.kirchhoff_polynomial()}")

    # Step 4: Known RG results
    results['known_results'] = known

    if verbose and known:
        print(f"\nKnown results (literature):")
        for key, val in known.items():
            print(f"  {key} = {val}")

    # ---- Route 2: Analytic Combinatorics ----
    if verbose:
        print(f"\n--- Route 2: Analytic Combinatorics ---")

    # AUTOMATIC AC route: derive DSE from Liouvillian, no manual input
    ac_results = {}

    try:
        from .ac.dse import dse_from_liouvillian, upper_critical_dimension
        from .ac.algebraic import AlgebraicSingularity

        # Step 1: Auto-construct DSE from Liouvillian
        lagrange_eq = dse_from_liouvillian(L)
        ac_results['lagrange_equation'] = lagrange_eq
        ac_results['phi'] = lagrange_eq.phi
        ac_results['d_c_from_vertices'] = upper_critical_dimension(L.vertices)

        if verbose:
            print(f"Auto-DSE: G = G₀ · ({lagrange_eq.phi})")
            print(f"d_c from vertex dimensions: {ac_results['d_c_from_vertices']}")

        # Step 2: Algebraic singularity analysis
        F = lagrange_eq.T - lagrange_eq.z * lagrange_eq.phi
        analysis = AlgebraicSingularity(F, lagrange_eq.T, lagrange_eq.z)

        bp = analysis.dominant_branch_point()
        if bp:
            T_star, z_star = bp
            ac_results['T_star'] = T_star
            ac_results['z_star'] = z_star

            pq = analysis.puiseux_exponent(T_star, z_star)
            ac_results['puiseux_exponent'] = pq

            info = analysis.singularity_type()
            ac_results['singularity_type'] = info['type']

            if verbose:
                print(f"Branch point: G* = {T_star}, G₀* = {z_star}")
                print(f"Singularity: {info['type']} (Puiseux exponent {pq})")

            # Step 3: Transfer theorem
            sing = analysis.to_singularity()
            asymp = sing.coefficient_asymptotics_simplified()
            ac_results['power_law_exponent'] = asymp['power_law_exponent']

            dens = sing.density_exponent()
            ac_results['density_exponent'] = dens

            if verbose:
                print(f"Transfer: [G₀^n] ~ n^{{{asymp['power_law_exponent']}}}")

    except Exception as e:
        if verbose:
            print(f"Auto-AC analysis: {e}")
        import traceback
        traceback.print_exc()

    # For all processes: use first-passage AC route for annihilation-type
    if process_name in ('pair_annihilation', 'coagulation'):
        fp = first_passage_1d()
        fp_sing = from_lagrange(fp)
        fp_asymp = fp_sing.coefficient_asymptotics_simplified()
        fp_dens = fp_sing.density_exponent()
        ac_results['first_passage_exponent'] = fp_asymp['power_law_exponent']
        ac_results['first_passage_density'] = fp_dens

        if verbose:
            print(f"\nFirst-passage route (AC for 2A→∅):")
            print(f"  F(z) = z·(1+F²)/2  [Lagrange eq]")
            print(f"  Branch: square-root at z*=1")
            print(f"  [z^n] ~ n^{{{fp_asymp['power_law_exponent']}}}")
            print(f"  ρ(t) ~ t^{{{fp_dens.get('density_exponent_1d', '?')}}}")

    results['ac'] = ac_results

    # ---- Correspondence Table ----
    if verbose:
        print(f"\n--- AC ↔ QFT Correspondence ---")

    corr = CorrespondenceTable(proc['description'])
    corr.add('Lagrange equation T = z·φ(T)',
             'Dyson-Schwinger equation G = G₀·Φ(G)')
    corr.add('Branch point (IFT failure)',
             'Landau pole / non-perturbative scale')
    corr.add('Singularity type',
             'Universality class')
    corr.add('Transfer theorem: n^{-α-1}',
             'Loop amplitude growth')

    if known and ac_results:
        rg_alpha = known.get('alpha')
        ac_alpha = ac_results.get('power_law_exponent')
        if rg_alpha is not None and ac_alpha is not None:
            corr.add('AC coefficient exponent',
                     'RG density exponent',
                     ac_val=ac_alpha,
                     qft_val=rg_alpha)

    results['correspondence'] = corr

    if verbose:
        print(corr.summary())

    # ---- Spectral Dimension Extension ----
    if verbose:
        d_c = proc.get('d_c', '?')
        print(f"\n--- Spectral Dimension Extension ---")
        print(f"Upper critical dimension: d_c = {d_c}")
        print(f"On graph G with spectral dimension d_s:")
        if known:
            alpha_expr = known.get('alpha', '?')
            print(f"  Exponent: α = {alpha_expr}  (with d → d_s)")
        print(f"  If d_s < d_c: fluctuation-dominated regime")
        print(f"  If d_s ≥ d_c: mean-field regime")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis complete.")
        print(f"{'='*60}")

    return results


# ================================================================== #
#  Convenience: run BRW worked example                                #
# ================================================================== #

def brw_worked_example():
    """
    Complete worked example: Branching Random Walk (Gribov process).

    This reproduces the analysis from:
    - Amarteifio (2019) PhD thesis, Chapter 3
    - Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590

    The BRW process: A → 2A (branching), A → ∅ (death), 2A → A (coagulation)
    Critical regime: birth rate = death rate (r = e - s = 0)
    Observable: volume explored (distinct sites visited)
    """
    print("\n" + "="*70)
    print("BRANCHING RANDOM WALK — Complete Worked Example")
    print("Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590")
    print("="*70)

    # 1. Define the process
    from .core.reaction_network import ReactionNetwork
    net = ReactionNetwork.gribov()
    print(f"\nProcess: {net.summary()}")

    # 2. Full pipeline
    results = analyze('gribov', max_loops=1, verbose=True)

    # 3. BRW-specific scaling predictions
    print("\n--- BRW Scaling Predictions (Bordeu+ 2019) ---")
    from .graphs.spectral import brw_scaling_exponents, KNOWN_SPECTRAL_DIMENSIONS

    d_s = sp.Symbol('d_s', positive=True)
    p = sp.Symbol('p', positive=True, integer=True)
    scaling = brw_scaling_exponents(d_s)

    print(f"\nFor d_s < d_c = 4 (fluctuation-dominated):")
    print(f"  ⟨a^p⟩(t) ~ t^{{(p·d_s - 2)/2}}")
    print(f"  ⟨a^p⟩(L) ~ L^{{p·d_s - 2}}")
    print(f"  P(a) ~ a^{{-(1+2/d_s)}}")

    print(f"\nFor d_s ≥ 4 (mean-field):")
    print(f"  ⟨a^p⟩(t) ~ t^{{2p-1}}")
    print(f"  P(a) ~ a^{{-3/2}}")

    print(f"\nPredicted exponents by graph type:")
    print(f"{'Graph':<25s} {'d_s':>6s} {'⟨a⟩ ~ t^α':>12s} {'P(a) ~ a^β':>12s}")
    print("-" * 58)

    for name, data in KNOWN_SPECTRAL_DIMENSIONS.items():
        ds_val = data['d_s']
        if ds_val < 4:
            alpha_1 = (ds_val - 2) / 2
            beta = -(1 + 2/ds_val)
        else:
            alpha_1 = 1.0
            beta = -1.5
        print(f"{name:<25s} {ds_val:>6.2f} {alpha_1:>12.3f} {beta:>12.3f}")

    return results


if __name__ == '__main__':
    brw_worked_example()
