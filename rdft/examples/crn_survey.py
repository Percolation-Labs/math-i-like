#!/usr/bin/env python3
"""
CRN Survey: Test the AC Pipeline on Complex Reaction Networks
==============================================================

Runs every CRN through the full pipeline:
    Stoichiometry -> Liouvillian -> Vertices -> DSE kernel phi(G)
    -> Branch point -> Newton polygon -> Puiseux exponent -> d_c
    -> Transfer theorem -> Universality class

Organized in three tiers:
    Tier 1: Known exact results (validate the pipeline)
    Tier 2: Known universality class (deeper test)
    Tier 3: Open problems (predictions to verify numerically)

Note on d_c: The `upper_critical_dimension` function returns the NAIVE d_c
from dimensional analysis: d_c = 4/(m+n-2) for the most relevant (smallest
leg-count) interaction vertex. For pure annihilation kA->0, a Ward identity
(particle number can only decrease) cancels the most relevant vertex, reducing
d_c from 4/(k-1) to 2/(k-1). This Ward identity correction is separate from
the automatic pipeline.
"""

import sympy as sp
from sympy import Symbol, Rational, simplify, expand
import sys
import signal

from rdft.core.reaction_network import ReactionNetwork
from rdft.core.generators import Liouvillian
from rdft.ac.dse import (
    combinatorial_dse_kernel,
    upper_critical_dimension,
    classify_vertices,
    dse_from_liouvillian,
    diagnose_singularity,
)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Branch point analysis timed out")


def analyze_crn(net: ReactionNetwork, expected: dict = None,
                numeric_rates: dict = None, timeout_sec: int = 15) -> dict:
    """
    Run the full AC pipeline on a reaction network.

    Parameters
    ----------
    net : ReactionNetwork
    expected : dict with optional keys 'd_c', 'singularity_type', 'phi_degree'
    numeric_rates : dict mapping rate symbol names to float values for branch point analysis
    timeout_sec : timeout in seconds for branch point computation
    """
    print(f"\n{'='*70}")
    print(f"  {net.name}")
    print(f"{'='*70}")

    # Step 1: Liouvillian
    L = Liouvillian(net)
    print(f"\nReactions:")
    for rxn in net.reactions:
        print(f"  {rxn}  (rate={rxn.rate})")

    print(f"\nLiouvillian Q = {L.total}")

    # Step 2: Vertices
    verts = L.vertices
    is_multispecies = net.n_species > 1
    if is_multispecies:
        print(f"\nVertices (multi-species, {len(verts)} terms):")
        for key, g in sorted(verts.items(), key=str):
            total_legs = sum(sum(pair) for pair in key)
            label = 'interaction' if total_legs >= 3 else 'mass'
            species_str = ', '.join(
                f"{net.species[i].name}:({m},{n})"
                for i, (m, n) in enumerate(key)
            )
            print(f"  [{species_str}]: {g}  [{label}, {total_legs} legs]")
    else:
        classified = classify_vertices(verts)
        print(f"\nVertices ({len(classified['interaction'])} interaction, "
              f"{len(classified['mass'])} mass):")
        for (m, n), g in sorted(verts.items(), key=str):
            label = 'interaction' if m + n >= 3 else 'mass/propagator'
            print(f"  phi_tilde^{m} phi^{n}: {g}  [{label}]")

    # Step 3: DSE kernel
    G = Symbol('G')
    if is_multispecies:
        # For multi-species, build an effective DSE by summing over all vertices
        # with total leg count >= 3. Each vertex contributes independently
        # (no cancellation between different species structures).
        phi_terms = {}
        for key, g in verts.items():
            total_legs = sum(sum(pair) for pair in key)
            if total_legs >= 3:
                power = total_legs - 2
                phi_terms[power] = phi_terms.get(power, sp.S.Zero) + abs(g)
        phi = sp.S.One
        for power, coeff in sorted(phi_terms.items()):
            phi += coeff * G**power
        phi = sp.expand(phi)
        print(f"\nEffective DSE kernel (total legs, |g|): phi(G) = {phi}")
        print(f"  [Multi-species: effective kernel uses |coupling| per leg count]")
    else:
        phi = combinatorial_dse_kernel(verts, G)
        print(f"\nDSE kernel: phi(G) = {phi}")

    try:
        poly = sp.Poly(phi, G)
        degree = poly.degree()
    except Exception:
        degree = '?'
    print(f"  Degree in G: {degree}")

    # Step 4: Upper critical dimension
    if is_multispecies:
        # For multi-species, compute d_c from total leg count
        d_c_values = []
        for key, g in verts.items():
            total_legs = sum(sum(pair) for pair in key)
            if total_legs >= 3:
                d_c_values.append(Rational(4, total_legs - 2))
        d_c = max(d_c_values) if d_c_values else sp.oo
    else:
        d_c = upper_critical_dimension(verts)
    print(f"Upper critical dimension (naive): d_c = {d_c}")

    result = {
        'name': net.name,
        'phi': phi,
        'degree': degree,
        'd_c': d_c,
        'n_interaction': len(classified['interaction']) if not is_multispecies else
                         sum(1 for k in verts if sum(sum(p) for p in k) >= 3),
    }

    # Run full singularity diagnostics (single-species only)
    if not is_multispecies:
        diag = diagnose_singularity(verts)
        result['singularity_class'] = diag['singularity_class']
        result['diagnosis'] = diag['diagnosis']
        result['transfer_exponent_predicted'] = diag.get('transfer_exponent')
        if diag.get('d_c_physical'):
            result['d_c_physical'] = diag['d_c_physical']
        print(f"\nAC Diagnosis: {diag['singularity_type']}")
        print(f"  {diag['diagnosis']}")
        for w in diag.get('warnings', []):
            print(f"  WARNING: {w}")

    # Step 5: Branch point analysis (with timeout and optional numeric substitution)
    if is_multispecies:
        # For multi-species, use the effective single-variable DSE
        print(f"\n[Multi-species: using effective single-variable DSE for singularity analysis]")

    try:
        # Build numeric version if needed
        if numeric_rates and not is_multispecies:
            net_num = _substitute_rates(net, numeric_rates)
            L_num = Liouvillian(net_num)
        else:
            L_num = L

        # For multi-species or single-species, build Lagrange equation from phi
        from rdft.ac.lagrange import LagrangeEquation
        G0 = Symbol('G0')

        if is_multispecies:
            # Substitute numeric rates directly into phi
            phi_num = phi
            if numeric_rates:
                for name, val in numeric_rates.items():
                    phi_num = phi_num.subs(sp.Symbol(name, positive=True), val)
                    phi_num = phi_num.subs(sp.Symbol(name), val)
            dse = LagrangeEquation(phi_num, T_var=G, z_var=G0)
        else:
            dse = dse_from_liouvillian(L_num)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        try:
            bp = dse.branch_point()
            signal.alarm(0)

            if isinstance(bp, list):
                print(f"\n{len(bp)} branch point(s):")
                for i, (T_s, z_s) in enumerate(bp[:4]):
                    print(f"  [{i+1}] G* = {T_s}, G_0* = {z_s}")
                bp = bp[0]
            else:
                T_s, z_s = bp
                print(f"\nBranch point: G* = {T_s}, G_0* = {z_s}")

            result['branch_point'] = bp

            sing = dse.singularity_type()
            stype = sing.get('type', '?')
            print(f"Singularity type: {stype}")
            result['singularity_type'] = stype

            if stype == 'square_root_branch':
                print(f"  -> Transfer: [G_0^n] ~ n^{{-3/2}}")
                result['transfer_exponent'] = Rational(-3, 2)
            elif stype == 'cubic_branch':
                print(f"  -> Transfer: [G_0^n] ~ n^{{-4/3}}")
                result['transfer_exponent'] = Rational(-4, 3)

        except TimeoutError:
            signal.alarm(0)
            print(f"\nBranch point: TIMEOUT (>{timeout_sec}s)")
            result['error'] = 'timeout'
        except Exception as e:
            signal.alarm(0)
            print(f"\nBranch point analysis: {e}")
            result['error'] = str(e)

    except Exception as e:
        print(f"\nAnalysis error: {e}")
        result['error'] = str(e)

    # Step 6: Validation
    if expected:
        print(f"\nValidation:")
        all_pass = True
        for key in ['d_c', 'singularity_type', 'phi_degree']:
            if key not in expected:
                continue
            if key == 'phi_degree':
                got = degree
            elif key == 'd_c':
                got = d_c
            else:
                got = result.get(key, '?')
            match = (got == expected[key])
            status = 'PASS' if match else 'FAIL'
            if not match:
                all_pass = False
            print(f"  {key}: got {got}, expected {expected[key]} -> {status}")
        result['all_pass'] = all_pass

    return result


def _substitute_rates(net, rate_map):
    """Create a copy of the network with numeric rate values."""
    from rdft.core.reaction_network import ReactionNetwork, Reaction, Species
    new_rxns = []
    for rxn in net.reactions:
        rate = rxn.rate
        for name, val in rate_map.items():
            rate = rate.subs(sp.Symbol(name, positive=True), val)
            rate = rate.subs(sp.Symbol(name), val)
        new_rxns.append(Reaction(
            dict(rxn.reactants), dict(rxn.products),
            rate=rate, name=rxn.name
        ))
    return ReactionNetwork(list(net.species), new_rxns, name=net.name)


# ================================================================== #
#  Tier 1: Known exact results                                        #
# ================================================================== #

def tier1_known_exact():
    """CRNs with known exact critical exponents."""
    print("\n" + "#" * 70)
    print("#  TIER 1: Known Exact Results")
    print("#" * 70)
    results = {}

    # For pure annihilation kA->0, naive d_c = 4/(k-1) from smallest vertex.
    # Physical d_c = 2/(k-1) after Ward identity. We test the naive d_c here.

    # 1a. Pair annihilation 2A -> 0
    # Naive d_c: smallest vertex is phi_tilde phi^2 (3 legs) -> d_c = 4
    # Physical d_c = 2 (Ward identity cancels cubic vertex contribution)
    results['2A->0'] = analyze_crn(
        ReactionNetwork.pair_annihilation(),
        expected={'d_c': 4, 'singularity_type': 'square_root_branch', 'phi_degree': 2},
        numeric_rates={'lambda': 1},
    )

    # 1b. Coagulation 2A -> A
    results['2A->A'] = analyze_crn(
        ReactionNetwork.coagulation(),
        expected={'d_c': 4, 'singularity_type': 'square_root_branch'},
        numeric_rates={'chi': 1},
    )

    # 1c. Triplet annihilation 3A -> 0
    # Naive d_c: smallest vertex has 4 legs -> d_c = 2
    # Physical d_c = 1 (Ward identity)
    results['3A->0'] = analyze_crn(
        ReactionNetwork.triplet_annihilation(),
        expected={'d_c': 2, 'singularity_type': 'square_root_branch'},
        numeric_rates={'lambda': 1},
    )

    # 1d. 4-particle annihilation 4A -> 0
    # Naive d_c: smallest vertex has 5 legs -> d_c = 4/3
    results['4A->0'] = analyze_crn(
        ReactionNetwork.k_particle_annihilation(4),
        expected={'d_c': Rational(4, 3)},
        numeric_rates={'lambda': 1},
    )

    # 1e. 5-particle annihilation 5A -> 0
    # Naive d_c: smallest vertex has 6 legs -> d_c = 1
    results['5A->0'] = analyze_crn(
        ReactionNetwork.k_particle_annihilation(5),
        expected={'d_c': 1},
        numeric_rates={'lambda': 1},
    )

    # 1f. Gribov process (BRW A-sector)
    results['Gribov'] = analyze_crn(
        ReactionNetwork.gribov(),
        expected={'d_c': 4, 'singularity_type': 'square_root_branch', 'phi_degree': 2},
        numeric_rates={'beta': 2, 'epsilon': 1, 'chi': 1},
    )

    return results


# ================================================================== #
#  Tier 2: Known universality class                                   #
# ================================================================== #

def tier2_known_class():
    """CRNs with known universality class."""
    print("\n" + "#" * 70)
    print("#  TIER 2: Known Universality Class")
    print("#" * 70)
    results = {}

    # 2a. Contact process: A -> 2A, 2A -> 0 (DP class)
    results['Contact'] = analyze_crn(
        ReactionNetwork.contact_process(),
        expected={'d_c': 4, 'singularity_type': 'square_root_branch'},
        numeric_rates={'lambda': 2, 'mu': 1},
    )

    # 2b. Schlogl II: 2A -> 3A, A -> 0 (DP class, different CRN)
    results['Schlogl II'] = analyze_crn(
        ReactionNetwork.schlogl_second(),
        expected={'d_c': 4, 'singularity_type': 'square_root_branch'},
        numeric_rates={'sigma': 1, 'mu': 1},
    )

    # 2c. BARW-odd: A -> 2A, 2A -> 0 (DP class, same as contact)
    results['BARW-odd'] = analyze_crn(
        ReactionNetwork.barw_odd(),
        expected={'d_c': 4, 'singularity_type': 'square_root_branch'},
        numeric_rates={'sigma': 2, 'lambda': 1},
    )

    # 2d. Schlogl I: 0 -> A, A -> 0, 2A -> 3A, 3A -> 2A (Ising at critical pt)
    results['Schlogl I'] = analyze_crn(
        ReactionNetwork.schlogl_first(),
        expected={'d_c': 4},
        numeric_rates={'alpha': 1, 'delta': 1, 'sigma': 1, 'lambda': 1},
    )

    # 2e. Lotka-Volterra (multi-species)
    results['Lotka-Volterra'] = analyze_crn(
        ReactionNetwork.lotka_volterra(),
        expected={},
        numeric_rates={'sigma': 1, 'lambda': 1, 'mu': 1},
    )

    # 2f. Reversible annihilation 2A <-> C
    results['2A<->C'] = analyze_crn(
        ReactionNetwork.reversible_annihilation(),
        expected={},
        numeric_rates={'lambda': 1, 'sigma': 1},
    )

    return results


# ================================================================== #
#  Tier 3: Open problems                                              #
# ================================================================== #

def tier3_open_problems():
    """CRNs where the universality class is unknown or poorly understood."""
    print("\n" + "#" * 70)
    print("#  TIER 3: Open Problems — AC Predictions")
    print("#" * 70)
    results = {}

    # 3a. BARW-even: A -> 3A, 2A -> 0 (Parity-Conserving class)
    # RG epsilon-expansion FAILS for this system.
    # Naive d_c: cubic vertex (3 legs) from A->3A gives d_c = 4,
    # but the quartic from 2A->0 gives d_c = 2.
    # The most relevant is cubic -> naive d_c = 4.
    # But with parity conservation, the cubic vertex may be irrelevant.
    results['BARW-even'] = analyze_crn(
        ReactionNetwork.barw_even(),
        expected={'d_c': 4},
        numeric_rates={'sigma': 1, 'lambda': 1},
    )

    # 3b. PCPD: 2A -> 3A, 2A -> 0 (controversial universality)
    results['PCPD'] = analyze_crn(
        ReactionNetwork.pcpd(),
        expected={'d_c': 4},
        numeric_rates={'sigma': 1, 'lambda': 1},
    )

    # 3c. Prion propagation: H+M -> 2M, 0 -> H, H -> 0, M -> 0
    # First field-theoretic analysis. AC predicts DP for mu_H > 0.
    results['Prion'] = analyze_crn(
        ReactionNetwork.prion_propagation(minimal=True),
        expected={},
        numeric_rates={'beta': 1, 'lambda': 1, 'mu_H': 0.5, 'mu_M': 0.5},
    )

    # 3d. Michaelis-Menten: E+S -> ES, ES -> E+S, ES -> E+P
    # No full Doi-Peliti RG exists for d > 1.
    results['Michaelis-Menten'] = analyze_crn(
        ReactionNetwork.michaelis_menten(),
        expected={},
        numeric_rates={'k_1': 1, 'k_m1': 0.5, 'k_2': 1},
    )

    return results


# ================================================================== #
#  Summary                                                             #
# ================================================================== #

def print_summary(all_results: dict):
    """Print comparison table."""
    print("\n\n" + "=" * 95)
    print("  SUMMARY: AC Pipeline Results for All CRNs")
    print("=" * 95)

    header = (f"  {'CRN':<18s} {'d_c(naive)':>10s} {'d_c(phys)':>10s} "
              f"{'phi deg':>8s} {'singularity':>22s} {'status':>8s}")
    print(header)
    print("-" * 95)

    # Physical d_c values from the literature
    phys_dc = {
        '2A->0': '2', '2A->A': '2', '3A->0': '1', '4A->0': '2/3',
        '5A->0': '1/2', 'Gribov': '4', 'Contact': '4', 'Schlogl II': '4',
        'BARW-odd': '4', 'Schlogl I': '4', 'Lotka-Volterra': '4',
        '2A<->C': '2', 'BARW-even': '2*', 'PCPD': '~2?',
    }

    for name, r in all_results.items():
        d_c = str(r.get('d_c', '?'))
        pdc = phys_dc.get(name, '?')
        deg = str(r.get('degree', '?'))
        stype = r.get('singularity_type', r.get('error', '?'))[:22]
        status = 'PASS' if r.get('all_pass') is True else (
                 'FAIL' if r.get('all_pass') is False else '-')
        print(f"  {name:<18s} {d_c:>10s} {pdc:>10s} {deg:>8s} {stype:>22s} {status:>8s}")

    print(f"\n  * BARW-even: d_c=2 formally, but epsilon-expansion fails (PC class)")
    print(f"  ? PCPD: d_c unknown, possibly ~2")

    # Key observations
    print(f"\n\nKey Observations:")
    print(f"  1. ALL single-species CRNs with a cubic vertex get naive d_c = 4")
    print(f"  2. Pure annihilation kA->0: naive d_c = 4/(k-1), physical d_c = 2/(k-1)")
    print(f"     The factor-of-2 reduction comes from the Ward identity")
    print(f"  3. ALL polynomial DSE kernels have square-root branch points")
    print(f"     This is because phi''(G*) != 0 generically (quadratic tangency)")
    print(f"  4. Parity conservation (BARW-even) must modify the singularity")
    print(f"     structure in a way not captured by the naive DSE kernel")

    # Literature reference table
    print(f"\n\nLiterature Reference:")
    print(f"  {'CRN':<18s} {'d_c':>5s} {'class':>12s} {'known exponent':>30s}")
    print(f"  {'-'*70}")
    lit = [
        ('2A->0',        '2',   'diff-limited', 'n(t) ~ t^{-d/2} exact'),
        ('3A->0',        '1',   'diff-limited', 'n(t) ~ t^{-d/2} for d<1'),
        ('Gribov/BRW',   '4',   'DP/BRW',       '<a^p> ~ t^{(pd-2)/2}'),
        ('Contact',      '4',   'DP',            'beta~0.277 (d=1)'),
        ('Schlogl II',   '4',   'DP',            'same as Contact'),
        ('Schlogl I',    '4',   'Ising',         'beta=1/2 (MF)'),
        ('Lotka-Volterra','4',  'DP(extinct)',    'DP exponents'),
        ('2A<->C',       '2',   '= 2A->0',      'delta_n ~ t^{-d/2}'),
        ('BARW-even',    '2*',  'PC',            'beta~0.92, z~1.77 (d=1)'),
        ('PCPD',         '~2?', '???',           'beta~0.58-0.64 (d=1)'),
    ]
    for name, dc, cls_, exp in lit:
        print(f"  {name:<18s} {dc:>5s} {cls_:>12s} {exp:>30s}")


def main():
    print()
    print("+" + "=" * 68 + "+")
    print("|  RDFT: CRN Survey — Testing the AC Pipeline                       |")
    print("|  on Complex Chemical Reaction Networks from the Literature         |")
    print("+" + "=" * 68 + "+")

    all_results = {}

    r1 = tier1_known_exact()
    all_results.update(r1)

    r2 = tier2_known_class()
    all_results.update(r2)

    r3 = tier3_open_problems()
    all_results.update(r3)

    print_summary(all_results)

    passes = sum(1 for r in all_results.values() if r.get('all_pass') is True)
    fails  = sum(1 for r in all_results.values() if r.get('all_pass') is False)
    total  = len(all_results)
    print(f"\n\nResults: {passes} passed, {fails} failed, "
          f"{total - passes - fails} unchecked, {total} total")

    return all_results


if __name__ == '__main__':
    main()
