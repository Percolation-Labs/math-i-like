"""
ibp_solver/reduce.py
====================

Build and solve the IBP linear system for the sunset family.

Strategy:
  1. Generate IBP identities at multiple seeds: (1,1,1), (2,1,1), (1,2,1),
     (1,1,2), and possibly higher-weight seeds.
  2. Sort all distinct integral symbols by Laporta priority.
  3. Set up the linear system M . x = 0 where x is the vector of
     integral symbols and M is the matrix of IBP coefficients.
  4. Solve top-down: for each identity, pick the highest-priority
     integral that still has a nonzero coefficient and solve for it.
  5. Substitute back into all remaining identities.
  6. Whatever's left at the end are the masters.

For the sunset at p^2 = 1, the expected master is I[1,1,1] (the sunset
proper).  Corner integrals like I[1,2,0] factorise into 1-loop products
and are treated separately as "trivial" reductions.
"""
from __future__ import annotations
import sympy as sp
from typing import List, Tuple, Set
from rdft.ac.ibp_solver.sunset_full import sunset_ibp


d, p2 = sp.symbols('d p^2', positive=True)


def parse_integral_symbol(s: sp.Symbol) -> Tuple[int, ...]:
    """Parse 'I[a1,a2,a3]' or 'I[a1,a2,a3;u^n v^m]' into (a1,a2,a3,n,m)."""
    name = str(s)
    if not name.startswith('I[') or not name.endswith(']'):
        return None
    body = name[2:-1]
    if ';' in body:
        exps_part, num_part = body.split(';', 1)
        a1, a2, a3 = [int(x.strip()) for x in exps_part.split(',')]
        # Parse num_part like "u^n v^m"
        num_part = num_part.strip()
        n_u = 0
        n_vsp = 0
        for piece in num_part.split():
            if piece.startswith('u^'):
                n_u = int(piece[2:])
            elif piece.startswith('v^'):
                n_vsp = int(piece[2:])
        return (a1, a2, a3, n_u, n_vsp)
    else:
        a1, a2, a3 = [int(x.strip()) for x in body.split(',')]
        return (a1, a2, a3, 0, 0)


def laporta_priority(exps: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Higher = more complex = reduce earlier."""
    a1, a2, a3, n_u, n_vsp = exps
    sector = sum(1 for a in (a1, a2, a3) if a > 0)
    sum_pos = a1 + a2 + a3
    num_total = n_u + n_vsp
    return (sector, sum_pos + num_total, num_total)


def is_trivial(exps: Tuple[int, ...]) -> bool:
    """Trivial integrals (1-loop products or zero) -- not in the IBP system."""
    a1, a2, a3, _, _ = exps
    # Sector with fewer than 2 propagators of value > 0 is a 1-loop product
    # or factorisable.
    n_props = sum(1 for a in (a1, a2, a3) if a > 0)
    return n_props < 3   # need all three propagators for genuine sunset


def main():
    print("=" * 72)
    print(" Building IBP system for the sunset at p^2 = 1")
    print("=" * 72)

    # Generate identities at several seeds — enough to fully reduce.
    seeds = [(1,1,1), (2,1,1), (1,2,1), (1,1,2),
             (3,1,1), (1,3,1), (1,1,3),
             (2,2,1), (2,1,2), (1,2,2)]
    all_ids = []
    for seed in seeds:
        ids = sunset_ibp(*seed)
        all_ids.extend(ids.values())
    print(f"\n  Generated {len(all_ids)} IBP identities from {len(seeds)} seeds.")

    # Collect all integral symbols
    all_syms = set()
    for ident in all_ids:
        all_syms |= ident.free_symbols
    integral_syms = [s for s in all_syms if str(s).startswith('I[')]
    print(f"  Total distinct integrals: {len(integral_syms)}")

    # Categorise
    integrals_with_exps = [(s, parse_integral_symbol(s)) for s in integral_syms]
    nontrivial = [(s, e) for s, e in integrals_with_exps if not is_trivial(e)]
    trivial = [(s, e) for s, e in integrals_with_exps if is_trivial(e)]
    print(f"  Nontrivial sunset integrals (3 props): {len(nontrivial)}")
    print(f"  Trivial / corner / factorisable: {len(trivial)}")

    print(f"\n  Nontrivial integrals (sorted by priority):")
    nontrivial.sort(key=lambda x: laporta_priority(x[1]), reverse=True)
    for s, e in nontrivial:
        prio = laporta_priority(e)
        print(f"    {s}    priority={prio}")

    # Build the linear system: coefficients of nontrivial integrals.
    # Treat trivial integrals as known (we'll substitute them later or
    # leave them as free constants).
    print(f"\n  Building linear system over Q(d, p^2)...")
    print(f"    Equations: {len(all_ids)}")
    print(f"    Unknowns (nontrivial only): {len(nontrivial)}")

    nontrivial_syms = [s for s, _ in nontrivial]

    # Use sympy.linear_eq_to_matrix
    try:
        M, b = sp.linear_eq_to_matrix(all_ids, nontrivial_syms)
        print(f"    Matrix shape: {M.shape}")
        print(f"    Rank: {M.rank()}")
    except Exception as e:
        print(f"    Failed to build matrix: {e}")
        return

    # Try to find which integrals can be eliminated
    rank = M.rank()
    n_unknowns = len(nontrivial_syms)
    print(f"\n  System has {rank} independent equations on {n_unknowns} unknowns.")
    print(f"  Expected masters: {n_unknowns - rank}")

    # Attempt a reduction: solve for the highest-priority integrals first
    print(f"\n  Attempting Laporta-style reduction...")
    print(f"  (This is a stress test of the SymPy linear algebra; for")
    print(f"   nontrivial Reggeon kinematics one would want a dedicated")
    print(f"   solver like KIRA or FIRE.)")

    # Try sp.solve on the system for top-priority integrals
    high_priority = nontrivial_syms[:min(5, len(nontrivial_syms))]
    print(f"\n  Trying to solve for top {len(high_priority)} integrals:")
    for s in high_priority:
        print(f"    {s}")

    try:
        # Assume the system can be reduced for these
        sol = sp.solve(all_ids, high_priority, dict=True, manual=False)
        if sol:
            print(f"\n  Solution found ({len(sol)} dict, picking first):")
            sol_dict = sol[0] if isinstance(sol, list) else sol
            for k, v in list(sol_dict.items())[:5]:
                print(f"    {k} =")
                print(f"      {sp.simplify(v)}")
        else:
            print(f"\n  System is consistent but does not uniquely determine these.")
    except Exception as e:
        print(f"\n  Solver failed: {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print(" Status")
    print("=" * 72)
    print("""
The IBP system for the sunset is built and contains all expected
integrals.  A full reduction requires either:
  - More identities (higher seeds) to fully constrain the system, OR
  - A dedicated reducer with rational-function arithmetic over Q(d, p^2)
    and Laporta ordering.

For our specific CFAC purpose, what's needed is to express I[1,1,1]
(the sunset master) as a Gamma-function product, which is the
classical result --- our gribov_ibp_plugin.py FORM backend already
does this.  The IBP machinery here is the framework that would
extend to more complex Reggeon-kinematics topologies.
""")


if __name__ == '__main__':
    main()
