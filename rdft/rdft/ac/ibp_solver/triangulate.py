"""
ibp_solver/triangulate.py
=========================

Laporta-style top-down triangulation of the IBP system.

Algorithm:
  1. Generate IBP identities at a list of seeds.
  2. Sort all integrals by Laporta priority (descending).
  3. For each integral I in priority order:
       - Find an identity that contains I with a nonzero, simple coefficient.
       - Solve that identity for I.
       - Substitute the solution into all remaining identities.
       - Add I to the "reduced" list.
  4. Whatever integrals remain at the end (unreduced) are the masters.
"""
from __future__ import annotations
import sympy as sp
from typing import Dict, List, Tuple
from rdft.ac.ibp_solver.sunset_full import sunset_ibp
from rdft.ac.ibp_solver.reduce import parse_integral_symbol, laporta_priority, is_trivial


d, p2 = sp.symbols('d p^2', positive=True)


def triangulate(seeds: List[Tuple[int,int,int]],
                max_iters: int = 80,
                verbose: bool = True) -> Dict[sp.Symbol, sp.Expr]:
    """
    Triangulate the IBP system from given seeds.
    Returns: dict mapping reduced integrals -> their expressions in
    terms of lower-priority integrals (masters + corner products).
    """
    # 1. Generate identities
    all_ids = []
    for s in seeds:
        all_ids.extend(sunset_ibp(*s).values())
    if verbose:
        print(f"Generated {len(all_ids)} IBP identities from {len(seeds)} seeds.")

    # 2. Find all integral symbols
    all_syms = set()
    for ident in all_ids:
        all_syms |= ident.free_symbols
    integral_syms = [s for s in all_syms if str(s).startswith('I[')]

    # Sort by priority descending (high priority first = reduce first)
    syms_with_prio = [(s, laporta_priority(parse_integral_symbol(s)))
                      for s in integral_syms]
    syms_with_prio.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"Total integrals: {len(integral_syms)}")
        print(f"Highest-priority targets:")
        for s, p in syms_with_prio[:5]:
            print(f"  {s}  prio={p}")

    # 3. Triangulate
    reductions = {}     # integral -> reduced expression
    remaining_ids = list(all_ids)

    for iter_count in range(max_iters):
        # Find the highest-priority integral that:
        #   (a) is not yet reduced
        #   (b) appears in at least one remaining identity
        target = None
        target_id_idx = -1
        target_coef = None

        for sym, prio in syms_with_prio:
            if sym in reductions:
                continue
            if is_trivial(parse_integral_symbol(sym)):
                continue
            # Find an identity containing this symbol
            for i, ident in enumerate(remaining_ids):
                if sym not in ident.free_symbols:
                    continue
                coef = ident.coeff(sym)
                if coef == 0:
                    continue
                # Prefer simple coefficients (linear in d)
                target = sym
                target_id_idx = i
                target_coef = coef
                break
            if target is not None:
                break

        if target is None:
            if verbose:
                print(f"\n[iter {iter_count}] No more reducible integrals.")
            break

        # Solve identity for target
        ident = remaining_ids[target_id_idx]
        try:
            sol = sp.solve(ident, target)
            if not sol:
                # Can't solve uniquely; remove the identity and continue
                remaining_ids.pop(target_id_idx)
                continue
            target_value = sp.simplify(sol[0])
        except Exception as e:
            remaining_ids.pop(target_id_idx)
            continue

        reductions[target] = target_value
        if verbose and iter_count < 8:
            print(f"\n[iter {iter_count}] Reduce {target} (coef = {target_coef}):")
            print(f"  {target} = {target_value}")

        # Substitute back into remaining identities
        new_remaining = []
        for j, other in enumerate(remaining_ids):
            if j == target_id_idx:
                continue
            new_other = other.subs(target, target_value)
            new_other = sp.expand(new_other)
            if new_other != 0:
                new_remaining.append(new_other)
        remaining_ids = new_remaining

        if not remaining_ids:
            if verbose:
                print(f"\n[iter {iter_count}] All identities exhausted.")
            break

    # 4. Find remaining (= masters)
    reduced_set = set(reductions.keys())
    all_set = set(s for s, p in syms_with_prio
                  if not is_trivial(parse_integral_symbol(s)))
    masters = sorted(all_set - reduced_set, key=lambda s: laporta_priority(parse_integral_symbol(s)))

    if verbose:
        print(f"\n{'='*70}")
        print(f" Triangulation summary")
        print(f"{'='*70}")
        print(f"  Iterations:       {iter_count + 1}")
        print(f"  Integrals reduced: {len(reductions)}")
        print(f"  Remaining (masters + irreducible): {len(masters)}")
        print(f"  Remaining identities: {len(remaining_ids)}")
        print(f"\n  Master candidates (lowest priority):")
        for m in masters[:10]:
            prio = laporta_priority(parse_integral_symbol(m))
            print(f"    {m}  prio={prio}")

    return reductions, masters


def main():
    print("=" * 72)
    print(" Laporta triangulation of the sunset IBP system")
    print("=" * 72)

    seeds = [(1,1,1), (2,1,1), (1,2,1), (1,1,2),
             (3,1,1), (1,3,1), (1,1,3),
             (2,2,1), (2,1,2), (1,2,2)]

    reductions, masters = triangulate(seeds, max_iters=100, verbose=True)

    # Show the I[2,1,1] reduction explicitly
    I_211 = sp.Symbol('I[2,1,1]')
    if I_211 in reductions:
        print(f"\n--- Final reduction of I[2,1,1] (after substitutions) ---")
        print(f"  I[2,1,1] = {sp.simplify(reductions[I_211])}")

    I_111 = sp.Symbol('I[1,1,1]')
    if I_111 in masters:
        print(f"\n  I[1,1,1] is identified as a MASTER (as expected).")


if __name__ == '__main__':
    main()
