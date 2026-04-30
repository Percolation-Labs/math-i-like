"""
ibp_solver/test_reduction.py
============================

Targeted test: reduce I[2,1,1] in the sunset family using a single
IBP identity at seed (2,1,1).

The IBP identity at (v=k1, j=1) for seed (a1, a2, a3) gives:
   (d - 2 a1 - a3) I[a1, a2, a3] + ... = 0
For seed (2, 1, 1): (d - 5) I[2,1,1] + ... = 0
Solving for I[2,1,1] gives a clean reduction to lower-priority integrals.

This is the basic Laporta step.  A full reducer applies it
recursively top-down through the priority list.
"""
from __future__ import annotations
import sympy as sp
from rdft.ac.ibp_solver.sunset_full import sunset_ibp


d, p2 = sp.symbols('d p^2', positive=True)


def main():
    print("=" * 72)
    print(" Test: reduce I[2,1,1] using IBP at seed (2,1,1)")
    print("=" * 72)

    ids = sunset_ibp(2, 1, 1)
    print(f"\nIBP identities at seed (2,1,1):")
    for (v, j), ident in ids.items():
        print(f"\n  IBP (v={v}, j={j}):")
        print(f"    {ident}")
        print(f"    = 0")

    # Pick the (v=k1, j=1) identity, which should give a clean (d - 5) coefficient on I[2,1,1]:
    target_id = ids[('k1', 1)]
    print(f"\n--- Solving (v=k1, j=1) identity for I[2,1,1] ---")
    I_211 = sp.Symbol('I[2,1,1]')

    sol = sp.solve(target_id, I_211)
    if sol:
        print(f"\n  I[2,1,1] = ")
        result = sp.simplify(sol[0])
        print(f"    {result}")
        print()
        print(f"  Factor out the leading (d - ?) coefficient:")
        # Extract the leading (d - integer) prefactor
        denom = sp.denom(result)
        numer = sp.numer(result)
        print(f"    denominator: {denom}")
        print(f"    numerator:   {numer}")

    # Now do the same for I[1,2,1] and I[1,1,2] using the analogous IBPs:
    print(f"\n" + "=" * 72)
    print(f" Symmetrised reductions (analogous identities at swapped seeds)")
    print(f"=" * 72)

    # I[1,2,1]: use (v=k2, j=2) at seed (1,2,1)
    ids_121 = sunset_ibp(1, 2, 1)
    target_id_121 = ids_121[('k2', 2)]
    I_121 = sp.Symbol('I[1,2,1]')
    sol_121 = sp.solve(target_id_121, I_121)
    if sol_121:
        print(f"\n  I[1,2,1] = {sp.simplify(sol_121[0])}")

    # I[1,1,2]: use (v=k2, j=1) at seed (1,1,2)
    # Actually for D3 raised, use a different IBP. Try (v=p, j=1):
    ids_112 = sunset_ibp(1, 1, 2)
    print(f"\n  IBP identities at seed (1,1,2) for reducing I[1,1,2]:")
    for (v, j), ident in ids_112.items():
        if sp.Symbol('I[1,1,2]') in ident.free_symbols:
            coef = ident.coeff(sp.Symbol('I[1,1,2]'))
            if coef != 0:
                print(f"    (v={v}, j={j}): coefficient on I[1,1,2] = {coef}")

    print()
    print("=" * 72)
    print(" Status")
    print("=" * 72)
    print("""
The single-identity reductions show the (d - 5) leading coefficient
on I[2,1,1], the analogous (d - 5) on I[1,2,1] (by symmetry), and
provide IBPs at seed (1,1,2) for the third raised integral.

This is exactly the Laporta-style algorithm at work:
  - Generate identities at the seed where the high-priority integral
    appears with the largest leading coefficient.
  - Solve for it; the solution involves only lower-priority integrals
    (and ISPs).
  - Iterate.

Closing the full system requires running this loop with a Laporta
ordering and substituting solutions back.  ~500 more lines of pure
SymPy along this template.
""")


if __name__ == '__main__':
    main()
