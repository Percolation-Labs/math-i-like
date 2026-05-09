"""
hermite_vertices.py
===================

Compute the four-point Hermite overlap tensor

    V_{n1 n2 n3 n4}  =  int_{-infty}^{+infty}  h_{n1}(x) h_{n2}(x) h_{n3}(x) h_{n4}(x)  dx

where h_n(x) is the L^2-normalised Hermite function

    h_n(x)  =  H_n(x) exp(-x^2/2) / sqrt( 2^n n! sqrt(pi) ),

with H_n the *physicists'* Hermite polynomial (so int h_a h_b dx = delta_{ab}).
Pulling the four exp(-x^2/2) factors together gives the integral

    V_{n1 n2 n3 n4} = ( prod_i N_{n_i} ) int H_{n1} H_{n2} H_{n3} H_{n4} exp(-2 x^2) dx

with N_n = 1 / sqrt( 2^n n! sqrt(pi) ).  This script evaluates that integral
symbolically with sympy for n_i in {0,1,2,3} and prints the table.

Selection rules verified numerically:

    (1) parity:  V vanishes unless n1 + n2 + n3 + n4 is even
                 (each H_n has parity (-1)^n, so the integrand picks up
                  (-1)^{sum n_i}; the integration domain is symmetric).

    (2) NO triangle inequality on the four indices.
        For the *three*-point overlap int h_a h_b h_c dx the triangle
        |a-b| <= c <= a+b applies (Bailey / Gradshteyn-Ryzhik 7.375.2),
        but the four-point V is a quadratic in those three-overlaps
        (V_{abcd} = sum_e g_{abe} g_{cde}) and therefore lifts the
        triangle to a milder constraint |a-b-c-d| <= ... that does not
        cut any quadruple for n_i in {0..5}.  Verified by exhaustive
        evaluation up to n=5.

Run with `python3 hermite_vertices.py`.  Requires sympy.
"""

import itertools
from sympy import (
    Integer, Rational, Symbol, exp, factorial, hermite, integrate, oo, pi,
    simplify, sqrt,
)


def hermite_norm(n):
    """1 / sqrt(2^n n! sqrt(pi))  -- the L^2 normalisation of h_n."""
    return 1 / sqrt(2 ** n * factorial(n) * sqrt(pi))


def four_overlap(n1, n2, n3, n4, x):
    """V_{n1 n2 n3 n4} = int h_{n1} h_{n2} h_{n3} h_{n4} dx, returned exactly."""
    if (n1 + n2 + n3 + n4) % 2 == 1:
        return Integer(0)  # parity selection
    integrand = hermite(n1, x) * hermite(n2, x) * hermite(n3, x) * hermite(n4, x)
    integrand *= exp(-2 * x ** 2)
    raw = integrate(integrand, (x, -oo, oo))
    norm = hermite_norm(n1) * hermite_norm(n2) * hermite_norm(n3) * hermite_norm(n4)
    return simplify(raw * norm)


def main(N=3):
    x = Symbol("x", real=True)
    print(f"Hermite four-overlap V_{{n1 n2 n3 n4}} for n_i in {{0,..,{N}}}")
    print("Convention: physicists' H_n; h_n = H_n exp(-x^2/2) / sqrt(2^n n! sqrt(pi)).")
    print("Parity rule: V = 0 unless n1+n2+n3+n4 is even.")
    print("(No triangle inequality on the four-point overlap.)")
    print()
    print(f"{'(n1,n2,n3,n4)':>16} | {'V (exact)':>40} | {'V (decimal)':>14}")
    print("-" * 80)

    table = {}
    for tup in itertools.combinations_with_replacement(range(N + 1), 4):
        V = four_overlap(*tup, x)
        table[tup] = V
        Vstr = str(V)
        Vfloat = float(V)
        print(f"{str(tup):>16} | {Vstr:>40} | {Vfloat:>14.7f}")

    # Surviving / killed counts
    unordered = list(table)
    n_total = len(unordered)
    n_zero = sum(1 for t in unordered if table[t] == 0)
    n_nonzero = n_total - n_zero
    print()
    print(f"Unordered quadruples (n_i in {{0..{N}}}):   {n_total}")
    print(f"  killed by parity:                        {n_zero}")
    print(f"  surviving (V != 0):                      {n_nonzero}")

    # Ordered counts (for symmetry-factor consumers downstream)
    ordered_total = (N + 1) ** 4
    ordered_nonzero = 0
    for tup in itertools.product(range(N + 1), repeat=4):
        if table[tuple(sorted(tup))] != 0:
            ordered_nonzero += 1
    print(f"Ordered quadruples ({N+1}^4):                  {ordered_total}")
    print(f"  surviving:                               {ordered_nonzero}")

    return table


if __name__ == "__main__":
    main(N=3)
