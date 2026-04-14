"""
Experiment 5: quartic-root family search.

Find a degree-4 CRN family that crosses C_4 the way Schlögl II crosses C_3.
The CFAC stratification predicts that for any polynomial DSE phi of degree d,
the locus C_k for 2 <= k <= d is reachable.  Here we test for k=4 in a
CRN that includes 3A->4A as the highest-degree reaction.

CRN candidate (degree-4 extension of Schlögl II):
  A -> 2A      rate p   (creation)
  2A -> A      rate q   (pair annihilation)
  3A -> 4A     rate r   (triple-to-quadruple autocatalysis)
  4A -> 3A     rate s   (quadruple-to-triple back-reaction)

Doi-Peliti vertex contributions to phi(G) = 1 + Sum g_{mn} G^{m+n-2}:
  A -> 2A   (vertex (2,1)):  +p G
  2A -> A   (vertex (1,2)):  -q G;   (vertex (2,2)):  -q G^2
  3A -> 4A  (vertex (4,3)):  +r G^5;  (vertex (3,3)):  +3r G^4;  (vertex (2,3)):  +3r G^3;  (vertex (1,3)):  +r G^2
  4A -> 3A  (vertex (3,4)):  -s G^5;  (vertex (2,4)):  -4s G^4;  (vertex (1,4)):  -6s G^3
                              (4,4) -s G^6, (5,4) ... (vertex with > 4 legs not allowed since outgoing<=4)
                              Actually outgoing from 4A->3A is 3, so vertices are (3,4) -s, (2,4) -4s, (1,4) -6s, (0,4) -4s
                              The (0,4) vertex has m+n=4 >= 3 so it's an interaction; contributes -4s G^2
  Also 3A -> 4A above includes (0,3) +r G^1.

Hmm this gets complicated.  Let me simplify using the Doi-Peliti generator directly.

For a single-species reaction k A -> l A at rate lambda, the generator is
  Q = lambda ((z+1)^l - (z+1)^k) partial^k = lambda ((z+1)^l - (z+1)^k) D^k.
The interaction part contributes to phi(G) via the vertex expansion:
  phi(G) += lambda * (sum of binomial terms in (z+1)^l - (z+1)^k giving (m,k) vertices for m >= 1)
After the shift z -> phi-tilde and identification G -> phi*phi-tilde, each vertex
contributes G^{m+k-2} where m is the outgoing power.

For 3A -> 4A: l=4, k=3.
  (z+1)^4 - (z+1)^3 = z^4 + 4z^3 + 6z^2 + 4z + 1 - (z^3 + 3z^2 + 3z + 1)
                    = z^4 + 3z^3 + 3z^2 + z
  Vertices: (4,3): r => r G^{4+3-2} = r G^5
            (3,3): 3r => 3r G^4
            (2,3): 3r => 3r G^3
            (1,3): r => r G^2

For 4A -> 3A: l=3, k=4.
  (z+1)^3 - (z+1)^4 = -(z^4 + 3z^3 + 3z^2 + z)
                    = -z^4 - 3z^3 - 3z^2 - z
  ... and also non-trivial expansion:
  (z+1)^3 - (z+1)^4 = -(z+1)^3 z = -z^4 - 3z^3 - 3z^2 - z
  Vertices: (4,4): -s => -s G^6
            (3,4): -3s => -3s G^5
            (2,4): -3s => -3s G^4
            (1,4): -s => -s G^3

So including 3A->4A and 4A->3A:
  phi += r G^2 + 3r G^3 + 3r G^4 + r G^5
       - s G^3 - 3s G^4 - 3s G^5 - s G^6
       + (vertices from A->2A and 2A->A as before)

Limiting ourselves to degree <= 4 in G (i.e., dropping G^5, G^6 contributions):
NO — that would lose the quartic tuning.  Instead let's look at degree <= 5.

phi(G) = 1 + (p - q) G + (r - q) G^2 + (3r - s) G^3 + (3r - 3s) G^4 + (r - 3s) G^5 - s G^6

To use the canonical degree-4 stratification, we want phi degree 4.  Drop the
4A->3A entirely (set s=0), keep only forward reactions plus pair annihilation:

phi(G) = 1 + (p - q) G + (r - q) G^2 + 3r G^3 + 3r G^4 + r G^5

Still degree 5.  To match canonical (1+G)^4 + beta G structure, we'd need
specific rate ratios.  Let's directly target:
  phi = (1+G)^4 + beta G = 1 + (4+beta) G + 6 G^2 + 4 G^3 + G^4

Comparing:
  G^0: 1  ✓
  G^1: p - q = 4 + beta
  G^2: r - q = 6
  G^3: 3r = 4 => r = 4/3
  G^4: 3r = 1 => r = 1/3
  CONTRADICTION.

So the canonical (1+G)^4 + beta G is NOT realisable by the CRN
{A->2A, 2A->A, 3A->4A} alone.  We need ADDITIONAL reactions to fine-tune
the G^3 and G^4 coefficients independently.

Let's add 2A -> 3A (rate u) and 3A -> 2A (rate v):
  2A -> 3A: vertices (3,2) +u G^3, (2,2) +2u G^2, (1,2) +u G
  3A -> 2A: vertices (2,3) -v G^3, (1,3) -v G^2 ... wait let me redo
    (z+1)^2 - (z+1)^3 = -(z^3 + 2z^2 + z) - 0 = -z^3 - 2z^2 - z
    Vertices (3,3): -v G^4, (2,3): -2v G^3, (1,3): -v G^2

Including all of {A->2A, 2A->A, 2A->3A, 3A->2A, 3A->4A, 4A->3A}:

  phi(G) = 1
         + (p - q + u - v) G
         + (-q + 2u - 2v + r) G^2 ... need to redo carefully
"""

import numpy as np
import sympy as sp
from math import comb


def vertex_contributions(reactions: list[tuple[int, int, float]]) -> dict[int, float]:
    """For a list of (k, l, rate) reactions kA -> lA, return dict {power: coefficient}
    of phi(G) = 1 + Sum_m coef[m] G^m."""
    phi_coefs = {}
    for k, l, rate in reactions:
        # (z+1)^l - (z+1)^k expanded in z; coefficient of z^j (j>=1) gives vertex
        # with j outgoing phi-tilde and k incoming phi.  Power in G is j + k - 2.
        for j in range(1, max(l, k) + 1):
            coef_l = comb(l, j) if j <= l else 0
            coef_k = comb(k, j) if j <= k else 0
            coef = rate * (coef_l - coef_k)
            if coef != 0:
                power = j + k - 2  # power in G
                phi_coefs[power] = phi_coefs.get(power, 0) + coef
    return phi_coefs


def phi_polynomial(reactions, max_degree=10):
    """Return phi(G) as a sympy polynomial."""
    G = sp.Symbol('G')
    coefs = vertex_contributions(reactions)
    expr = sp.Integer(1)
    for power, coef in sorted(coefs.items()):
        if power <= max_degree:
            expr += sp.Rational(int(coef * 1000000), 1000000) * G ** power if isinstance(coef, float) else coef * G ** power
    return sp.expand(expr)


def find_dominant_branch_order(phi_expr) -> tuple[int, complex]:
    """Compute the dominant branch order of F = G - z phi as a polynomial in G."""
    G, z = sp.symbols('G z')
    F = G - z * phi_expr
    try:
        disc = sp.discriminant(sp.Poly(F, G).as_expr(), G)
        disc_poly = sp.Poly(disc, z)
        coeffs = [float(c) for c in disc_poly.all_coeffs()]
        roots = np.roots(coeffs)
        nontrivial = [r for r in roots if abs(r) > 1e-9]
        if not nontrivial:
            return -1, np.nan + 0j
        closest = min(nontrivial, key=lambda r: abs(r))
        # multiplicity at closest
        mult = sum(1 for r in nontrivial if abs(r - closest) < 5e-3)
        return 1 + mult, closest
    except Exception as e:
        return -1, np.nan + 0j


def main():
    # Try to engineer a CRN that realises phi = (1+G)^4 + beta G for various beta.
    # This requires solving for rates given the target phi coefficients.

    print('=' * 80)
    print('Target: degree-4 canonical family phi(G) = (1+G)^4 + beta G')
    print('       = 1 + (4+beta) G + 6 G^2 + 4 G^3 + G^4')
    print('=' * 80)

    # Candidate CRN with reactions {A->2A:p, 2A->A:q, 2A->3A:u, 3A->2A:v, 3A->4A:r}
    # The vertex map gives phi coefficients in terms of (p, q, u, v, r).
    # We want to match degree-4 canonical phi = 1 + (4+beta)G + 6G^2 + 4G^3 + G^4.
    # System of equations: solve for rates to match coefficients up to degree 4.
    #
    # From vertex_contributions, the coefficient of G^m is a linear combination
    # of the rates.  Need to solve for non-negative rates.

    p, q, u, v, r = sp.symbols('p q u v r', nonnegative=True)
    G = sp.Symbol('G')
    reactions_sym = [(1, 2, p), (2, 1, q), (2, 3, u), (3, 2, v), (3, 4, r)]
    coefs_sym = {}
    for k, l, rate in reactions_sym:
        for j in range(1, max(l, k) + 1):
            cl = comb(l, j) if j <= l else 0
            ck = comb(k, j) if j <= k else 0
            c = rate * (cl - ck)
            if c != 0:
                power = j + k - 2
                coefs_sym[power] = coefs_sym.get(power, 0) + c

    print(f'Symbolic phi coefficients from rates (p, q, u, v, r):')
    for power, coef in sorted(coefs_sym.items()):
        print(f'  G^{power}: {sp.simplify(coef)}')

    # Match to (1+G)^4 + beta G
    beta = sp.Symbol('beta', real=True)
    target = {0: 1, 1: 4 + beta, 2: 6, 3: 4, 4: 1}
    # Write match equations (only for coefficients we have; degree > 4 must be zero
    # but our reactions only produce up to G^4, so check)

    eqs = []
    for power in sorted(target.keys()):
        if power == 0:
            continue
        lhs = coefs_sym.get(power, 0)
        rhs = target[power]
        eqs.append(sp.Eq(lhs, rhs))

    # Also require degree > 4 contributions to be zero
    for power in sorted(coefs_sym.keys()):
        if power > 4:
            eqs.append(sp.Eq(coefs_sym[power], 0))

    print(f'\n{len(eqs)} equations to solve for {{p, q, u, v, r}} given beta:')
    for eq in eqs:
        print(f'  {eq}')

    sol = sp.solve(eqs, [p, q, u, v, r], dict=True)
    print(f'\nSolutions (parameterised by beta):')
    for s in sol:
        print(f'  {s}')

    if not sol:
        print('No solution: this CRN family cannot realise the canonical phi_4 family.')
        print('Need a richer set of reactions or accept that engineering a degree-4')
        print('CRN on the canonical curve requires more vertex types.')

    # Alternative: take a specific CRN and find which stratum it lands on
    print()
    print('=' * 80)
    print('Alternative: pick rates and read off stratum.')
    print('=' * 80)
    candidates = [
        # (description, reactions list as (k, l, rate))
        ('A->2A:1, 2A->A:1, 3A->4A:1',
         [(1, 2, 1), (2, 1, 1), (3, 4, 1)]),
        ('A->2A:1, 2A->3A:1, 3A->4A:1',
         [(1, 2, 1), (2, 3, 1), (3, 4, 1)]),
        ('A->2A:2, 2A->A:1, 3A->4A:1, 4A->3A:1',
         [(1, 2, 2), (2, 1, 1), (3, 4, 1), (4, 3, 1)]),
        ('A->4A:1, 2A->A:2, 3A->2A:1',
         [(1, 4, 1), (2, 1, 2), (3, 2, 1)]),
    ]
    for desc, rxns in candidates:
        coefs = vertex_contributions(rxns)
        phi = sp.Integer(1)
        for power, c in sorted(coefs.items()):
            phi += sp.Rational(int(c * 1000000), 1000000) * sp.Symbol('G') ** power
        phi = sp.expand(phi)
        deg = sp.Poly(phi, sp.Symbol('G')).degree()
        k_dom, z_star = find_dominant_branch_order(phi)
        print(f'\n{desc}')
        print(f'  phi(G) = {phi}  (degree {deg})')
        print(f'  dominant Puiseux order k = {k_dom},  |z*| = {abs(z_star):.4f}')


if __name__ == '__main__':
    main()
