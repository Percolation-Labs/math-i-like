"""
rdft.ac.topology_sketch
========================
Speculative scoping note: what would CFAC look like if applied to
moduli-space counting (Gromov-Witten, Donaldson-Thomas)?

Context
-------
Feynman-graph counting is the standard combinatorial substrate of
perturbative QFT, and CFAC handles it natively.  Topological field
theories (TFTs) have a DIFFERENT combinatorial substrate: instead
of counting graphs, one counts MAPS or SHEAVES on a target space.

Gromov-Witten theory (Witten 1990, Kontsevich 1995):
    GW invariants N_{g, d}(X) count pseudoholomorphic maps
    f: Sigma_g -> X
    from a genus-g Riemann surface into a target X with image
    class d in H_2(X; Z).  The partition function
    Z(u, t) = sum_{g, d} N_{g,d}(X) u^{2g-2} e^{t d}
    is the GW generating function.

Donaldson-Thomas theory (Donaldson-Thomas 1998; Maulik-Nekrasov-
Okounkov-Pandharipande 2006):
    DT invariants count ideal sheaves on a threefold X with
    prescribed Chern classes.  GW/DT conjectural equivalence:
    Z_{GW}(X) ~ Z_{DT}(X) after a change of variables.

These are genuinely counting problems — but of MAPS or SHEAVES, not
Feynman graphs.  The question for CFAC: does the stratification
theorem (tau_k = 1 + 1/k from the Puiseux order of a dominant
branch point) have a natural analogue when the generating function
is a GW / DT partition function?

What this module does
---------------------
Provides a concrete toy case — the genus-0 GW invariants of the
projective line P^1 — and asks:
1. What is the analytic structure of the GW generating function?
2. Does it have Puiseux-type singularities?
3. What (if anything) does CFAC's stratification say about it?

Observation (no proof)
----------------------
For many target spaces, the GW partition function is MODULAR
(quasi-modular forms) rather than algebraic.  The singularity
structure sits at the boundary of the modular domain — typically
ESSENTIAL singularities, not algebraic branches.  This is the
analogue of TODO-17 (non-isolated / essential singularities) in
the CFAC framework.

Conclusion: GW/DT are genuine counting problems but their
generating functions sit in a DIFFERENT analytic category from
CFAC's algebraic / D-finite setting.  A stratification theorem
for modular generating functions would be the required extension
— a research problem in its own right, sharing flavour with
Zagier's work on quasi-modular forms for counting problems.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


def gw_p1_generating_function(d_max: int = 10) -> Dict:
    """Genus-0 GW invariants of the projective line P^1.

    For P^1, the genus-0 GW invariants with 3 marked points (degree 1)
    are trivially 1.  The generating function is well-known:
        Z_{g=0}(t) = sum_d N_{0,d}(P^1) e^{t d}
    The degree-d invariant is
        N_{0, d}(P^1) = 1   (for all d >= 1 with 3+ insertions).
    So the generating function is
        Z(q) = sum_{d=1}^infty q^d = q / (1 - q)
    with q = e^t.  This has a SIMPLE POLE at q = 1 (rational, not algebraic).

    Puiseux order: a simple pole is "1/(1-q)" which is the k=1 case
    (would be tau = 1+1/k with k=1 giving tau = 2 — the boundary of
    the stratification ladder).  This sits OUTSIDE the Drmota-Lalley-
    Woods square-root universality (k >= 2) and inside the
    meromorphic class (TODO-16).
    """
    coeffs = [1] * d_max  # N_{0, d}(P^1) = 1
    # The generating function q / (1 - q) has [q^n] = 1 for n >= 1.
    return {
        'system': 'P^1 genus-0 GW',
        'coefficients': coeffs,
        'generating_function': 'q / (1 - q)',
        'singularity_at': 'q = 1 (simple pole)',
        'analytic_type': 'rational (meromorphic)',
        'puiseux_order_k': 1,
        'tau': 1 + 1/1,  # formally 2
        'note': ('simple pole lies at the boundary of CFAC stratification '
                  '(k=1 limit).  Meromorphic case falls under TODO-16.'),
    }


def dt_toy_partition_function_sketch() -> Dict:
    """Sketch: the simplest DT partition function.

    For a Calabi-Yau threefold X, the DT partition function is
        Z_DT(X; q) = sum_n DT_n(X) q^n
    where DT_n counts ideal sheaves with Chern character (1, 0, 0, -n).
    For X = local P^2 (or similar simple CY3), the DT generating function
    has a specific product formula (MacMahon function).

    The MacMahon function is
        M(q) = prod_{n >= 1} (1 - q^n)^{-n}
    which has NATURAL-BOUNDARY-like behaviour at |q| = 1 (every root of
    unity is a singular point).  This is a non-D-finite generating function.

    CFAC implication: the MacMahon-like DT functions do NOT admit the
    stratification theorem as stated; they sit in the TODO-2
    (non-D-finite) category of boundary cases.
    """
    return {
        'system': 'local CY3 DT (MacMahon sketch)',
        'generating_function': 'M(q) = prod_n (1 - q^n)^{-n}',
        'analytic_type': 'non-D-finite (natural boundary at |q|=1)',
        'puiseux_order_k': None,
        'note': ('MacMahon function is a non-D-finite GF with natural '
                  'boundary.  CFAC stratification does not apply.  '
                  'Extension would require modular / quasi-modular '
                  'analogue (Zagier-style).  TODO-2 or research-level.'),
    }


def compare_to_cfac_strata() -> Dict:
    """Summary: where GW/DT generating functions sit relative to CFAC
    stratification."""
    return {
        'categories': [
            {
                'name': 'P^1 genus-0 (rational GW)',
                'class': 'meromorphic (simple pole)',
                'in_CFAC': 'boundary case (k=1 limit); TODO-16',
            },
            {
                'name': 'local CY3 DT / MacMahon',
                'class': 'non-D-finite (natural boundary)',
                'in_CFAC': 'not in scope; TODO-2 or research',
            },
            {
                'name': 'Higher-genus topological recursion (Eynard-Orantin)',
                'class': 'algebraic generating functions on spectral curves',
                'in_CFAC': 'potentially in scope; requires multivariate '
                           'CFAC on the spectral curve',
            },
            {
                'name': 'Quantum cohomology of Fano varieties',
                'class': 'algebraic (solves WDVV equations, polynomial DSE-like)',
                'in_CFAC': 'in scope with Theorem I / II once the '
                           'WDVV system is written as a multivariate DSE',
            },
        ],
        'verdict': (
            'GW/DT is a broad territory.  Rational subcases (projective '
            'lines) fit CFAC at the k=1 boundary.  CY3 DT with MacMahon '
            'structure is outside scope.  Topological recursion and quantum '
            'cohomology of Fano varieties sit inside scope in principle, '
            'if the algebraic structure is written as a multivariate DSE. '
            'No concrete CFAC-native calculation has been done; this is '
            'genuinely speculative territory.'
        ),
    }


if __name__ == '__main__':
    print('=' * 70)
    print('Topology / moduli-space counting: CFAC scoping sketch')
    print('=' * 70)

    print('\nCase 1: Genus-0 GW of P^1')
    p1 = gw_p1_generating_function()
    for k, v in p1.items():
        if k != 'coefficients':
            print(f'  {k}: {v}')

    print('\nCase 2: CY3 DT sketch (MacMahon)')
    dt = dt_toy_partition_function_sketch()
    for k, v in dt.items():
        print(f'  {k}: {v}')

    print('\nSummary of where GW/DT sits in CFAC:')
    c = compare_to_cfac_strata()
    for row in c['categories']:
        print(f'  - {row["name"]}')
        print(f'    class: {row["class"]}')
        print(f'    in CFAC: {row["in_CFAC"]}')
    print()
    print('Verdict:')
    print(c['verdict'])
