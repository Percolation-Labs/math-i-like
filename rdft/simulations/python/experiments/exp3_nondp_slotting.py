"""
Experiment 3: non-DP universality classes vs. CFAC stratification ladder.

For each known non-DP universality class with measured size-distribution
exponent tau, solve tau = 1 + 1/k for k, the implied Puiseux order.  Report
distance to nearest integer stratum.

Literature values (size-distribution / avalanche-size exponent tau):

  Class                                        tau (d=1)    tau (d=2)    refs
  -----                                        ---------    ---------    ----
  Directed percolation (DP)                    1.50         1.50*        Odor 2004
  Manna / conserved directed percolation       1.28-1.29    1.27         Manna 1991, Vespignani 1998
  Voter model (dim independent)                -            -
  Parity-conserving BARW (PCPD/BAW)            1.17(5)      1.04         Cardy-Tauber 1996
  Conserved Manna (C-Manna)                    1.29         1.30         Bonachela-Munoz 2008
  Reversible parity                            1.28         -
  Lee-Yang (Parisi-Sourlas)                    5/2          5/2          Parisi-Sourlas 1981
  Absorbing-state with infinitely many         1.15         -            Munoz et al.
    absorbing states

*DP d=2 tau is close to mean-field 3/2.

Targets of the CFAC stratification ladder tau_k = 1 + 1/k:
  k =  2 :  tau = 3/2   = 1.5000  (DP, square-root, allowed)
  k =  3 :  tau = 4/3   ≈ 1.3333  (cube-root, FORBIDDEN to N-algebraic)
  k =  4 :  tau = 5/4   = 1.2500  (quartic, dyadic-allowed)
  k =  5 :  tau = 6/5   = 1.2000  (forbidden)
  k =  6 :  tau = 7/6   ≈ 1.1667  (forbidden)
  k =  7 :  tau = 8/7   ≈ 1.1429  (forbidden)
  k =  8 :  tau = 9/8   = 1.1250  (dyadic-allowed)
  k = 10 :  tau = 11/10 = 1.1000
"""

import numpy as np


# Experimental / literature values
classes = [
    # (name, tau values dict by dimension, references)
    ('DP (directed percolation)', {1: 1.5000, 2: 1.5000}, 'Ódor 2004 (3/2 for Gribov/Reggeon)'),
    ('Manna', {1: 1.286, 2: 1.27}, 'Manna 1991; Vespignani et al. 1998'),
    ('C-Manna (conserved Manna)', {1: 1.29, 2: 1.30}, 'Bonachela-Muñoz 2008'),
    ('Parity-conserving BARW', {1: 1.17, 2: 1.04}, 'Cardy-Täuber 1996'),
    ('Pair contact process (PCPD)', {1: 1.20, 2: 1.30}, 'Carlon et al. 2001'),
    ('Infinitely many absorbing states', {1: 1.15, 2: None}, 'Muñoz et al. 1998'),
    ('Lee-Yang (Parisi-Sourlas)', {1: 2.500, 2: 2.500}, 'Parisi-Sourlas 1981'),
]


def implied_k(tau: float) -> float:
    if tau <= 1:
        return np.inf
    return 1 / (tau - 1)


def closest_stratum(k: float) -> tuple[int, float, str]:
    """Return (nearest integer k, distance, BD status of exponent)."""
    if not np.isfinite(k):
        return 0, np.inf, ''
    k_int = int(round(k))
    dist = abs(k - k_int)
    # BD status: k must be power of 2 for tau = 1+1/k to be allowed
    allowed = (k_int > 0) and ((k_int & (k_int - 1)) == 0)
    return k_int, dist, 'allowed (dyadic)' if allowed else 'FORBIDDEN (non-dyadic)'


# Print table
print('=' * 100)
print(f'{"Class":<45} {"d":<3} {"τ (lit)":<10} {"k = 1/(τ-1)":<15} {"nearest k":<12} {"distance":<10} {"BD status"}')
print('=' * 100)

rows = []
for name, taus, ref in classes:
    for d, tau in taus.items():
        if tau is None:
            continue
        k_imp = implied_k(tau)
        k_near, dist, status = closest_stratum(k_imp)
        rows.append((name, d, tau, k_imp, k_near, dist, status, ref))
        print(f'{name:<45} {d:<3} {tau:<10.4f} {k_imp:<15.4f} {k_near:<12} {dist:<10.4f} {status}')

# ----------------------------------------------------------
#  Interpretation
# ----------------------------------------------------------
print()
print('=' * 100)
print('Interpretation:')
print('=' * 100)
print('''
Goal: test whether known non-DP universality classes "slot into" integer
Puiseux strata of the CFAC stratification.  A clean fit would mean
tau_class = 1 + 1/k for some integer k, up to spatial/finite-size corrections.

The headline numbers:
  • Manna (d=1): tau=1.286 → k=3.50.  Between C_3 (1.33) and C_4 (1.25).  Loose fit.
  • DP: tau=3/2 exact → k=2 exactly.  Trivial, by construction.
  • Parity-conserving BARW: tau=1.17 (d=1) → k=5.88.  Near C_6 (forbidden) or C_8 (allowed).
  • Lee-Yang: tau=5/2 → k=2/3.  NOT in our ladder at all — different universality.

What this tells us:
  The integer stratification ladder is NOT a direct classification scheme
  for spatial non-DP universality.  The reason is that our ladder lives in
  0-dimensional (combinatorial) DSE space.  Spatial corrections (via the
  epsilon-expansion or bridge-function anomalous dimensions) shift the
  exponents away from integer k.

  But: the observation is still useful as a CHECK.  If a class's exponent
  sits close to an integer k (within ~0.2), CFAC predicts that the spatial
  corrections away from mean-field are small for that class.  Conversely,
  large deviations from integer k signal strong spatial dressing.

  For Manna specifically: implied k ≈ 3.50, so Manna is halfway between
  cube-root and quartic-root.  The d=1 Manna is *not* simply on C_3.
  However the CONSERVED-Manna (which has an extra conservation law modifying
  the DSE) sits closer to 1.29–1.30, giving implied k ≈ 3.33, slightly
  closer to C_3 = 4/3.  The trend suggests the "signed current conservation"
  is part of what produces the near-C_3 stratum behaviour.

A cleaner test would be to extract the 0-dimensional / mean-field exponent
for each class (above the upper critical dimension) and check that THAT
exponent matches an integer k exactly.  That's the second-order experiment.
''')

# Save table
import json
from pathlib import Path
outdir = Path(__file__).parent.parent.parent.parent / 'paper' / 'cfac' / 'figures'
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / 'exp3_nondp_slotting.json', 'w') as f:
    json.dump([{
        'class': r[0], 'd': r[1], 'tau': r[2], 'k_implied': r[3],
        'k_nearest': r[4], 'distance': r[5], 'BD_status': r[6], 'ref': r[7]
    } for r in rows], f, indent=2)

print(f'Saved table to {outdir / "exp3_nondp_slotting.json"}')
