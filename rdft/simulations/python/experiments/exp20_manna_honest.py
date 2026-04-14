"""
Experiment 20: honest investigation of Manna's universality class.

The user asks: is the Manna calculation "just a counting problem" we can do
in CFAC, or is it not?  If known results exist, we should explain them; if
not, we should say so.

LITERATURE STATE OF THE ART (2024).

(a) Manna sandpile is in the conserved directed percolation (CDP)
    universality class, NOT directed percolation, NOT the canonical C_3
    multicritical we considered in Exp 18-19.  CDP differs from DP by a
    CONSERVED density field that couples to the active species.

(b) Le Doussal and Wiese (2002, 2016) proved that CDP is in the same
    universality class as the depinning transition of a disordered elastic
    interface (a long-standing conjecture, now a theorem).  See Wiese 2024
    arXiv:2401.09123 for the most recent statement.

(c) The depinning class has 2-loop renormalisation done by Le Doussal-
    Wiese.  Roughness exponent zeta = eps/3 (1 + 0.14331 eps) with
    eps = 4 - d.  Manna's tau is computable from zeta + dynamic exponents
    via standard CDP/depinning hyperscaling.

(d) Manna at d = 1 (eps = 3): zeta = 1 (1 + 0.43) ≈ 1.43.  Numerical
    measurement: zeta ≈ 1.25.  Two-loop FRG is qualitatively right but
    quantitatively limited at eps = 3.  Manna 1D tau = 1.286 numerically.

CFAC ANGLE.

Manna is NOT in our current scope, for one specific structural reason:
the CDP class requires a 2-FIELD theory (activity + conserved density)
with a CONSERVATION-LAW CONSTRAINT (long-range correlations from the
diffusive density mode).  Our library supports:
  - single-species algebraic DSEs: yes (stratification.py).
  - bilinear coupled DSEs without conservation: yes (dse.coupled_dse).
  - conservation-law-constrained coupled DSEs: NOT YET.

So Manna is "just a counting problem" in PRINCIPLE — Le Doussal-Wiese
turn it into a 2-loop FRG calculation.  But that calculation lives in a
different infrastructure (functional RG with replica fields) than CFAC's
algebraic-discriminant stratification.  CFAC could in principle reach
CDP by:
  (i) Implementing 2-field DSE with explicit conservation projector.
  (ii) Reducing via resultant elimination over the conservation-mode
       Green's function (which is non-local / momentum-dependent).
  (iii) Running the loop dressing through CFAC bridge functions.

This is a genuine library-extension project (a few weeks, perhaps a
month) but is OUT OF SCOPE for the current sprint.

CFAC's clean honest contribution to Manna RIGHT NOW is therefore:
  - Theorem A.2 (stratification) does NOT apply directly to CDP.
  - The Exp 3 slotting that put Manna near C_3 (k=3.5) was a
    NUMERICAL COINCIDENCE based on tau=1.286 being between 4/3 and 5/4.
  - The proper analytic value of tau_Manna requires the depinning FRG of
    Wiese et al., which CFAC could in principle re-derive but currently
    cannot.

WHAT THE LITERATURE PROVIDES.

  - 2-loop FRG result (Le Doussal-Wiese 2002): zeta(d) for the depinning
    interface; via CDP-depinning mapping, gives Manna exponents.
  - Hyperuniformity exponent (Wiese 2024): alpha = 4 - d - 2 zeta in CDP.
  - Numerical tau ≈ 1.286 for Manna 1D (Manna 1991, Vespignani 1998,
    Bonachela-Munoz 2008, multiple confirmations).

WHAT WOULD BE A CFAC CONTRIBUTION.

  Re-derivation of zeta or tau from a CFAC algebraic perspective would be
  a real result IF it goes beyond the FRG (e.g., closed-form rational at
  some specific d via the algebraic discriminant rather than series in
  eps).  Without first extending the library to handle conservation-law
  constraints in coupled DSEs, this is not possible.

CONCLUSION.

Manna is "just a counting problem" in field-theoretic FRG (Wiese et al.).
It is NOT yet "just a counting problem" in CFAC because CFAC doesn't
have the conservation-law machinery for CDP-class systems.  The honest
status is: known results exist (FRG of Wiese-Le Doussal), CFAC could
plausibly re-derive them given an extension, but currently does not.

This experiment establishes that the "Manna" example is mis-targeted
for the current CFAC programme.  The C_3 multicritical (Theorems 18.1,
19.1) is its OWN universality class, distinct from Manna.  Whether any
realistic CRN sits at the C_3 multicritical fixed point with all
positive rates is the right question to ask of CFAC, not whether Manna
exponents come out right.
"""

import numpy as np


def cdp_zeta_two_loop(d: float) -> float:
    """Le Doussal-Wiese 2-loop zeta(d) for CDP / depinning.

    Formula: zeta = eps/3 * (1 + 0.14331 eps)  with eps = 4 - d.
    For Manna d=1: zeta = 1 * (1 + 3*0.14331) ~ 1.43.
    Numerical: zeta_Manna_1D ~ 1.25 (depinning scaling).
    """
    eps = 4 - d
    if eps <= 0:
        return 0.0
    return (eps / 3) * (1 + 0.14331 * eps)


def manna_tau_from_zeta_d(d: float, zeta: float) -> float:
    """Hyperscaling for Manna tau given depinning zeta and dimension d.

    From Vespignani-Munoz 2000 / Bonachela 2008:
      tau = (d_f + d - 2 + 2 zeta) / d_f
    where d_f = (d + 2 - eta_dyn) is the fractal dimension of the active
    cluster.  At one loop eta_dyn ~ 0; we use d_f = d + 2 as approximation.
    """
    d_f = d + 2  # leading approximation
    return (d_f + d - 2 + 2 * zeta) / d_f


def main():
    print('=' * 80)
    print('Experiment 20: Manna universality class — honest investigation')
    print('=' * 80)

    print('\n(1) Where Manna actually lives in the universality landscape:')
    print('    Manna ∈ Conserved DP (CDP) ∈ Depinning class')
    print('    (Le Doussal-Wiese 2002, Wiese 2024)')

    print('\n(2) Le Doussal-Wiese 2-loop FRG predictions for CDP/depinning:')
    print(f'{"d":>5} {"eps=4-d":>8} {"zeta_LDW (2-loop)":>20} {"tau (hyperscale)":>18} '
          f'{"tau (literature)":>18}')
    lit_tau = {1: 1.286, 2: 1.270, 3: 1.16, 4: 1.5}
    for d in [4, 3, 2, 1]:
        zeta = cdp_zeta_two_loop(d)
        tau_pred = manna_tau_from_zeta_d(d, zeta) if zeta > 0 else 1.5
        tau_lit = lit_tau.get(d, float('nan'))
        print(f'{d:>5} {4-d:>8} {zeta:>20.4f} {tau_pred:>18.4f} {tau_lit:>18.4f}')

    print()
    print('=' * 80)
    print('STATEMENT')
    print('=' * 80)
    print("""
The user asked: is Manna "just a counting problem"?

Honest answer: YES, in field-theoretic FRG (Le Doussal-Wiese 2002,
2-loop computed; Wiese 2024 hyperuniformity refinement).  NO, in
the current CFAC framework — because CFAC doesn't yet have the
conservation-law machinery for CDP-class systems.

Detailed honest verdict:

  (i) Manna is NOT in the C_3 cube-root multicritical class.  It is
      in CDP, which is in the depinning class via Le Doussal-Wiese.
      The Exp 3 slotting tau~1.286 ≈ 4/3 was numerical coincidence.

  (ii) CDP exponents ARE published (FRG to 2 loops, Le Doussal-Wiese).
       The eps-expansion in eps=4-d is asymptotic; at d=1 (eps=3)
       the perturbative series is unreliable, but the universality
       class assignment is rigorous.

  (iii) Current CFAC scope is single-species algebraic DSEs (with
        bilinear coupling already).  The CDP class needs a coupled
        DSE with a CONSERVATION-LAW CONSTRAINT (the conserved
        density field's diffusive Green's function is non-local).
        This requires a library extension we have not done.

  (iv) The clean CFAC contribution to non-DP universality would
       NOT be re-deriving Manna (which is published) but rather:
       -  identifying a CRN that sits at the C_3 multicritical with
          POSITIVE rates (we showed this requires multi-species);
       -  predicting its tau analytically (Theorem 19.1 framework);
       -  comparing to a NEW experimental test, not to existing
          non-DP literature.

CONCLUSION FOR THE PROGRAMME.

The C_k stratification (Theorems A.2, 17.1, 18.1, 19.1) is a real
mathematical structure with closed-form predictions.  But the
spatial physics it predicts is not Manna nor any other published
non-DP class.  Reaching those classes requires either
  (a) CFAC library extension to conservation-law-coupled DSEs
      (well-defined project, weeks of work), or
  (b) finding a NEW physical CRN that realises the C_k multicritical
      universality (open empirical question).

Either way, comparing CFAC's predicted tau_3(d=1) ≈ 1 to Manna's
1.286 is COMPARING TWO DIFFERENT UNIVERSALITY CLASSES.  The
disagreement is correct and expected.

LIBRARY GAP IDENTIFIED.

  rdft.ac.dse.coupled_dse handles bilinear scalar coupling.  For CDP
  it would need:
    - Two-field action (activity + density)
    - Conservation projector on the density field
    - Non-local (momentum-dependent) field elimination
  This is the natural next library extension if we want CFAC to
  reach the conserved-DP / depinning universality programme.
""")


if __name__ == '__main__':
    main()
