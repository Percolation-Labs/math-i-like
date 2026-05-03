# `rdft.crn` — honest self-critique

This file captures the gaps in the current implementation, written at the
state of commit `652c0d2` (refactor that derives all rationals). Read it
before trusting the package to do something it doesn't claim.

## What we deliver

1. **Doi shift → vertex dictionary → φ(G).** Mechanical for any CRN.
2. **Lagrange counts and φ-tree symmetry factors.** Mechanical for $\phi(G) = 1+G^2$
   today (cubic single-species); the same scheme generalises to any
   polynomial $\phi$, but the enumerator currently only handles binary trees.
3. **1-loop algebra factor $a_X^{(1)}$.** Factored as
   $a_X^{(1)} = c_X(\text{CRN}) \cdot K_X(\text{scheme})$.
   The combinatorial factor $c_X$ is mechanical from the Doi vertices
   (Wick + cumulant + sign products). The kinematic kernel $K_X$ is
   *currently shipped as a pre-computed rational in
   `Schemes.jt05_reggeon_dp().kinematic_kernels`*. See gap (1) below.
4. **Hopf antipode → $Z^{(2,2)}_X$.** Universal Connes–Kreimer formula,
   genuinely mechanical.
5. **BPHZ counterterm → 2-loop simple-pole structure.** Mechanical.
6. **2-loop primitive residues.** *Currently shipped as 12 pre-computed
   rationals in `scheme.kinematic_kernels_2loop`*. See gap (1).
7. **Täuber relation → RG functions.** Mechanical.
8. **MSbar pole-cancellation → β(u).** Mechanical, given $Z_\text{combined}$
   structure (the `coupling_z_exponents` ansatz).
9. **Wilson–Fisher u\* → critical exponents.** Mechanical.

For Reggeon DP at JT05's symmetric subtraction point, the pipeline returns
$\eta, z, \nu, \beta_{DP}$ matching JT05 Eq. (60) with **zero residual**.

## Where the gaps are

### Where the line really sits — and what's mechanical

Reframed by user: the ω-integration on a 1-loop bubble IS a graph contraction
(close contour, collapse two propagators into one). Topologically combinatorial.
Same for the Feynman shift, the standard $\int d^dk/(k^2+M^2)^n$ closed form, and
the Z-extraction derivative at the sub-point. **All of these are mechanical
operations on (graph + propagator + sub-point + extraction operator).**

The honest line is therefore:

* **Combinatorial (ours, mechanical):** Doi shift; Lagrange counts; φ-tree |Aut|;
  Wick + cumulant + sign-product algebra; ω-contour graph contraction; Feynman
  shift; closed-form spatial loop integral; ε-expansion + 1/ε residue extraction;
  Z-extraction derivative at the sub-point; IBP reduction to master basis;
  Hopf antipode; BPHZ; Täuber; Wilson–Fisher.
* **Physical ansatz (gap (a), theirs):** propagator $G(ω, k^2)$; subtraction-point
  values; Z-factor extraction operators; coupling Z-exponents (definition of $u_R$).
* **Bridge integrals (gap (b), theirs):** numerical *values* of the master
  integrals $\{B_2, B_3^{\text{sun}}, B_V\}$ at the sub-point.

The kinematic kernels currently shipped in the scheme should NOT be there. They
are derivable from the propagator + sub-point + extraction operator + IBP
relations — all of which are combinatorial/topological, not physical.

### Gap (1) — kinematic kernels are not yet derived from the propagator

What needs to happen:

* **At 1-loop:** symbolically compute the d-dim bubble integral
  $I(\omega, q^2, \tau) = \int d^dk\, dω'/(2\pi)^{d+1}\, G(ω-ω', q-k)\, G(ω', k)$
  by ω-contour residues + Feynman shift + the standard
  $\int d^dk / (k^2+M^2) = \Gamma(1-d/2)/(4\pi)^{d/2}\, M^{d-2}$, then apply
  $\partial/\partial v$ at the user's sub-point and read off the 1/ε residue.
  Mechanical given the propagator. Implemented in `algebra.derive_one_loop_kernel`
  (see commit log).
* **At 2-loop:** the same idea, with each topology's loop-momentum routing
  recorded as graph data. The IBP reductions to the master basis are pure rational
  linear algebra on the propagator's algebraic structure — Smirnov 2012 / Panzer
  2015 document the procedure for Reggeon-style propagators. Larger project
  (~2–3 days). NOT YET DONE.

### Gap (2) — the enumerator is binary-only

`enumerate_phi_trees(n)` produces plane binary trees, hard-coded for
$\phi(G) = 1+G^2$. To generalise to $\phi^4$ Doi-Peliti ($\phi(G)=1+G^4$)
or BARW ($\phi$ of higher degree) the enumerator needs to handle multi-arity
branching. Lagrange's formula is general; the enumerator just needs to
match.

### Gap (3) — only one theory demonstrated end-to-end

We have `Schemes.jt05_reggeon_dp()` and reproduce its exponents. Until we
demonstrate a *second* theory (say, $\varphi^4$ Doi–Peliti at 1-loop
giving the standard $\nu = 1/2 + \varepsilon/12$), the abstraction is not
proven to be reusable.

### Gap (4) — propagator zoo is a single propagator

`Propagator.reggeon_dp()` is the only built-in. To support fresh CRNs
the user has to instantiate `Propagator(...)` themselves. That's fine in
principle, but the kernels they'd need to provide for their propagator
family aren't auto-derived (gap 1).

## What the USP honestly is

> **For any CRN with a cubic Doi-Peliti vertex pair and a Reggeon-style
> propagator, the pipeline runs from reactions to critical exponents
> mechanically. The combinatorial layer (Doi shift, Lagrange, φ-tree
> |Aut|, Hopf antipode, BPHZ, Täuber, Wilson–Fisher) is theory-agnostic.
> The kinematic layer for non-Reggeon propagators is currently the user's
> responsibility.**

That is a real contribution to the CFAC programme. It is *not*
"hand us a CRN, get exponents." We should not say that until gap (1) is
closed.

## Plan to close the gaps

In order of value:

1. **1-loop kernel derivation from `Propagator` + `SubtractionPoint`.** Half-day.
   Removes the 4 kernel rationals from `Schemes.jt05_reggeon_dp()` (verified
   against JT05).

2. **Demo on $\varphi^4$ Doi-Peliti at 1-loop.** Proves the abstraction.
   Produces $\nu = 1/2 + \varepsilon/12$ end-to-end.

3. **Generalise `enumerate_phi_trees` to any $\phi$ polynomial.** Half-day.

4. **2-loop kernel derivation.** Days. Either we do it (Smirnov-style IBP
   in sympy) or we honestly say "supply your own IBP for non-Reggeon
   propagators."
