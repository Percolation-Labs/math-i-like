# CFAC — Coupled Field Analytic Combinatorics

**What it is.** CFAC is a framework for extracting the structural / asymptotic content of stochastic physical systems by treating them as counting problems and applying analytic combinatorics to the resulting generating functions. At its core is the observation that for a wide class of physical systems — chemical reaction networks embedded in fluctuating fields — the Doi–Peliti / Martin–Siggia–Rose (MSR) generating function is a *Lagrange equation* in the sense of Flajolet–Sedgewick, and therefore carries the full transfer-theorem machinery of analytic combinatorics.

Once this recognition is in place, CFAC becomes a scaffolding that:

- Classifies universality classes by the Puiseux order of the dominant branch point.
- Factorises perturbative integrals as `(counting) × (bridge) × (algebra)`.
- Reaches published 2-loop FRG coefficients from combinatorial primitives.
- Has explicit extensions to transcendental kernels, multivariate (coupled) DSEs, and log-corrected universality.
- Admits hybrid use with traditional field theory: anywhere CFAC's combinatorial decomposition leaves a gap, one plugs in the standard RG / saddle-point / integrable-systems machinery alongside.

This folder contains the theorem papers, applications, and boundary studies that collectively constitute the CFAC programme.

---

## The core claim

A stochastic system with a CRN-in-field description has a dressed-propagator generating function `G(z)` satisfying a polynomial (or more generally algebraic / admissible / log-corrected) Dyson–Schwinger equation
$$ G(z) = z \,\phi(G(z)). $$
The dominant branch point `(z*, G*)` controls large-scale physics. The Puiseux order `k` at that branch is a *universality invariant*:
$$ \tau_k = 1 + 1/k. $$
The Banderier–Drmota theorem caps `k` at dyadic values `{2, 4, 8, …}` for `N`-algebraic (positivity-preserving) systems; signed structure (conservation Ward identities, annihilation channels) opens the non-dyadic strata `k ∈ {3, 5, 7, …}`.

That is the content of **Theorem I**. Theorem II extends to transcendental, multivariate, and log-corrected generating functions without breaking the stratification. An enumerative-boundary tier handles non-D-finite cases via direct counting arguments, and a field-theory-extensions layer injects traditional RG/OPE/EFT-matching machinery where the pure-combinatorial reach stops short.

---

## Structure of the folder

### Theorem papers (core)

- **`cfac_theorem.tex`** — Theorem I. The basic stratification theorem for polynomial DSEs. Proves the Puiseux-order classification, the dyadic cap, and the branch-gap contour integral for non-perturbative nucleation rates. Includes the scope section explicitly listing what CFAC does / does not reach.

- **`cfac_theorem_II.tex`** — Theorem II. Extension to (IIa) admissible / transcendental DSE kernels, (IIb) multivariate coupled DSEs via Pemantle–Wilson, (IIc) algebraic-logarithmic singularities for marginal universality. Each sub-theorem has a full proof and at least one textbook reproduction (Cayley tree, Harris multi-type GW, FS sqrt·log). Includes a landscape figure showing where each system class sits. Also contains the signed-projector derivation of the C₃ stratum for conserved-DP.

### Applications (specific systems)

- **`ldw_kirchhoff_recovery.tex`** — CFAC rederivation of the Le Doussal–Wiese–Chauve two-loop roughness coefficient `c = 0.14331`, to relative error 2×10⁻⁴. The most rigorous numerical demonstration that CFAC reaches 2-loop FRG results from combinatorial primitives.

- **`manna_c3_slotting.tex`** — The `C₃` stratum for the Manna / conserved-DP class. Headline is explicitly labelled a *Slotting Conjecture*: algebraic accessibility of `C₃` is proven, the skeleton `τ₀ = 4/3` matches observation to within 3–5%, but the rigorous derivation of `γ₃` from the RPV action is deferred. Non-trivial cross-check: `τ₀ = 4/3` also equals `ν_⊥ = 1/(2−ζ)` read off LDW's `ζ = 5/4`. Includes the methodological warning about a wrong-path "DP plus small correction" framing.

- **`ac_ants.tex`** — Ant colonies as canonical coupled particle-field systems. Six predictions verified by simulation. The "recipe book" test case for the framework.

### Work-in-progress / supporting papers

Four papers previously sitting here have been moved to **`../wip/cfac/`** because they either restate material already contained in the theorem papers / this README, or (in one case) rewrite classical results in CFAC language without producing new constants or theorems. They are preserved as part of the record:

- **`wip/cfac/cfac_paper.tex`** — The original foundational framing paper (CFAC as a reorganisation of Doi–Peliti / MSR). Largely superseded by Theorem I + this README; retained for historical reference to the framing.

- **`wip/cfac/cfac_experiments.tex`** — Landscape / computational-experiments paper. DSE rate-space plots, signed-tree Monte Carlo, stratification tests. Useful material, but overlaps with the stratification figure in Theorem II and with `dse_landscape`.

- **`wip/cfac/dse_landscape.tex`** — Visual / geometric view of the DSE stratification in coupling-space slices. Short companion piece; more naturally a figure-and-commentary appendix to Theorem I than a standalone paper.

- **`wip/cfac/enumerative_boundary.tex`** — *Counting Past the Boundary*. Rewrites three classical results (LERW as Kirchhoff ratio via Wilson's algorithm; TAP complexity via Harer–Zagier ribbon-graph enumeration; KPZ upper tails as replica CRN with Schur structure) in CFAC enumeration language. The paper itself concedes "these are not new mathematics" and that none produces the open exponent in closed form — `ν_3` remains a finite-size scaling extraction of a known Kirchhoff ratio; TAP `Φ_0` is still open for general `p`-spin; KPZ moments only control the integer-`n` side of the replica continuation. Valuable as an honest scope-widening note, but does not deliver the results the earlier README pitch claimed ("Kenyon 5/4 / Kozma 1.624 recovered") — the Kenyon value is SLE₂, and Kozma's is extracted numerically, not derived.

### Field-theory extensions (the "plug in physical ansatz" layer)

These are the items from the scope discussion where CFAC's pure-combinatorial reach ends and we need to borrow from traditional field theory. The modules in `rdft/ac/` (not `paper/cfac/`) implement them concretely:

- **Multi-point correlations** (`rdft/ac/multipoint.py`) — n-point correlator generating functions via n-variable Lagrange equations. Reduces the per-species amplitude as 1/n while preserving the total 1/√(2π), matching the Theorem IIa single-species result.

- **Connes–Kreimer Hopf flow** (`rdft/ac/hopf_flow.py`) — RG flow as antipode in the Hopf algebra of rooted forests. Concrete antipodes for the first three loop orders, reproducing the classical Zimmermann forest formula (`S(T₂) = T₁² - T₂` etc.).

- **OPE anomalous dimensions** (`rdft/ac/ope.py`) — composite-operator scaling dimensions via pointed-graph counting. Reproduces the textbook `γ_{φ²} = (N+2)/(N+8) · ε` at 1-loop for O(N) theory exactly.

- **EFT matching** (`rdft/ac/eft_matching.py`) — tree-level heavy-mode integration in coupled CRNs, giving explicit Wilson coefficients as symbolic expressions with the expected 1/(1+m_H) decoupling.

- **Topology sketch** (`rdft/ac/topology_sketch.py`) — scoping note on Gromov–Witten / Donaldson–Thomas-style moduli-space counting. Honest verdict: rational subcases (P¹) fit CFAC at the k=1 boundary; CY3 DT with MacMahon structure is outside scope; topological recursion and Fano quantum cohomology are potentially in scope via multivariate DSE.

### Reproducibility

- **`reproduce.py`** / **`reproduce_README.md`** — script and instructions to regenerate the numerical and symbolic content of the papers from primitives. Verification against 98+ passing tests in `rdft/tests/`.

- **`figures/`**, **`figures_ants/`** — plotting scripts and generated PDFs.

---

## Scope statement

**CFAC is a combinatorial scaffolding that makes one projection of field-theoretic content visible.** It is strongest when:

1. The physical system admits a Doi–Peliti / MSR action with polynomial (or admissible) vertices.
2. The target observable is a *tail exponent* or *branch-point residue* — not the full correlation structure.
3. The generating function is algebraic, `D`-finite, or transcendental-admissible (Hayman class).
4. The observable is scalar — not matrix-valued (replicas, random matrices).

Within this scope, CFAC provides:

- A classification of universality classes by Puiseux order (Theorem I, II).
- A computational algorithm for 2-loop and some 3-loop FRG coefficients (LDWC paper).
- A scoping framework that cleanly separates what is inside / outside reach.

**Outside the stated scope:** non-`D`-finite generating functions (SAW, LERW in `d=3` except via Kirchhoff enumeration, lattice Green's functions in general); matrix-valued observables (Kac–Rice determinants, TAP complexity beyond 1-genus approximation, random matrix theory); full conformal-field-theory spectra (beyond the leading scaling operator); real-time dynamics (transport, Kubo, Keldysh); and topology / gauge / integrable structures where the combinatorics is radically different.

The enumerative-boundary note (`wip/cfac/enumerative_boundary.tex`) rewrites some of these cases (LERW, TAP, KPZ upper tails) in CFAC enumeration language, but — as that paper itself states — without producing the open exponents in closed form; it is a scope-widening exercise rather than a result-delivering one. The field-theory extensions layer addresses the rest by plugging in standard RG / OPE / EFT-matching machinery alongside CFAC's combinatorial skeleton.

---

## `d`-dimensional space and random graphs

CFAC is *zero-dimensional* at its core: the generating function `G(z)` encodes the sum over all diagrams without regard to spatial structure. Spatial dimension `d` enters CFAC in three ways:

1. **Upper critical dimension from the DSE.** The engineering dimension of the coupling in `φ` fixes `d_c(k) = 2k/(k-1)` for the `C_k` stratum: DP has `d_c = 4`, Manna/CDP has `d_c = 4`, etc. Above `d_c` the mean-field skeleton `τ_k = 1 + 1/k` is exact; below `d_c` it is dressed by `γ_k(d) ∼ ε²` corrections from loop diagrams.

2. **Rank-`k` bridge values.** The loop integrals CFAC reaches (via `rdft/ac/bridge.py`) carry explicit `d`-dependence through the Feynman-parameter measure and the Symanzik polynomial. At `d = d_c`, the bridge is a dimensionless constant (`B_k = 2 / [(k-1)! (4π)^k]`); below `d_c`, it becomes a function of `d` through gamma-function ratios.

3. **Spatial correlations via coupled DSE.** For systems where spatial structure matters explicitly (Manna via qEW depinning, ants via Keller–Segel), the coupled DSE itself depends on the spatial propagators of the background field. The Le Doussal–Wiese mapping `n(x) = n₀ + ∇²u(x)` is the cleanest example: a purely spatial relation translating between the particle-count generating function and the interface roughness.

**Random graphs and network substrates.** The full generality of Theorem I is that the DSE is a Lagrange equation for any system that counts *something* combinatorial. On random graphs (Erdős–Rényi, configuration model, scale-free), the relevant DSE captures:
- The Susceptible–Infected (SI) / SIR generating function on a random graph as `G = z · φ_{graph}(G)` where `φ_{graph}` encodes the degree distribution (solved exactly for ER, handles configuration models cleanly).
- Percolation / giant-component transitions as branch-point transitions of the graph DSE.
- Random regular graphs as a specific integer degree `k`, giving `τ_k = 1 + 1/k`.

CFAC on random graphs is therefore a direct application of Theorem I to a non-physical (social / epidemiological / ecological) substrate: the stratification applies identically because the mathematics is the same. This is a principal reason why the framework deserves the "C" (coupled, and also combinatorial) rather than just being a field-theoretic reformulation — the underlying structure is transferable across any counting problem with an algebraic generating function.

---

## Using CFAC alongside traditional methods

A practical point: CFAC does not claim to replace standard field-theoretic calculation. Its value is as a COMPLEMENTARY LENS:

- **Where CFAC exposes structure cleanly**, use it and save work. The LDWC 2-loop coefficient 0.14331 comes out as `X^(2) / (9√2 γ)` with the residual integral `γ = 0.5482…` — a clean factorisation that standard FRG obscures.
- **Where CFAC runs out** (non-`D`-finite GF, matrix observables, topology), fall back to direct enumeration (enumerative boundary) or to traditional FT (RG flow, FRG, bootstrap).
- **In mixed cases**, do both: use CFAC to identify the stratum and the skeleton exponent; use standard RG / FRG to dress it; verify that the two arrive at the same answer. The Manna `C_3` paper is this pattern executed explicitly.

The phrase "plug in physical ansatz" is literal: the framework does not ban you from adding a standard one-loop self-energy calculation or a bootstrap bound by hand — it just provides the scaffolding on which such additions hang cleanly.

---

## Pointers to the code

- `rdft/ac/stratification.py` — Puiseux-order computation for polynomial DSEs.
- `rdft/ac/bridge.py` — bridge functions for 1-loop self-energies in coupled systems.
- `rdft/ac/admissible.py` — Theorem IIa (transcendental kernels).
- `rdft/ac/multivariate.py` — Theorem IIb (coupled DSE with smooth / multiple / cone point detection).
- `rdft/ac/log_corrections.py` — Theorem IIc (algebraic-log singularities).
- `rdft/ac/signed_projector.py` — signed-projector origin of the `C_3` stratum for CDP.
- `rdft/ac/multipoint.py`, `hopf_flow.py`, `ope.py`, `eft_matching.py`, `topology_sketch.py` — field-theory-extension modules.
- `rdft/ac/manna_depinning.py`, `manna_full.py`, `manna_2loop.py` — Manna / CDP specific.
- `tests/` — 100+ passing tests verifying each module against published results.

---

## Reading order (suggested)

1. `cfac_theorem.tex` § 1–3 — the core theorem and stratification claim.
2. `cfac_theorem.tex` scope section (§ Discussion) — what CFAC does and does not do.
3. `cfac_theorem_II.tex` — the three extensions (admissible, multivariate, log-corrected) with proofs.
4. `ldw_kirchhoff_recovery.tex` — the 2-loop validation (strongest numerical result).
5. `manna_c3_slotting.tex` — canonical non-DP application + methodological warning (slotting claim is conjectural — see the paper's own status grading).
6. `ac_ants.tex` — six predictions verified by simulation.
7. Field-theory extension modules in `rdft/ac/` — concrete demos.
8. `../wip/cfac/` — superseded framing (`cfac_paper`), landscape material (`cfac_experiments`, `dse_landscape`), and the scope-widening enumeration note (`enumerative_boundary`).

---

*Percolation Labs, April 2026.*
