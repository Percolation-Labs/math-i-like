# RDFT: Reaction-Diffusion Field Theory via Analytic Combinatorics

## Vision

**Given any chemical reaction network on any graph, automatically produce its
critical exponents — via combinatorics, not just traditional RG.**

A user describes a stochastic reaction-diffusion system (species, reactions,
rates, underlying graph). The software:

1. Constructs the Doi-Peliti Liouvillian from the stoichiometry matrix.
2. Extracts Feynman vertices and generates **all** Feynman diagrams to a given
   loop order.
3. Computes Symanzik polynomials (Ψ, Φ) — encoding the graph topology and
   causality structure of each diagram.
4. Evaluates parametric integrals via ω-integration and Euler-Beta reduction.
5. Renormalises via BPHZ (Connes-Kreimer Hopf algebra).
6. Extracts RG functions (β, η, ν) and critical exponents.
7. **Independently**, derives the same exponents from the Analytic Combinatorics
   route: Lagrange inversion of the Dyson-Schwinger equation, singularity
   analysis, and the transfer theorem.
8. Renders all Feynman diagrams and provides the AC↔QFT correspondence table.
9. Substitutes spectral dimension d_s for arbitrary graphs (fractals,
   random trees, scale-free networks, real networks).
10. Verifies predictions against Monte Carlo simulations (Rust engine).

The key intellectual contribution: **the AC route provides a cleaner argument
for dimensional analysis and RG** than the traditional approach. The singularity
type of the generating function (square-root branch point, pole, essential
singularity) directly determines the universality class. This is the unifying
principle from the companion tutorial.

---

## The Three Routes to Critical Exponents

All three arrive at the same singularity:

```
                    Stoichiometry Matrix (C, G, W)
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
          DOI-PELITI      GENERATING     ANALYTIC
         FIELD THEORY      FUNCTION    COMBINATORICS
                │          (PGF/EGF)        │
                │             │              │
          Path integral    Master eq.    Symbolic method
          Peliti action    PDE → chars    grammar → GF
                │             │              │
          Feynman rules   Saddle point   Lagrange eq.
          loop integrals   = det. orbit  T = zφ(T)
                │             │              │
          RG, β(λ*)=0    Conservation    IFT failure
          ε-expansion      law orbit    branch point
                │             │              │
                └─────────────┼──────────────┘
                              ▼
                      SINGULARITY TYPE
                     (universality class)
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
               Square-root   Pole    Essential
               branch pt.           singularity
               n^{-3/2}    n^{k-1}   n! growth
               ρ~t^{-d/2}  mean-field  instantons
```

---

## Architecture

```
rdft/
├── rdft/                          # Python package
│   ├── core/                      # Layer 1: CRN → Liouvillian
│   │   ├── reaction_network.py    # Species, Reaction, stoichiometry  [DONE]
│   │   ├── generators.py          # Q[∂_z, z], Liouvillian, vertices  [DONE]
│   │   └── field_theory.py        # Lagrangian, Doi shift, action
│   │
│   ├── graphs/                    # Layer 2: Feynman diagram generation
│   │   ├── incidence.py           # FeynmanGraph, E matrix, Kirchhoff [DONE]
│   │   ├── corolla.py             # Primitive corollas from vertices
│   │   ├── shuffle.py             # Shuffle product (Amarteifio Def.12)
│   │   ├── enumerate.py           # Systematic 1PI enumeration to L loops
│   │   ├── render.py              # Diagram rendering (graphviz/tikz/SVG)
│   │   └── spectral.py            # Spectral dimension d_s for graphs
│   │
│   ├── integrals/                 # Layer 3-4: Graph polynomials & integrals
│   │   ├── symanzik.py            # Ψ, Φ from spanning trees/2-trees  [Ψ DONE, Φ kinematic STUB]
│   │   ├── parametric.py          # ω-integration, parametric I(G;d)  [DONE]
│   │   └── kirchhoff.py           # Kirchhoff polynomial utilities
│   │
│   ├── rg/                        # Layer 5-6: Renormalisation & RG
│   │   ├── bphz.py                # Connes-Kreimer coproduct, antipode [DONE]
│   │   ├── rg_functions.py        # β, η, ν, fixed points, exponents  [DONE]
│   │   └── renormalize.py         # Z-factors, MS-bar scheme
│   │
│   ├── ac/                        # Layer 7: Analytic Combinatorics route
│   │   ├── lagrange.py            # DSE as Lagrange eq, inversion formula
│   │   ├── singularity.py         # IFT failure, branch point detection
│   │   ├── transfer.py            # Transfer theorem → coefficient asymptotics
│   │   ├── borel.py               # Borel transform, factorial divergence
│   │   └── correspondence.py      # AC↔QFT dictionary (the grand table)
│   │
│   └── pipeline.py               # End-to-end: CRN → exponents (both routes)
│
├── simulations/                   # Rust simulation engine
│   ├── src/
│   │   ├── lattice.rs             # Hypercubic lattice in d dimensions
│   │   ├── graph.rs               # General graph (adjacency list)
│   │   ├── brw.rs                 # Branching random walk (Gillespie)
│   │   ├── reaction.rs            # General reaction-diffusion sim
│   │   └── lib.rs                 # PyO3 bindings
│   ├── Cargo.toml
│   └── README.md
│
├── tests/
│   ├── validation/                # Reproduce known results
│   │   ├── test_tier1.py          # Trivial: A→∅, ∅→A                [DONE]
│   │   ├── test_tier2.py          # One-loop exact: 2A→∅, A+B→∅
│   │   ├── test_tier3.py          # BWS, directed percolation
│   │   ├── test_tier4.py          # Non-regular graphs (Sierpinski etc.)
│   │   └── test_ac_route.py       # AC reproduces same exponents
│   └── unit/
│       ├── test_generators.py
│       ├── test_shuffle.py
│       ├── test_symanzik.py
│       ├── test_parametric.py
│       ├── test_bphz.py
│       └── test_lagrange.py
│
├── examples/
│   ├── pair_annihilation.py       # 2A→∅: full pipeline, both routes
│   ├── brw_hypercubic.py          # BRW paper reproduction
│   ├── brw_sierpinski.py          # BRW on fractal
│   ├── contact_process.py         # DP universality class
│   └── three_species.py           # Novel: A+B+C→∅
│
├── docs/                          # Reference material
│   ├── Amarteifio-S-2019-Phd-Thesis.pdf
│   ├── brw.pdf
│   └── generating_functions_field_theory_AC_tutorial.tex
│
├── PLAN.md                        # This file
└── README.md
```

---

## Implementation Plan

### Phase 0: Foundations [DONE]
Core data structures, Heisenberg-Weyl generators, Feynman graph representation.

**Completed:**
- `core/reaction_network.py` — CRN dataclass, 7 factory methods
- `core/generators.py` — Q[∂_z, z] from thesis eq. (1.36), Liouvillian, vertices
- `graphs/incidence.py` — FeynmanGraph, incidence matrix, Kirchhoff, 1PI check
- `integrals/symanzik.py` — Ψ polynomial (complete), Φ kinematic (stubbed)
- `integrals/parametric.py` — ω-integration, parametric integral, ε-expansion
- `rg/bphz.py` — Connes-Kreimer coproduct, antipode, forest formula
- `rg/rg_functions.py` — β, η, fixed points, critical exponents
- `tests/validation/test_tier1.py` — 25+ tests, all passing

### Phase 1: Diagram Generation Engine
**Goal:** Automatically generate all 1PI Feynman diagrams to loop order L from any CRN.

This is the combinatorial heart. From the Liouvillian vertices, construct:

1. **Corollas** — primitive half-edge stars from each vertex type.
   For a vertex with k incoming and l outgoing legs, the corolla has k+l half-edges.
   (Amarteifio §2.5, Def. 11)

2. **Shuffle product** — pair half-edges to form internal propagators.
   The shuffle product of corollas generates all graphs at a given loop order.
   (Amarteifio Def. 12, Theorem 2.5.1)

3. **Isomorphism filtering** — canonical hash to remove duplicates.
   Symmetry factor = 1/|Aut(G)| from the EGF exponential formula.

4. **1PI filter** — discard graphs with bridges (already implemented).

**Files:** `graphs/corolla.py`, `graphs/shuffle.py`, `graphs/enumerate.py`

**Validation:**
- Gribov process (BWS): reproduce the 7 distinct one-loop integrals from thesis §2.5
- 2A→∅: single one-loop diagram
- A+B→∅: two vertex types, enumerate all one-loop diagrams

### Phase 2: Symanzik Φ and Graph Polynomials
**Goal:** Complete the second Symanzik polynomial and enable general kinematics.

The Symanzik polynomials encode the **causality domain** — the topology of
momentum flow through the diagram. They are the most powerful objects in the
parametric representation:

- **Ψ (first Symanzik):** sum over spanning trees T of ∏_{e∉T} α_e.
  Encodes which edges form loops. Already implemented.

- **Φ (second Symanzik):** sum over 2-trees (spanning 2-forests) weighted
  by squared momentum flow. Encodes kinematic dependence.
  Currently stubbed — needs the 2-tree enumeration algorithm.

**The causality connection:** For a graph G with external momenta {p_i},
Φ determines which momentum channels are "visible" to the integral.
The zeros of Ψ are the Landau singularities — they define the
causality constraints of the S-matrix (Cutkosky rules). Even in the
Euclidean (reaction-diffusion) setting, Ψ and Φ completely determine
the integral's analytic structure in d.

**Implementation:**
1. Enumerate all 2-forests (spanning forests with exactly 2 components)
   via deletion-contraction on the Kirchhoff matrix.
2. For each 2-forest, compute the squared momentum flowing between components.
3. Φ = Ψ · Σ_e m_e² α_e + Σ_{2-forests} s_F · ∏_{e∉F} α_e
   where s_F is the Mandelstam invariant for the 2-forest F.

**Files:** `integrals/symanzik.py` (complete the Φ computation)

**Validation:**
- One-loop self-energy: Φ = (m₁² α₁ + m₂² α₂)(α₁ + α₂) + p² α₁α₂
- Sunset (2-loop): verify against Bogner-Weinzierl (2010) formula
- Triangle: verify momentum routing through 3 channels

### Phase 3: Feynman Diagram Rendering
**Goal:** Visualise all generated diagrams with proper labelling.

For each diagram, render:
- Vertices (labelled by reaction type and coupling)
- Internal edges (labelled by Schwinger parameter α_e and mass)
- External legs (labelled by species and momentum)
- Loop number, symmetry factor, degree of divergence

**Output formats:** SVG (interactive), TikZ (LaTeX), GraphViz (quick view)

**Files:** `graphs/render.py`

### Phase 4: The AC Route
**Goal:** Derive critical exponents purely from analytic combinatorics.

This is the novel contribution. For each CRN:

1. **Dyson-Schwinger equation → Lagrange equation:**
   The DSE for the dressed propagator G = G₀·Φ(G) is a Lagrange equation
   T = z·φ(T). The perturbative expansion (Feynman diagram sum) is
   Lagrange inversion: [z^n]T = (1/n)[T^{n-1}]φ(T)^n.

2. **Singularity detection:**
   The Lagrange conditions 1 = z*φ'(T*), T* = z*φ(T*) determine the
   branch point. The IFT failure is the Landau pole / critical scale.

3. **Transfer theorem → asymptotics:**
   Near the branch point, T ~ T* - C√(z* - z) (square-root branch).
   Transfer theorem: [z^n]T ~ C·n^{-3/2}·z*^{-n}.
   Integrated: survival probability ~ t^{-1/2} → density exponent.

4. **Correspondence table:**
   For each process, produce the dictionary:
   - Lagrange equation ↔ Dyson-Schwinger equation
   - Branch point ↔ Landau pole
   - Singularity type ↔ universality class
   - Transfer theorem exponent ↔ critical exponent from RG
   - Symmetry factors ↔ EGF overcounting

5. **Borel analysis:**
   Perturbation series coefficients grow as n!·g*^{-n}. The Borel
   transform has a singularity at g* (the Lagrange branch point).
   This is the instanton scale. The transfer theorem in the Borel
   plane determines the non-perturbative corrections.

**Why this is better than traditional RG for dimensional analysis:**
In the traditional approach, the upper critical dimension d_c comes from
power counting (engineering dimensions of couplings). The AC approach
derives d_c from the singularity structure: d_c is where the Lagrange
branch point hits the boundary of the convergence disk (z* = 1).
Below d_c, fluctuations dominate and the singularity moves inside
the disk. The exponent is read off from the singularity type, not
from ε-expansion loop integrals. This is conceptually cleaner and
computationally simpler for one-loop exact results.

**Files:** `ac/lagrange.py`, `ac/singularity.py`, `ac/transfer.py`,
         `ac/borel.py`, `ac/correspondence.py`

**Validation:**
- 2A→∅: AC gives α = 1/2 in d=1 (matches Doi-Peliti and Lee 1994)
- SIR: AC gives n^{-3/2} final-size tail (matches Borel distribution)
- A+A→∅ via first-passage: AC gives t^{-1/2} from Lagrange GF

### Phase 5: Spectral Dimension and General Graphs
**Goal:** Substitute d → d_s for processes on arbitrary graphs.

From the BRW paper (Bordeu, Amarteifio et al. 2019):
- On regular lattices: d_s = d (trivial)
- On Sierpinski carpet: d_s ≈ 1.86
- On random trees: d_s = 4/3
- On preferential attachment: d_s ≥ 4 (mean-field)

The substitution d → d_s is valid when the Laplacian does not renormalise
(no anomalous dimension). The scaling of the BRW volume explored is:

- ⟨a^p⟩(t) ~ t^{(pd-2)/2} for d < d_c = 4
- ⟨a^p⟩(t) ~ t^{2p-1} for d ≥ 4 (mean-field)
- P(a) ~ a^{-(1+2/d)} (cluster size distribution)

Replace d → d_s everywhere: exponents, d_c comparisons, scaling forms.

**Files:** `graphs/spectral.py`

**Validation:** Reproduce Tables 3.8, 3.9 from thesis and all BRW paper figures.

### Phase 6: Rust Simulation Engine
**Goal:** High-performance Monte Carlo to verify theoretical predictions.

The BRW paper used simulations with 10^6-10^9 realisations per lattice size.
We need a fast engine for:

1. **Hypercubic lattices** in arbitrary d (periodic/absorbing BCs)
2. **General graphs** (adjacency list: Sierpinski, random tree, PA network, real networks)
3. **Gillespie algorithm** for exact stochastic simulation
4. **Reaction-diffusion:** hopping + arbitrary reactions from CRN specification
5. **Observables:** density ρ(t), distinct sites visited a(t), moments ⟨a^p⟩

**Implementation:**
- Rust core with PyO3 bindings for Python interop
- Parallel realisations via rayon
- Memory-efficient site tracking (bitsets for visited sites)

**Files:** `simulations/src/`

**Validation:**
- Reproduce BRW paper Fig. 2 (regular lattices d=1,2,3,5)
- Reproduce BRW paper Fig. 3 (cluster size distributions)
- Reproduce BRW paper Fig. 4 (Sierpinski, random tree, PA)
- Reproduce BRW paper Fig. 5 (Facebook, yeast networks)

### Phase 7: Novel Results
**Goal:** Publishable predictions for systems not in the literature.

1. **Multi-species annihilation A+B→∅ on Sierpinski carpet**
   - Expected: α = d_s/4 ≈ 0.473 (vs d/4 = 0.5 for d=2)

2. **Contact process on random tree (d_s = 4/3)**
   - d_s < d_c = 2 → mean-field applies? But d_s/2 = 2/3 < 1, needs careful analysis.

3. **Three-species A+B+C→∅** (novel)
   - Automatically generate diagrams, compute d_c
   - AC route: what is the Lagrange equation?

4. **AC-improved dimensional analysis**
   - For each process, compare traditional ε-expansion with AC singularity route
   - Show that the AC route gives exact results where ε-expansion is approximate
   - The Lagrange branch point encodes the non-perturbative scale

5. **Gribov process on Erdős-Rényi random graph**
   - d_s computed numerically from graph Laplacian spectrum

---

## Validation Targets (Ordered by Difficulty)

### Tier 1: Trivial [DONE — all tests passing]
1. A → ∅: Q = -δz∂_z, no diagrams, ρ = ρ₀e^{-δt}
2. ∅ → A: Q = σ(z-1), constant source
3. A ⇌ 2A (birth-death at criticality): ρ ~ 1/t

### Tier 2: One-loop exact (Lee 1994)
4. 2A → ∅: d_c = 2, α = d/2 exact
5. 2A → A: same universality class
6. A+B → ∅: d_c = 4, α = d/4

### Tier 3: Branching processes (thesis Ch.3, BRW paper)
7. BWS (Gribov process): d_c = 4, reproduce thesis Tables 3.8-3.9
8. A → 2A, 2A → ∅ (directed percolation): ν_⊥ = 1 + O(ε)

### Tier 4: Non-regular graphs
9-11. Tiers 2-3 on Sierpinski (d_s ≈ 1.86), random tree (d_s = 4/3), PA (d_s ≥ 4)

### Tier 5: AC route validation
12. For each Tier 2-3 process, derive exponents via Lagrange/transfer theorem
13. Verify AC exponent matches RG exponent
14. Produce correspondence tables

### Tier 6: Novel
15. A+B+C → ∅: compute d_c, predict exponents
16. Contact process on random tree
17. Gribov on Erdős-Rényi

---

## Key Design Decisions

### Symbolic vs Numeric
- All graph and polynomial operations are symbolic (SymPy)
- Numerical only at final stage (fixed point search, exponent evaluation)
- ε kept symbolic throughout

### Graph Representation
- Primary: incidence matrix E as SymPy Matrix
- Derived: Laplacian L = E·Eᵀ, Kirchhoff polynomial
- Isomorphism: canonical edge hash (faster than full iso check)

### Integral Representation
- Always parametric form (not momentum space)
- Parametric form handles arbitrary d, arbitrary graphs, BPHZ naturally

### Renormalisation Scheme
- Minimal subtraction (MS-bar): extract 1/ε poles
- Matches Lee (1994) and Amarteifio (2019)

### The AC Layer
- Lagrange inversion via SymPy formal power series
- Singularity detection via implicit function theorem conditions
- Transfer theorem coefficients via asymptotic expansion
- Borel transform for factorial divergence diagnostics

### Rust Simulation
- PyO3 for seamless Python↔Rust interop
- Gillespie SSA for exact stochastic dynamics
- rayon for parallel realisations
- Graph input: NetworkX → adjacency list → Rust

---

## Dependencies

```
# Python
sympy          # symbolic algebra
numpy          # numerical support
scipy          # fixed-point solving
networkx       # graph operations, isomorphism
matplotlib     # plotting
graphviz       # diagram rendering
pytest         # testing
pyo3           # Rust bindings (via maturin)

# Rust
rand           # random number generation
rayon          # parallelism
pyo3           # Python bindings
petgraph       # graph data structures
```

---

## References

1. Amarteifio, S. (2019). *PhD Thesis*, Imperial College London.
   — Chapters 1-3: CRN → Liouvillian → Feynman graphs → Symanzik → RG
2. Bordeu, I., Amarteifio, S., et al. (2019). *Sci. Rep.* 9:15590.
   — BRW on general graphs, scaling with spectral dimension
3. Amarteifio, S. (2026). *Generating Functions, Field Theory, and AC Tutorial.*
   — The AC↔QFT correspondence table, Lagrange = DSE identification
4. Lee, B.P. (1994). *J. Phys. A* 27:2633. — Exact exponents for kA→∅
5. Connes, A. & Kreimer, D. (1998, 2000). — Hopf algebra of renormalisation
6. Flajolet, P. & Sedgewick, R. (2009). *Analytic Combinatorics*. — Transfer theorem
7. Yeats, K. (2017). *A Combinatorial Perspective on QFT*. — DSE as combinatorial equations
8. Tauber, U.C. et al. (2005). *J. Phys. A* 38:R79. — Doi-Peliti review
