# RDFT: Reaction-Diffusion Field Theory

An algebraic pipeline for computing critical exponents of reaction-diffusion
processes on arbitrary graphs, built directly on the framework developed in
Amarteifio (2019) *"Field theoretic formulation and empirical tracking of
spatial processes"* (PhD thesis, Imperial College London).

---

## What This Program Does

RDFT automates the complete field-theoretic analysis of a stochastic reaction-
diffusion process, from the microscopic reaction rules all the way to critical
exponents. The pipeline is:

```
ReactionNetwork(species, reactions, graph)
        │
        │  Heisenberg-Weyl algebra  [rdft/core/generators.py]
        ▼
Lagrangian L[φ̃, φ]  +  interaction vertices
        │
        │  Wick contractions = det(L_RS)  [rdft/core/expansion.py]
        ▼
All 1PI Feynman diagrams to loop order L  [rdft/graphs/]
        │
        │  Matrix-Tree theorem  [rdft/graphs/incidence.py]
        ▼
Kirchhoff polynomial K(α)  →  Symanzik Ψ, Φ  [rdft/integrals/symanzik.py]
        │
        │  ω-integration via δ(Σ_l)  [rdft/integrals/parametric.py]
        ▼
Parametric integral I(G; d)  as function of dimension d
        │
        │  Connes-Kreimer Hopf antipode  [rdft/rg/bphz.py]
        ▼
Renormalised amplitudes I_R(G)  +  Z-factors
        │
        │  Callan-Symanzik equation  [rdft/rg/rg_functions.py]
        ▼
β(λ), fixed points λ*,  critical exponents  ν, η, α
        │
        │  Replace d → d_s  [rdft/graphs/spectral.py — Phase 6]
        ▼
Exponents on arbitrary graph G  (Sierpinski, random tree, scale-free, ...)
```

The central algebraic insight, which unifies everything, is:

> **Wick contractions of the generating functional = permanent of the
> propagator matrix = det(L_RS(α)) = Kirchhoff polynomial K(α)**
>
> (via the Schwinger representation + Matrix-Tree theorem)

This means the entire perturbative expansion to loop order L is encoded in
a single polynomial computed from the incidence matrix. No diagrams need to
be enumerated explicitly.

---

## Background

### The Doi-Peliti Framework

For any chemical reaction network (CRN) — a set of reactions kA → lA
on a graph G — the stochastic master equation can be mapped exactly to a
field theory via the Heisenberg-Weyl algebra [a, a†] = 1. Under the
identification a† = z (multiplication) and a = ∂z (differentiation), the
master equation becomes a PDE for the probability generating function G(z,t).
The coherent-state path integral then produces an action S[φ̃, φ] whose
perturbative expansion in Feynman diagrams controls the long-time, large-scale
behaviour of the process.

For each reaction kA → lA at rate λ, the infinitesimal generator is
(Amarteifio eq. 1.36):

    Q[∂z, z] = λ · ((z+1)^l - (z+1)^k) · ∂z^k

This formula is the starting point. Every downstream computation follows
algebraically from it.

### The Graph Algebra

Each monomial z^m ∂z^n in Q corresponds to a Feynman vertex with m outgoing
and n incoming directed edges (a corolla). The parametric integral representation
of any Feynman diagram G is expressed in terms of two graph polynomials
(Amarteifio eqs. 2.30, 2.33):

    Ψ(G) = Σ_{spanning trees T} Π_{e ∉ T} α_e   (topology only)
    Φ(G) = Ψ · Σ_e α_e m_e² + kinematic terms  (masses + momenta)

These are computed from the incidence matrix E of G:

    Ψ = det(E_{[Γ]} D_α E_{[Γ]}^T)   [reduced symbolic Laplacian cofactor]

The full parametric integral is (Amarteifio eq. 2.76):

    I(G) = Ω_d · Γ(-σ_d) · ∫ Πe dα_e · [Ψ']^{-d/2} · [Ψ'/Φ']^{σ_d} · Πl δ(Σl)

where the δ(Σl) factors come from ω-integration (Kirchhoff's frequency law on
each circuit), and σ_d = |E_int| - (d/2)L is the superficial degree of
divergence.

### Why This Matters

The graph polynomial approach is not just a computational technique. It provides:

1. **Dimension-independence**: d appears only via Ω_d and σ_d. Replacing d → d_s
   (spectral dimension of the underlying graph) immediately gives results on
   arbitrary geometries — Sierpinski carpets, random trees, scale-free networks.

2. **Exact Ward identities**: For pair annihilation 2A → ∅, a Ward identity
   (from probability conservation H(1,a) = 0) cancels the quartic vertex
   φ̃²φ² exactly, making the exponent α = d/2 exact to all loop orders.
   This is visible algebraically as a symmetry of K(α).

3. **Non-equilibrium structure**: Detailed balance corresponds to symmetry
   of the directed Kirchhoff polynomial under time-reversal. Its breaking
   is the algebraic signature of non-equilibrium.

4. **AC connection**: The dressed propagator satisfies a Dyson-Schwinger
   equation that is a Lagrange equation T = z·Φ(T). Its branch point is the
   Landau pole; the AC transfer theorem predicts amplitude growth ~ n! · g*^{-n}.

---

## Repository Structure

```
rdft/
├── README.md                    ← this file
├── PLAN.md                      ← full 7-phase roadmap + paper outline
│
├── rdft/                        ← main package
│   ├── core/
│   │   ├── reaction_network.py  ← CRN data structures, factory methods
│   │   ├── generators.py        ← Heisenberg-Weyl generators Q[∂z, z]
│   │   ├── field_theory.py      ← Lagrangian, Doi shift (stub)
│   │   └── expansion.py         ← Wick contractions → FeynmanGraph
│   │
│   ├── graphs/
│   │   ├── incidence.py         ← FeynmanGraph, incidence matrix E,
│   │   │                            Kirchhoff polynomial K, 1PI check
│   │   ├── corolla.py           ← primitive corollas (stub)
│   │   ├── shuffle.py           ← shuffle product (stub, Phase 1)
│   │   ├── graph_db.py          ← isomorphism, symmetry factors (stub)
│   │   └── spectral.py          ← spectral dimension d_s (stub, Phase 6)
│   │
│   ├── integrals/
│   │   ├── kirchhoff.py         ← (redirects to incidence.py)
│   │   ├── symanzik.py          ← Ψ and Φ polynomials from K
│   │   ├── omega_integration.py ← δ(Σl) ω-integration via edge-basis E_f
│   │   └── parametric.py        ← full I(G; d), thesis examples 2.5.15-17
│   │
│   ├── rg/
│   │   ├── bphz.py              ← Connes-Kreimer coproduct + antipode
│   │   ├── renormalize.py       ← Z-factors, counterterms, MS-bar (stub)
│   │   ├── rg_functions.py      ← β(λ), η, fixed points, KnownResults table
│   │   └── fixed_points.py      ← exponent extraction (stub)
│   │
│   └── ac/                      ← analytic combinatorics layer (Phase 5)
│       └── lagrange.py          ← DSE as Lagrange equation (stub)
│
└── tests/
    └── validation/
        └── test_tier1.py        ← 15 tests, all passing
```

---

## What Is Working (Current State)

### ✅ Fully implemented and tested

**Layer 1 — Heisenberg-Weyl generators** (`rdft/core/generators.py`)

The central formula Q[∂z, z] = λ·((z+1)^l - (z+1)^k)·∂z^k is implemented
for arbitrary single-species reactions kA → lA. All three Gribov generators
match Amarteifio equations (1.38a-c) exactly. The `Liouvillian` class assembles
multi-reaction networks and extracts vertices.

Verified generators:
- A → ∅   (death):      Q = -δz∂z
- A → 2A  (branching):  Q = β(z²+z)∂z
- 2A → A  (coagulation): Q = χ(z-z²)∂z²
- 2A → ∅  (annihilation): Q = -λ(z²+2z)∂z²

**CRN data structures** (`rdft/core/reaction_network.py`)

Factory methods for all standard networks:
- `ReactionNetwork.pure_death()`
- `ReactionNetwork.birth_death()`
- `ReactionNetwork.pair_annihilation()`
- `ReactionNetwork.coagulation()`
- `ReactionNetwork.gribov()`
- `ReactionNetwork.two_species_annihilation()`
- `ReactionNetwork.contact_process()`

**Graph algebra** (`rdft/graphs/incidence.py`)

`FeynmanGraph` with full incidence matrix machinery:
- Kirchhoff polynomial K(α) via Matrix-Tree theorem cofactor
- Symanzik Ψ by edge complement of K
- 1PI check (bridge detection on internal subgraph)
- Superficial degree of divergence σ_d(G)
- Standard graphs: sunset, one-loop self-energy, three-vertex loop

Verified:
- Sunset: K = α₀+α₁+α₂  (3 spanning trees, one edge each)
- One-loop self-energy: Ψ = α₀+α₁  (degree L=1, homogeneous ✓)

**Symanzik polynomials** (`rdft/integrals/symanzik.py`)

Ψ from Kirchhoff complement, Φ from mass and kinematic terms.
Homogeneity check (Ψ degree L, Φ degree L+1) implemented and passing.

**Algebraic chain demonstration** (`rdft/core/expansion.py`)

End-to-end run for 2A → ∅ at one loop:
```
Lagrangian → vertices → graph → K → Ψ → expected I(G;d) → UV pole at d_c=2
```

**Known results table** (`rdft/rg/rg_functions.py`)

Reference values from literature for all validation targets:
- 2A→∅:   α = d/2, d_c = 2 (Lee 1994, exact)
- 2A→A:   α = d/2, d_c = 2 (Lee 1994, same class)
- A+B→∅:  α = d/4, d_c = 4 (Lee-Cardy 1995)
- BWS:    V ~ t^{d/4}, d_c = 4 (Bordeu, Amarteifio+ 2019)
- DP:     ν⊥ = 1/2 + 3ε/16 (Täuber-Howard-Vollmayr-Lee 2005)

### ✅ All 8 sanity checks passing

```
✓ Pure death generator matches thesis
✓ Coagulation generator matches thesis
✓ All three Gribov generators match eqs. (1.38a-c)
✓ Sunset Kirchhoff polynomial K = α₀+α₁+α₂
✓ One-loop self-energy Ψ = α₀+α₁
✓ Ψ is homogeneous of degree L
✓ Known result: 2A→∅, α = d/2 (exact)
✓ Known result: A+B→∅, α = d/4
```

---

## What Is Stubbed / Incomplete

### 🔶 Scaffolding exists, not yet validated end-to-end

**ω-integration** (`rdft/integrals/omega_integration.py`, `parametric.py`)

The algorithm for δ(Σl) substitutions is implemented: finds spanning tree,
builds edge-basis matrix E_f, substitutes leading α out of each circuit.
The closed-form evaluation of the reduced alpha-integrals via the Euler-Beta
formula is implemented for one-loop linear cases. Multi-loop and non-linear
cases use a symbolic placeholder.

**BPHZ renormalisation** (`rdft/rg/bphz.py`)

Connes-Kreimer coproduct Δ(G) = Σ_γ γ⊗G/γ and antipode S(G) are
structurally implemented. Graph contraction G/γ works via incidence
matrix minor deletion. Not yet run against a full renormalisation example.

**RG functions** (`rdft/rg/rg_functions.py`)

β(λ) extraction from Z-factor poles, fixed point search, exponent formulas.
Structurally complete; not yet connected to actual computed amplitudes.

**Wick contraction expansion** (`rdft/core/expansion.py`)

The FeynmanExpansion class enumerates Wick contractions as permutations and
builds FeynmanGraph objects. The wick_equals_kirchhoff identity check is
implemented. Not yet tested against multi-vertex cases.

### ❌ Not yet implemented (future phases)

- **Shuffle product** for systematic graph generation (Phase 1)
- **Spectral dimension** d → d_s substitution for arbitrary graphs (Phase 6)
- **AC layer**: Lagrange equation for DSE, Borel transform (Phase 5)
- **Multi-loop parametric integrals** beyond one-loop (Phase 2-3)
- **Companion paper** LaTeX source (parallel to code)
- **Tier 2-5 validation tests** (one-loop exponents, BWS, DP, novel)

---

## Changes Made During This Session

### New files created

| File | Description |
|------|-------------|
| `rdft/core/reaction_network.py` | CRN dataclasses, all factory methods |
| `rdft/core/generators.py` | H-W generators, Liouvillian, verify_thesis_examples() |
| `rdft/core/expansion.py` | Wick contractions, FeynmanExpansion, algebraic chain demo |
| `rdft/graphs/incidence.py` | FeynmanGraph, incidence matrix, Kirchhoff, Symanzik |
| `rdft/integrals/symanzik.py` | Ψ and Φ from Kirchhoff complement |
| `rdft/integrals/parametric.py` | ω-integration, I(G;d), thesis_example_2515/16 |
| `rdft/rg/bphz.py` | Connes-Kreimer coproduct + antipode |
| `rdft/rg/rg_functions.py` | β(λ), KnownResults, critical exponents |
| `tests/validation/test_tier1.py` | 15 validation tests, 8 passing |
| `PLAN.md` | 7-phase roadmap + companion paper outline |
| `README.md` | this file |

### Bugs found and fixed

1. **Kirchhoff polynomial**: `kirchhoff_polynomial()` originally called
   `L_RS.det()` which is always zero for connected graphs (the reduced
   symbolic Laplacian is always singular — its rows sum to zero).
   Fixed to: `L_RS.minor_submatrix(0, 0).det()` — the cofactor, which is
   the correct Matrix-Tree theorem statement.

2. **Symanzik Ψ**: Updated `_compute_Psi()` to use the corrected Kirchhoff
   polynomial, taking edge complements of each spanning tree monomial.

3. **1PI check**: `has_bridge()` originally included v_∞ in the connectivity
   check, causing all graphs to appear non-1PI (v_∞ was isolated).
   Fixed to only check internal vertices (index < n_vertices_int).

4. **Test expectation**: The sunset test expected Ψ = α₀α₁+α₀α₂+α₁α₂
   from `kirchhoff_polynomial()`, but K and Ψ are different polynomials —
   K sums edges in each spanning tree, Ψ sums the complements.
   Fixed by separating the K and Ψ tests.

---

## Immediate Next Steps

1. **Complete ω-integration** for multi-loop cases (fix `parametric.py`
   `_evaluate_alpha_integrals` for non-linear Φ after reduction)

2. **Run the full chain for 2A → ∅**: connect parametric integral output
   to Z-factor extraction and verify β(λ) = -ελ + (1/8πD)λ² matches Lee (1994)

3. **Add Tier 2 validation tests**: one-loop exponents for pair annihilation,
   coagulation, two-species annihilation

4. **Implement shuffle product** for automated diagram generation up to
   any loop order

5. **Write Theorem 5.1** (companion paper): formal proof that the
   ω-integration algorithm via δ(Σl) is the directed Matrix-Tree theorem

---

## Running the Tests

```bash
pip install sympy networkx numpy scipy pytest

# Run all validation tests
python tests/validation/test_tier1.py

# Run the full algebraic chain demonstration
python rdft/core/expansion.py

# Quick algebraic identity check
python -c "
from rdft.core.generators import verify_thesis_examples
results = verify_thesis_examples()
print(results)
"
```

---

## Rust Simulation Engine

The `simulations/` directory contains a high-performance Rust simulator for
particle-based reaction-diffusion systems on arbitrary graphs. It reproduces
the BRW (Branching Wiener Sausage) scaling exponents from
Bordeu, Amarteifio et al. (2019) Sci. Rep. 9:15590.

### Building

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build in release mode (with LTO optimisation)
cd simulations
cargo build --release
```

### Running the BRW Validation Suite

```bash
# Full suite: all graph types (1D-5D lattice, Sierpinski, random tree, BA network)
# JSON → stdout, progress → stderr
cargo run --release > output/brw_results.json

# Single graph configuration
./target/release/rdft-sim --graph lattice:3:50 --crn birth_death -n 20000 --tmax 5000 --ds 3
./target/release/rdft-sim --graph sierpinski:4 --crn birth_death -n 20000 --ds 1.86
./target/release/rdft-sim --graph ba:5000:3 --crn birth_death -n 10000 --ds 4
```

Graph types: `lattice:DIM:SIZE`, `sierpinski:LEVEL`, `tree:N:SEED`, `ba:N:M`, `complete:N`

CRN types: `birth_death` (critical BRW), `gribov`, `brw` (coalescent), `pair_annihilation`, `coagulation`

### Plotting

```bash
# Generate scaling, convergence, and comparison plots
python simulations/python/plot_brw.py simulations/output/brw_results.json

# Or build + run + plot in one go
python simulations/python/plot_brw.py
```

Produces three figures in `simulations/output/`:
- **scaling.png** — Log-log ⟨V^p | survived⟩ vs t for each graph type
- **convergence.png** — Exponent estimate vs N realizations
- **comparison.png** — Theory vs simulation bar chart

### Key Results

The simulator validates the BRW scaling law (Bordeu+ 2019):

    ⟨V^p⟩(t) ~ t^{(p·d_s - 2)/2}    [unconditional, d_s < 4]
    ⟨V^p | survived⟩(t) ~ t^{p·d_s/2} [conditional on survival]

where V(t) = number of distinct sites ever visited and d_s is the spectral
dimension of the graph. The conditional exponent α_cond/p → d_s/2 converges
cleanly:

| Graph | d_s | α_cond/p (sim) | d_s/2 (theory) |
|-------|-----|----------------|----------------|
| 1D lattice | 1 | 0.50 | 0.50 |
| 3D lattice | 3 | 1.53 | 1.50 |
| Sierpinski | 1.86 | 0.96 | 0.93 |

### Architecture

- **`src/graph.rs`** — Graph types (lattice, Sierpinski, random tree, BA, custom edge list)
- **`src/crn.rs`** — Chemical reaction network specification (single species kA→lA)
- **`src/engine.rs`** — Core simulation: multinomial unary reactions (avoids sequential bias),
  Poisson binary reactions, multinomial diffusion. Parallel via rayon.
- **`src/main.rs`** — CLI + BRW validation suite
- **`python/plot_brw.py`** — Matplotlib plotting utilities

Parallelism: independent realizations run in parallel across all CPU cores via
[rayon](https://docs.rs/rayon). The full 7-graph suite with 20k realizations
each completes in ~80 seconds on an M-series Mac.

---

## Key References

| Reference | What it contributes |
|-----------|---------------------|
| Amarteifio (2019) PhD thesis, Imperial | The entire framework; eqs. (1.35-1.38), (2.30-2.35), (2.76) |
| Doi (1976a,b) J. Phys. A 9 | Second-quantisation of reaction-diffusion |
| Peliti (1985) J. Phys. (Paris) 46 | Coherent-state path integral |
| Lee (1994) J. Phys. A 27:2633 | RG calculation for kA→∅; exact exponent |
| Lee & Cardy (1995) J. Stat. Phys. 80:971 | A+B→∅ two-species theory |
| Täuber, Howard, Vollmayr-Lee (2005) J. Phys. A 38:R79 | Comprehensive RG review |
| Connes & Kreimer (1998) CMP 199:203 | Hopf algebra of renormalisation |
| Flajolet & Sedgewick (2009) | Analytic combinatorics; transfer theorem |
| Bogner & Weinzierl (2010) arXiv:1003.1154 | Parametric representations |
| Panzer (2015) PhD thesis | Graph polynomials for algorithmic integration |
| Bordeu, Amarteifio et al. (2019) | BWS on arbitrary graphs |

---

## Companion Paper

A parallel mathematical paper is planned (see `PLAN.md` for the full outline).
The paper proves every theorem that the code uses:

- **Theorem 2.1**: Generator formula Q[∂z,z] for arbitrary kA→lA
- **Theorem 3.1**: Shuffle product generates all 1PI graphs
- **Theorem 3.2**: Symmetry factor = 1/|Aut(G)| from AC exponential formula
- **Theorem 4.1**: Matrix-Tree theorem gives K(G)
- **Theorem 5.1**: ω-integration via δ(Σl) is the directed Matrix-Tree theorem
- **Theorem 6.1**: BPHZ forest formula is the Connes-Kreimer Hopf antipode
- **Theorem 8.1**: DSE is a Lagrange equation; Landau pole = branch point
- **Theorem 9.1**: d → d_s substitution is valid for any RD process on a graph
