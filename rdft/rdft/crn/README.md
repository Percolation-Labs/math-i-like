# `rdft.crn` — CRN to RG, in one auditable pipeline

A clean, auditable API for going from a chemical reaction network (CRN) to its
full RG programme. This package replaces the scattered `rdft/ac/gribov/*.py`
and `letters/.../poc_*.py` scripts with one unified surface.

> **Status: WIP. Read [`critique.md`](critique.md) before trusting it for
> publication.** The Reggeon-DP / Gribov pipeline runs end-to-end and matches
> JT05 Eq. (60) with zero residual. Other CRNs run through the combinatorial
> layer (Doi shift, Lagrange, φ-tree |Aut|, diagram enumeration) cleanly, but
> the kinematic-kernel derivation from a fresh propagator is being closed
> incrementally — at 1-loop today, 2-loop is a follow-up.

The line we draw:

* **Combinatorial / topological (ours, mechanical):** Doi shift, Lagrange
  counts, φ-tree symmetry factors, Wick + cumulant algebra, ω-contour graph
  contraction, Feynman shift, the closed-form spatial loop integral,
  ε-expansion + simple-pole extraction, Z-extraction derivatives at the
  sub-point, IBP to master basis, Hopf antipode, BPHZ, Täuber, Wilson–Fisher.
* **Physical ansatz (yours, gap (a)):** propagator, subtraction-point values,
  Z-factor extraction operators, coupling Z-exponents.
* **Bridge integrals (yours, gap (b)):** *values* of the master integrals at
  the sub-point.

If a result requires you to do something that's not in (a) or (b), that's a
bug in our factorisation; please file it.

## Quick start

```python
from rdft.crn import CRN, RGProgram

# Define the CRN once
crn = CRN.reggeon_dp()                 # or CRN.dyadic_brw(), CRN.brw_thesis(), ...

# Run the full RG programme
rg = RGProgram(crn, loop_order=2)
rg.run()

# Inspect anything
print(rg.audit())                       # full per-step ledger
rg.zfactors["psi"].display()            # Z-factor with provenance
rg.exponents.compare_to_jt05()          # {"eta":True,"z":True,"nu":True,"beta_DP":True}
```

## The chain

```
CRN
 │
 │  Doi shift  (Layer 1: vertex dictionary, phi(G))
 ▼
phi(G)
 │
 │  Lagrange inversion  (Layer 2: counts and phi-tree |Aut|)
 ▼
diagrams (Diagram objects with .lineage)
 │
 │  Hopf antipode  (Layer 2: closed-form Z^(2,2))
 │  IBP closure    (Layer 2: Z^(2,1) via 12 q^X_Gamma rationals)
 ▼
Z-factors
 │
 │  Tauber relation + beta(u*)=0 + exponents
 ▼
{eta, z, nu, beta_DP}    -- vs JT05 Eq.(60), zero residual.
```

Every node in this chain produces objects (`Diagram`, `ZFactor`, `Exponents`)
that carry a `Provenance` chain explaining where each field came from.

## Module map

| Module | What it does |
|--------|--------------|
| `crn.py`         | `CRN`, `Reaction`, `Vertex`. Doi shift. `phi(G)` extraction. Builders for Reggeon DP / dyadic BRW / phi^4-DP / thesis BRW. |
| `diagram.py`     | `Diagram` and `Provenance` dataclasses. `Diagram.explain()` prints a per-graph audit. |
| `enumerator.py`  | `enumerate_phi_trees(n)` (plane binary trees, Catalan-style); `enumerate_bubbles(vertices)` (V=2, multi-species); `enumerate_tadpoles(vertices)` (V=1 self-loop); `classify_reggeon_topology(shape)`. |
| `symmetry.py`    | `aut_phi_tree(shape)` returning $2^{k(T)}$. `aut_bubble(...)` for the directed-graph route. `cross_check_bubble_aut(...)` to assert agreement. |
| `legendre.py`    | `legendre_reggeon_dp(N_g, J_max)` returns `LegendreResult` with `.W_coef(a,b,n)` and `.Gamma_coef(a,b,n)` projection methods. |
| `rg.py`          | `RGProgram`: ties everything together. Each step records `Provenance` on `self.history`. |
| `audit.py`       | `format_audit(rg)` walks the history and returns a multi-line ledger. |
| `viz.py`         | `render_diagram_grid(diagrams, out_path)` renders a `Diagram` list to a PDF panel with figure + s(G) + relevance + lineage notes. |

## What each result carries

Every `Diagram` has `.lineage`, a list of `Provenance` entries:

```python
>>> d = diagram_from_phi_tree("(L,(L,(L,L)))")
>>> for p in d.lineage:
...     print(p)
[Layer 1] Doi shift -> phi(G): Reggeon DP: phi(G) = 1+G^2 (cubic interactions)
[Layer 2] Lagrange inversion: shape '(L,(L,(L,L)))' appears in plane phi-tree
          enumeration at size 7
[Layer 2] phi-tree |Aut| via 2^k: k(T) = 1 symmetric internal node(s)
          => |Aut| = 2^1 = 2
[Layer 3] 1PI verdict from shape: 1PI: no single propagator cut disconnects
```

Same for `ZFactor.provenance` and the entries in `RGProgram.history`.

## Auditing the pipeline

```
$ python -c "from rdft.crn import CRN, RGProgram; \
              print(RGProgram(CRN.reggeon_dp(), loop_order=2).run().audit())"
```

prints a ledger that flags every step as one of:

* AC-derived (Lagrange / Hopf antipode / IBP / phi-tree |Aut|)
* structural rule on shape strings (1PI cut rule, etc.)
* QFT bookkeeping (external-leg sectors)
* external input (master integral values from JT05 / Panzer / Borinsky)
* derived (Tauber relation, exponents)

with the residuals against JT05 Eq.~(60) reported as zero for all four exponents.

## Where the legacy code went

| Old script | Replaced by |
|------------|-------------|
| `rdft/ac/gribov/two_loop.py`         | `RGProgram.final_tauber_and_exponents()` |
| `rdft/ac/gribov/actrick.py`           | `RGProgram.layer2_hopf_antipode_double_poles()` |
| `rdft/ac/gribov/ibp_coefficients.py`  | `RGProgram.layer2_ibp_simple_poles()` |
| `rdft/ac/gribov/assembly.py`          | `RGProgram.layer1_doi_shift()` + `.layer2_lagrange_counts()` |
| `letters/.../poc_legendre.py`         | wrapper around `rdft.crn.legendre.legendre_reggeon_dp` |
| `letters/.../poc_brw.py`              | wrapper around `rdft.crn.CRN.brw_thesis()` + `enumerate_bubbles` |
| `letters/.../poc_topologies.py`       | wrapper around `enumerate_phi_trees` + `legendre_reggeon_dp` + draw helpers |
| `letters/.../brw_figures.py`          | wrapper around `rdft.crn.viz.draw_corolla` / `draw_bubble` |

The old scripts in `rdft/ac/gribov/` still work and are kept for the
`run_all_tests.py` harness; the new API in `rdft/crn/` is the maintained
surface going forward.

## Defining a new CRN

Three routes:

```python
# 1. From explicit reactions (Doi shift produces vertex dictionary)
crn = CRN.from_reactions(
    name="Annihilation",
    species=("A",),
    reactions=(
        Reaction(reactants=(("A", 2),), products=(), rate=sp.Symbol("lambda")),
    ),
)

# 2. From a hand-specified vertex set (skips Doi shift)
crn = CRN.from_vertices(
    name="My theory",
    species=("A", "B"),
    vertices=(Vertex(name="V1", in_legs=(("A",1),), out_legs=(("A",2),), sign=-1),),
)

# 3. Use a pre-built model
crn = CRN.reggeon_dp()        # the Janssen-Tauber CRN
crn = CRN.dyadic_brw()        # pure A -> 2A
crn = CRN.brw_thesis()        # the 7-vertex BRW set from thesis Eqs. 3.16+3.21
crn = CRN.phi4_doi_peliti()   # 2A -> 0
```

Then `RGProgram(crn).run()` is the full pipeline; what fails is what's not yet
mechanical for that CRN. Power-counting filters live at `crn.interaction_vertices(max_legs=...)`.
