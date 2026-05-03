# Letter: Gunnar Pruessner, 29 Apr 2026 — messy diagrammatics

This folder contains the response to Gunnar's note of 29 Apr 2026, the
companion tutorial walking the four-layer chain from CRN to 1PI, and the
proof-of-concept scripts that produce the figures.

All scripts are now thin wrappers around the [`rdft.crn`](../../rdft/crn/README.md)
package — the unified API that ties the Doi shift, Lagrange counts, phi-tree
symmetry factors, Hopf antipode, IBP closure, and exponents together with a
provenance ledger.

## Contents

| File | What it is |
|------|------------|
| `note.tex` / `note.pdf`                             | Gunnar's original note + commentary. |
| `Q1.tex` / `Q1.pdf`                                 | Reply to Q1: the $H$-vertex ladder $T(z) = (z_1{-}z_2)(w_1{-}w_2)$, eigenvalue $2$. |
| `Q2.tex` / `Q2.pdf`                                 | Reply to Q2: the JT05 / Gribov RG end-to-end (Eqs. 57-60). |
| `legendre_tutorial.tex` / `legendre_tutorial.pdf`   | Companion tutorial: four-layer chain CRN -> phi(G) -> diagrams -> Z-factors. |
| `letter_notes.tex` / `letter_notes.pdf`             | Short-form notes (CRN -> generator -> diagrams). |
| `poc_legendre.py`                                   | Wrapper around `rdft.crn.legendre.legendre_reggeon_dp`. Verifies $W=-7608$, $\Gamma=-504$. |
| `poc_brw.py`                                        | Wrapper around `CRN.brw_thesis()` + `enumerate_bubbles`. Verifies $7$ three-point bubbles + $s(G)=2$. |
| `poc_topologies.py`                                 | Wrapper around `enumerate_phi_trees(7)` + `legendre_reggeon_dp` + drawing. Produces `poc_topologies.pdf`. |
| `brw_figures.py`                                    | Wrapper around `rdft.crn.viz.draw_corolla` / `draw_bubble`. Produces `brw_corollas.pdf`, `brw_bubbles.pdf`. |
| `verify_2x2.py`, `eq58_from_z.py`, `poc_select.py`  | Q1 / Q2 numerical sanity checks (kept independent of `rdft.crn` for self-containment). |
| `*.pdf`                                             | Compiled outputs: tutorials + figures. |

## Reproducing the figures

```bash
cd letters/2026-04-29_gunnar_messy_diagrammatics
PYTHONPATH=../../ python3 poc_topologies.py    # poc_topologies.pdf
PYTHONPATH=../../ python3 brw_figures.py       # brw_corollas.pdf, brw_bubbles.pdf
PYTHONPATH=../../ python3 poc_legendre.py      # numerical checks (W, Gamma)
PYTHONPATH=../../ python3 poc_brw.py           # numerical checks (bubbles, s(G))

# Build the tutorial
tectonic legendre_tutorial.tex
```

## How to read the diagrams

Every figure produced by the wrapper scripts now carries:

- the topology label (e.g. *ladder*, *box*, *ice-cream*, *$\Sigma_1$ on $\psi$-leg*);
- the relevance verdict (1PI / reducible) with the cut-rule reason;
- the symmetry factor $s(G)$ with the rule that produced it
  ("$\phi$-tree shape $(L,L)$, $k=1$, $|\mathrm{Aut}|=2^k=2$"); and
- a directed-graph $|\mathrm{Aut}|$ cross-check (BRW figures only).

The lineage is also accessible programmatically: each `Diagram.lineage` is a
list of `Provenance` entries that name the layer (1, 2, 3, 4) and the rule
that fired. Use `print(diagram.explain())` to dump the per-graph audit to
stdout.
