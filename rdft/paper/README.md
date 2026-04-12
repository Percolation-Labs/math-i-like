# Papers

Promoted papers (tracked in git).

## `main.tex` — Reaction-Diffusion Field Theory via Analytic Combinatorics
The broader AC pipeline: stoichiometry → DSE → critical exponents
for arbitrary reaction-diffusion on arbitrary graphs.
PDF: `main.pdf`.

## `worked-example.tex` — Worked example companion piece
Complete AC derivation of the BRW scaling exponents with extensive
appendices (AC tutorial, Lagrange equations, SIR example,
BARW-even, branch points, scaling corrections).
PDF: `worked-example.pdf`.

## `two_loop_worked/` — BARW-even 2-loop calculation

## `cfac/` — Coupled Field Analytic Combinatorics
Extension of the AC framework to coupled particle-field systems
(Doi-Peliti + Martin-Siggia-Rose). Contains:
- `cfac_paper.tex` — Factoring the counting problem.
  One-loop exactness for Keller-Segel, branch-gap instanton
  theorem, three-loop Kirchhoff factorisation. Applications to
  amyloid fibril nucleation and ice crystal formation.
- `cfac_theorem.tex` — Standalone theorem companion.
- `reproduce.py` — Reproduces all numerical results.
- `figures/` — Generated plots.

Run: `cd paper/cfac && python reproduce.py` (~1-2 min).

## `wip/` — gitignored drafts
Precursor documents that fed into the promoted papers. Not under
version control.
