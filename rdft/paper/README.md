# Papers

Promoted papers (tracked in git).

## `cfac_paper.tex` — Factoring the Counting Problem
Coupled DP-MSR theory for particle-field systems; proves one-loop
exactness for Keller-Segel, the branch-gap instanton theorem, and
three-loop Kirchhoff factorisation. Applications to amyloid fibril
nucleation and ice crystal formation.
PDF: `cfac_paper.pdf` (9 pages).

## `main.tex` — Reaction-Diffusion Field Theory via Analytic Combinatorics
The broader AC pipeline: stoichiometry → DSE → critical exponents
for arbitrary reaction-diffusion on arbitrary graphs.
PDF: `main.pdf`.

## `worked-example.tex` — Worked example companion piece.

## `reproduce.py` — Reproduces all numerical results from `cfac_paper.tex`
Generates figures in `figures/`:
- `mfpt_vs_V.pdf` — Gillespie MFPT vs branch-gap prediction
- `branch_structure.pdf` — Cubic DSE branches
- `barrier_vs_coupling.pdf` — Barrier S vs coupling λ
- `oneloop_deltaD.pdf` — Numerical vs analytical one-loop integral

Run: `python paper/reproduce.py` (~1-2 min).

## `wip/` — gitignored drafts
Precursor documents that fed into the promoted paper. Not under
version control.
