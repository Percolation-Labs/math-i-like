# Letter: Gunnar Pruessner, 1 May 2026 — RG of DP, exponents without explicit loops

Reply to Gunnar's note of 1 May 2026, "RG of DP." His thesis: the Q2 framing
("exponents are CFAC outputs") was misleading. He derives the full 1-loop DP
exponents in two pages to make the point: yes, exponents *without explicit loops*
(only one master integral is computed); no, *not from combinatorics alone*
(the master must be evaluated).

## Contents

| File | What it is |
|------|------------|
| `note.tex` / `note.pdf`               | Gunnar's note transcribed: 1-loop bubble + triangle, $Z$-system, $\beta$, exponents. |
| `reply.tex` / `reply.pdf`             | Concession + reframing. The 1-loop calculation in CFAC vocabulary: one master, four projections; $-\partial_r$ = IBP. |
| `hermite_basis.tex` / `hermite_basis.pdf` | Concept sketch in response to Gunnar's aside: $\varphi^4$ Doi-Peliti in a Hermite basis via multivariate Lagrange. Worked $N\le 2$ truncation, build path. |

## Headline of the reply

- **Concede:** "exponents from combinatorics" was wrong. The honest claim is *one loop integral, computed once; four exponents read off via Lagrange + IBP.*
- **Map his calculation to CFAC vocabulary:** "two diagrams to consider" = Lagrange on $\phi(G)=1+G^2$; "$-\partial_r$ doubles a propagator" = IBP closure; "four numbers $\{2,1,\tfrac12,4\}$" = four projections of the single master $\mathcal{I}_{-O-}$.
- **Action item:** edit `paper/cfac/gribov.tex` headline + Q2 conclusion to reflect "$1+1=2$, not $0$."
- **Real next target:** the Hermite-basis aside, where the diagram set is genuinely infinite — see `hermite_basis.tex`.

## Reproducing

```bash
cd letters/2026-05-01_gunnar_dp_one_loop
tectonic note.tex
tectonic reply.tex
tectonic hermite_basis.tex
```
