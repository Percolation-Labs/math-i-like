# Reproducing results from the CFAC paper

## `keller_segel_reproduce.py`

Runs all numerical experiments from the paper and generates the figures:

```bash
python paper/keller_segel_reproduce.py
```

Produces in `paper/figures/`:

| File | What it shows | Paper ref |
|------|--------------|-----------|
| `mfpt_vs_V.pdf` | Gillespie MFPT vs branch-gap prediction `C·exp(V·S)` | Fig. for Theorem 1 |
| `branch_structure.pdf` | Three branches of the cubic DSE `x(z)` for two systems | §4 |
| `barrier_vs_coupling.pdf` | Barrier `S` vs coupling `λ` for protein & cloud systems | Applications |
| `oneloop_deltaD.pdf` | Numerical quadrature of `δD_A` vs exact formula | Appendix A |

Prints to stdout:

1. **Branch-gap identity** on 7 parameter sets (ratio `S_gap/S_WKB = 1.000000`)
2. **MFPT scaling**: Gillespie vs analytic prediction
3. **One-loop δD_A**: numerical quadrature vs theory, 5-digit match

Runtime: ~1-2 minutes.

## Key functions

- `branch_gap_S(a, lam, delta, d)`: nucleation barrier from polynomial algebra
- `S_exact(a, lam, delta, d, x1, x2)`: WKB quasi-potential (comparison)
- `gillespie_mfpt(...)`: direct stochastic simulation

## Extending

To apply to a different coupled system:
1. Write down the effective rates `w+(x) = a + λx²`, `w-(x) = δx + dx³`
2. Identify `λ = g·μ/κ` from the microscopic coupling
3. Call `branch_gap_S(a, λ, δ, d)` → `S`
4. Mean first-passage time: `τ ≈ C · exp(V·S)`

Change the reaction network → new polynomial → new `S`. No simulation needed.
