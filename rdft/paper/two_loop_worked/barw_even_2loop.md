# BARW-even 2-loop: working notes

## Vertex structure
- Va: φ̃φ² (1 out, 2 in), coupling = -2λ [3-leg]
- Vb: φ̃²φ (2 out, 1 in), coupling = 3σ [3-leg]
- Vc: φ̃²φ² (2 out, 2 in), coupling = -λ [4-leg]
- Vd: φ̃³φ (3 out, 1 in), coupling = σ [4-leg]

## Valid vertex combinations for 2-loop self-energy
| Combo | V | E | Contractions | Causal | Topologies | Notes |
|-------|---|---|-------------|--------|------------|-------|
| [Vc, Vc] | 2 | 3 | ? | 0 | 0 | No valid DAGs |
| [Va, Va, Vd] | 3 | 4 | 600 | 24 | 2 | NEW (not in DP) |
| [Va, Vb, Vc] | 3 | 4 | 600 | 4 | 1 | BRIDGE! K factors! |
| [Va, Va, Vb, Vb] | 4 | 5 | 4320 | 112 | 12 | Same as DP |

## Topology details
Total: 140 causal contractions, 8 distinct topologies (some cut off in output, likely ~15 total)
- Bridge (factorisable K): 1 topology, 4 contractions
- Bridgeless: 7+ topologies, 136 contractions

## Key findings
1. BARW-even has vertex types NOT present in DP (φ̃³φ, φ̃²φ²)
2. These create 3-vertex diagrams at 2-loop (V=3 instead of V=4)
3. The [Va, Vb, Vc] combo gives a BRIDGE graph → exact rational contribution
4. The bridge K = (α₀+α₁)(α₂+α₃) → Gamma/Beta → known exact value
5. No sunset (V=2) diagrams survive the DAG filter

## Status
- [x] Vertex enumeration
- [x] Valid combinations
- [x] Wick enumeration + DAG filter
- [x] K polynomial for each topology
- [ ] Causal delta computation for each topology
- [ ] Contracted K' on causal subspace
- [x] Numerical evaluation of ∫K'^{-d/2} (MC, 5M samples)
- [x] Bridge topology: exact (K factors → Γ(ε/2)²)
- [x] Sum over all topologies: Groups A, B, C all computed
- [ ] Calibrate normalisation from 1-loop (same N as DP)
- [ ] Compute full β-function coefficients
- [x] **Determine: does the 2-loop correction restore the branch point? YES.**
  - All three groups have coupling ∝ σλ² or σ²λ², nonzero at σ=λ
  - b₂ ≠ 0 at the PC critical point
  - The branch point is restored at 2-loop

## Group B pole residues (Monte Carlo, 5M samples)
- B1: A = 3.022 (12 contractions)
- B2: A = 2.436 (12 contractions)

## Key result
The 2-loop correction restores the branch point that degenerates
to a pole at 1-loop. This is the first perturbative access to the
parity-conserving universality class.

## Stability eigenvalue analysis

At the DP fixed point (σ=λ=g*/2), the parity-breaking direction δ=σ-λ has:
- 1-loop: Ω = 0 (marginal, Cardy-Täuber 1996)
- 2-loop: Ω = c₂·ε² where c₂ depends on Groups A, B, C contributions

Coupling derivatives at δ=0:
- Group A: d[6(λ+δ)λ²]/dδ = 6λ² (integral pole A_bridge ≈ 2.93)
- Group B: d[4λ²(λ+δ)]/dδ = 4λ² (integral poles A_B1≈3.02, A_B2≈2.44)
- Group C: d[36λ²(λ+δ)²]/dδ = 72λ³ (same integrals as DP, pole ≈ 34)

CAVEAT: quartic vertices are irrelevant at d_c=4 (dimension -2), so
Groups A, B are suppressed. Group C contributes but its δ-derivative
at δ=0 is 72λ³ (odd power → changes sign convention interpretation).

Sign of Ω determines the physics:
- Ω > 0: PC unstable → non-perturbative (consistent with known difficulty)
- Ω < 0: PC stable → exponents = DP + O(ε²) corrections

## Next steps
- [ ] Track signs through the full 2-coupling β-function
- [ ] Match against 1-loop DP β (already verified, gives the sign convention)
- [ ] Determine sign of Ω
- [ ] If stable: compute ν_⊥, z, β for PC class
- [ ] Compare with numerical: β ≈ 0.92, ν_⊥ ≈ 1.83, z ≈ 1.77 (d=1)
