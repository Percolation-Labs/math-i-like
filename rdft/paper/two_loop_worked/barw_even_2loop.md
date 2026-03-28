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
- [ ] Numerical evaluation of ∫K'^{-d/2}
- [ ] Bridge topology: exact evaluation
- [ ] Sum over all topologies
- [ ] Calibrate normalisation from 1-loop
- [ ] Compare with... (no published result exists!)
- [ ] Determine: does the 2-loop correction restore the branch point?
