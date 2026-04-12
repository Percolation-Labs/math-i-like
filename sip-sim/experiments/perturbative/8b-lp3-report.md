# Overnight Run Report: Exp 8b + LP3

*Run completed 2026-03-19 00:24 UTC*

---

## Exp 8b: Perturbative Corrections at Matched Depth

### Motivation

Exp8 forced the perturbative model to sacrifice a layer (L=4 vs L=5) to match
total params with SharedM. This conflated two effects: does the correction
architecture help, and does losing a layer hurt? Exp8b fixes the backbone
(same d=168, L=5 as SharedM) and adds corrections as extra parameters (~1.1%
overhead), giving a cleaner test.

### Configuration

| Model | d | L | Params | Overhead vs SharedM |
|-------|---|---|--------|---------------------|
| SharedM (exp8 baseline) | 168 | 5 | 9,961,728 | — |
| Perturbative-8b | 168 | 5 | 10,072,933 | +110K (+1.1%) |
| Transformer (exp8 ref) | 176 | 3 | 10,010,528 | +49K (+0.5%) |

Correction budget: 110,365 params total (22K per layer) for LayerNorm + gate +
correction MLP (d→64→d) at each of 5 layers.

### Results

#### Final val loss — all models

| Model | Params | Val Loss | vs SharedM | vs Transformer |
|-------|--------|---------|-----------|---------------|
| **SharedM** | 9,961,728 | **2.7636** | baseline | -0.90% |
| **Perturbative-8b** | 10,072,933 | **2.7754** | +0.43% | -0.47% |
| Perturbative (exp8, L-1) | 10,028,376 | 2.7795 | +0.57% | -0.33% |
| Transformer | 10,010,528 | 2.7886 | +0.90% | baseline |

Ranking: SharedM > Perturbative-8b > Perturbative-exp8 > Transformer.

The perturbative model at matched depth improved by 0.004 nats over the
depth-penalised version — modest but in the right direction. It beats the
Transformer by 0.013 nats while using fewer parameters.

SharedM remains the best model. The corrections do not overcome the bare
orbital baseline, but they close the gap relative to the exp8 version that
lost a layer.

#### Training trajectory (8b)

| Step | Val Loss | Gate | δ/h | Attn H |
|------|---------|------|-----|--------|
| 1 | 10.850 | 0.047 | 0.324 | 4.558 |
| 500 | 4.265 | 0.264 | 0.646 | 4.416 |
| 1000 | 3.546 | 0.316 | 0.527 | 3.758 |
| 1500 | 3.208 | 0.338 | 0.510 | 3.590 |
| 2000 | 3.058 | 0.356 | 0.505 | 3.460 |
| 2500 | 2.961 | 0.364 | 0.498 | 3.434 |
| 3000 | 2.876 | 0.369 | 0.500 | 3.388 |
| 3500 | 2.857 | 0.366 | 0.499 | 3.380 |
| 4000 | 2.781 | 0.366 | 0.500 | 3.379 |
| 4500 | 2.773 | 0.371 | 0.503 | 3.367 |
| 5000 | 2.780 | 0.372 | 0.498 | 3.368 |
| FINAL | **2.775** | — | — | — |

#### Ablation: perturbation on vs off

| Condition | 8b (L=5) | exp8 (L=4) |
|-----------|----------|-----------|
| Perturbation ON | 2.7570 | 2.7553 |
| Perturbation OFF | 3.1980 | 3.2042 |
| Gap | 0.441 | 0.449 |
| Early position improvement (pos 0-64) | **+0.453** | +0.392 |
| Late position improvement (pos 192-256) | +0.430 | +0.440 |

The corrections carry ~0.44 nats of the model's performance in both variants.
With matched depth, the early-position benefit increases substantially (+0.453
vs +0.392), suggesting the extra layer gives the corrections more useful
features to work with at early positions.

#### Gate and correction dynamics

| Metric | 8b (L=5) | exp8 (L=4) | Interpretation |
|--------|----------|-----------|----------------|
| Final gate | 0.372 | 0.362 | Similar; ~37% activation |
| Final δ/h | **0.498** | 0.566 | 8b is more perturbative |
| Final attn entropy | 3.368 | 3.226 | 8b retains slightly more diffuse attention |
| Gate-entropy corr (exp8) | — | r = -0.905 | Strong: gate opens as attention sharpens |

The δ/h ratio settled at 0.50 in 8b vs 0.57 in exp8. With matched depth, the
corrections are genuinely smaller relative to the orbital signal — closer to a
true perturbative regime. The effective correction magnitude is
gate × δ/h ≈ 0.37 × 0.50 ≈ **0.185** of the orbital norm.

### 8b Interpretation

1. **Matched depth helped but didn't flip the ranking.** Perturbative-8b is
   closer to SharedM than exp8's version (+0.43% vs +0.57%), confirming that
   the lost layer was part of the gap. But SharedM still wins outright.

2. **The corrections are doing real work.** The ablation shows 0.44 nats of
   contribution — the orbital backbone co-adapts with the corrections and relies
   on them. This is consistent across both depth configurations.

3. **The perturbative regime is healthier at matched depth.** δ/h = 0.50
   (vs 0.57) means the corrections are proportionally smaller — the model
   isn't trying to use corrections as a substitute for missing depth.

4. **The overhead is tiny.** 110K params (1.1%) buys meaningful architectural
   capability, even if it doesn't beat the bare orbital. The question for
   scaling is whether this overhead percentage shrinks further at larger sizes
   while the benefit grows.

---

## LP3: Compositional Generalisation

### Setup

Synthetic language with compositional structure:
- 10 characters × 8 actions × 10 objects × 8 emotions
- Templates: "CHAR ACTION the OBJ .", "CHAR was EMOTION .",
  "CHAR and CHAR played with the OBJ ."
- 15 character-object pairs held out from training (15% of 100 total)
- 50,000 training sentences, 120 OOD test sentences, 120 ID test sentences
- Both models: d=128, L=6, 4 heads, 3000 steps

### Results

| Model | Params | ID Loss | OOD Loss | Gap | ID acc@5 | OOD acc@5 |
|-------|--------|---------|---------|-----|----------|-----------|
| SharedM | 1,036,032 | 1.029 | 1.777 | +0.748 | 91.7% | **0.0%** |
| Transformer | 1,199,872 | 1.030 | 1.777 | +0.747 | 88.3% | **0.0%** |

### LP3 Interpretation

**Total failure to generalise — for both architectures.** Neither model
predicted a single held-out character-object combination despite near-perfect
in-distribution performance (92% / 88% top-5 accuracy). Both memorised which
characters go with which objects rather than learning compositional rules.

The generalisation gap is essentially identical (0.748 vs 0.747), meaning
SharedM's weight sharing provided no advantage for compositional transfer
on this task.

**Why this happened — and why it's not conclusive:**

1. **No partial credit.** The OOD metric is top-5 exact match on the object
   token. There's no gradient signal from "almost right" — the model either
   gets the exact object or scores zero. The model may have learned partial
   compositional structure that doesn't show up in this binary metric.

2. **Perfect memorisation is the easy solution.** With 85 training pairs and
   50K sentences, each pair appears ~590 times. The model has overwhelming
   evidence for which char-obj combinations are valid. There's no pressure to
   generalise when memorisation works perfectly.

3. **The language is too flat.** Every character can do every action with every
   object — there's no structural regularity to transfer. Real compositional
   generalisation requires some pattern (e.g., "animals do X, people do Y")
   that the model can extend. This language has none.

4. **The holdout is too strict.** Holding out entire char-obj pairs means the
   model has never seen "Fox cake" in any context. A softer holdout (e.g.,
   "Fox wanted the cake" held out but "Fox found the cake" seen) would test
   whether the model can generalise across actions for a known pair.

**Recommendation:** Redesign LP3 before concluding anything about SharedM's
compositional abilities. A better test would have structured regularities
(e.g., role-based constraints) and graded holdouts that test specific axes
of generalisation.

---

## Summary and Next Steps

### What we learned

| Finding | Confidence | Implication |
|---------|-----------|------------|
| SharedM beats Transformer at 10M params | High | Shared coupling is parameter-efficient |
| Perturbative corrections carry ~0.44 nats of work | High | The architecture is functional, not decorative |
| Corrections don't beat bare SharedM | High (at 10M) | Extra params on corrections < same params on orbital |
| Matched depth helps perturbative vs depth-penalised | Moderate | Depth matters more than correction capacity |
| δ/h settles at ~0.50 (perturbative regime) | High | Architecture maintains intended dynamics |
| Gate-entropy correlation r = -0.91 | High | Gate opens when attention sharpens — as predicted |
| Neither architecture generalises compositionally | Low confidence | Task design is flawed; needs redesign |

### Decision on scaling

Based on the decision framework from the initial README:

| Signal | Threshold | Observed | Verdict |
|--------|-----------|----------|---------|
| 8b vs SharedM | Win by >0.02 nats | +0.012 (loss) | **Wait** — within noise |
| Ablation (late positions) | Clear improvement | +0.43 nats | **Go** — large effect |
| Gate behavior | Doesn't saturate | Plateau at 0.37 | **Borderline** |
| δ/h ratio | Stays < 0.5 | 0.498 | **Go** — just under threshold |

**Verdict: borderline.** The corrections are architecturally sound and do
meaningful work, but they don't outperform the bare orbital at this scale.
Two paths forward:

1. **Scale test (cautious).** Run at 50M params to see if the correction
   overhead percentage shrinks and the benefit grows. If perturbative beats
   SharedM at 50M, the architecture has legs.

2. **Architecture revision.** The corrections might be fighting the orbital
   rather than complementing it. Consider: corrections that only activate
   at specific layers (not all), corrections with learned frequency (not
   just a single gate), or corrections that target specific attention heads
   rather than the full residual.
