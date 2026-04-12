# Perturbative Ontological Corrections

## Idea

The SharedM model provides a "classical orbit" — a mean-field prediction from
structurally shared coupling matrices (A, B). The perturbative correction adds
small, context-dependent deviations on top of this orbital path:

```
x = orbital_output + gate(x) * correction(x)
```

- **gate(x)**: scalar sigmoid, reads the residual stream. Starts near 0 (bias
  init at -3), opens as context matures during training.
- **correction(x)**: small MLP (d → d_corr → d), output-initialized to zero.
- Corrections act on the value/FFN stream only — coupling (M) remains untouched.

The physics analogy: the orbital is the zeroth-order solution; the perturbation
is a first-order correction that accounts for boundary conditions (context).

## Experiments

### Exp 8 — Matched-Param Comparison (running)

**Status: In progress.** SharedM and Perturbative phases complete (up to step 4000).
Transformer baseline still training. Full results with ablation will follow.

Three models at matched ~10M params on TinyStories (5M tokens, 5000 steps):

| Model | d | L | Params | Notes |
|-------|---|---|--------|-------|
| SharedM | 168 | 5 | 9,961,728 | Orbital baseline |
| Perturbative | 172 | 4 | 10,028,376 | **Lost 1 layer** to fit corrections at matched params |
| Transformer | 176 | 3 | ~10,006,656 | Standard baseline (pending) |

#### Training curves (val loss)

| Step | SharedM | Perturbative | Delta | Gate | δ/h | Attn H |
|------|---------|-------------|-------|------|-----|--------|
| 1 | 10.878 | 10.846 | -0.032 | 0.051 | 0.337 | 4.558 |
| 500 | 4.276 | 4.258 | -0.018 | 0.247 | 0.800 | 4.434 |
| 1000 | 3.539 | 3.579 | +0.040 | 0.262 | 0.628 | 3.696 |
| 1500 | 3.253 | 3.269 | +0.016 | 0.306 | 0.595 | 3.480 |
| 2000 | 3.057 | 3.073 | +0.016 | 0.337 | 0.580 | 3.362 |
| 2500 | 2.975 | 2.974 | -0.001 | 0.348 | 0.571 | 3.300 |
| 3000 | 2.891 | 2.891 | +0.000 | 0.359 | 0.570 | 3.264 |
| 3500 | 2.845 | 2.843 | -0.002 | 0.363 | 0.572 | 3.240 |
| 4000 | 2.797 | 2.787 | -0.010 | 0.360 | 0.567 | 3.233 |

SharedM FINAL: **2.7636**. Perturbative still running (step 4000 so far).

#### Interpretation so far

1. **The perturbative model recovered from a structural handicap.** It was forced
   to sacrifice a layer (L=4 vs L=5) to match params. Despite losing an entire
   layer of orbital propagation, it converged to the same loss — and by step 4000,
   is slightly ahead (-0.010 nats).

2. **The crossover story is clean.** Early training (steps 1-500): perturbative
   was marginally better (the corrections hadn't activated yet, but the wider d=172
   helped). Mid training (1000-2000): perturbative fell behind as the missing 5th
   layer hurt. Late training (2500+): the corrections caught up and the gap closed
   to zero, then turned slightly negative (perturbative winning).

3. **Gate dynamics confirm the design.** The gate opened from 0.05 → 0.36 over
   training. This is the intended behavior: corrections start dormant and activate
   as the model learns when context is informative. The gate has plateaued around
   0.36, suggesting the model found its correction regime.

4. **δ/h ratio stayed moderate.** After an initial overshoot (0.80 at step 500),
   the perturbation-to-orbital ratio settled at ~0.57. This is larger than "small"
   in a strict perturbative sense, but the gating keeps the effective contribution
   at 0.36 × 0.57 ≈ 0.20 of the orbital norm — a genuine correction rather than
   a replacement.

5. **Attention entropy dropped steadily** (4.56 → 3.23), meaning attention patterns
   sharpened over training. The gate opened as entropy fell — consistent with the
   prediction that corrections become useful when the model has learned to attend
   to specific context rather than spreading attention uniformly.

#### The depth-vs-params confound

The key issue with exp8: matching total params forced the perturbative model to
trade a layer for correction parameters. This conflates two effects:
- Does the correction architecture help? (the question we want to answer)
- Does losing a layer hurt? (an artifact of the experimental design)

The fact that perturbative-at-L=4 matched SharedM-at-L=5 is arguably *more*
impressive than it looks — it recovered a full layer's worth of capacity through
structured corrections. But it's not a clean test.

### Exp 8b — Matched-Depth Comparison (ready to run)

**Status: Script ready.** Will run after exp8 completes.

This fixes the confound. The perturbative model uses the **same backbone** as
SharedM (same d=168, same L=5) and adds corrections as extra parameters:

| Model | d | L | Params | Overhead |
|-------|---|---|--------|----------|
| SharedM | 168 | 5 | 9,955,680 | baseline |
| Perturbative | 168 | 5 | ~10,066,000 | +1.1% |
| Transformer | 168 | 5 | ~10,181,000 | +2.3% |

The question becomes: do ~110K params of structured, gated corrections outperform
~110K params of additional learned coupling (Transformer's per-layer Q, K)?

This is the cleaner comparison and the one that will determine whether scaling up
is worth pursuing.

```bash
# Run after exp8 finishes:
cd /Users/icey-sirsh/code/sip-sim && uv run python3 experiments/perturbative/exp8b_matched_depth.py
```

## Decision framework for scaling

After exp8 + exp8b complete, the go/no-go for a larger run depends on:

| Signal | Go | Wait | No-go |
|--------|-------|------|-------|
| Exp8b vs SharedM | Perturbative wins by >0.02 nats | Within 0.02 nats | Perturbative loses |
| Ablation (late positions) | Clear improvement at pos 128+ | Marginal | No effect or hurts |
| Gate behavior | Continues opening, doesn't saturate | Plateaus early | Collapses to 0 |
| δ/h ratio | Stays < 0.5 (perturbative regime) | 0.5-1.0 (borderline) | > 1.0 (not perturbative) |

If exp8b shows a clear win, the natural next step is a 50-100M param run on
a larger dataset slice, checking whether the correction overhead stays small
(~1%) while the performance gain grows with scale.

## Files

```
experiments/perturbative/
├── README.md                           # This file
├── exp8_perturbative.py                # Matched-param comparison (3 models)
├── exp8b_matched_depth.py              # Matched-depth follow-up (perturbative only)
└── results/
    └── exp8_training_log.txt           # Raw training output (partial, updating)
```

Results from the full runs (JSON metrics, diagnostic plots, checkpoints) will be
saved to the perturbative_package results directory by the scripts and copied here
after completion.
