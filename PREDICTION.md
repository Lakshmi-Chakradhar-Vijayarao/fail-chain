# FAIL-CHAIN — Pre-Registered Predictions

**Date filed:** 2026-03-28
**Committed before:** any pipeline generation or experiment execution
**Author:** Lakshmi Chakradhar Vijayarao

---

## Experiment Overview

Self-refinement pipeline (3 steps) on Qwen 2.5 3B, TruthfulQA dataset, n=600 samples.
Fisher separability proxy measured at L32 (89% depth) — optimal layer confirmed in HaRP and GEOM-PROOF.

---

## Pre-Registered Predictions

### P1 — Failure Mode Distribution
Prediction: CASCADE and LATENT failures together will account for ≥ 30% of all non-correct samples.

Rationale: Self-refinement amplifies errors because step 2 and step 3 are conditioned on
prior steps. A model that hallucinated at step 1 will use that hallucination as context
for steps 2 and 3, increasing cascade probability.

Refutation condition: CASCADE + LATENT < 20% of non-correct samples.

### P2 — Early Warning Signal
Prediction: Fisher proxy features from steps 1+2 will predict step 3 failure with
AUROC ≥ 0.60, exceeding late detection (step 3 features only) by ≥ 0.05.

Rationale: Fisher geometry at L32 encodes factual retrieval state. A model that
will hallucinate in step 3 is likely already in a low-separability region at step 2,
before the final answer is generated.

Refutation condition: AUROC(early) - AUROC(late) < 0.02.
Refutation interpretation: Latent failures are geometry-invisible — cascade risk
cannot be detected before the final step. This directly motivates formal multi-step
bounds in GUARDIAN.

### P3 — Latent Failure Detectability
Prediction: Latent failures (intermediate steps appear correct, final step fails)
will show lower mean Fisher proxy at step 2 than CORRECT samples, despite passing
ROUGE-L threshold at step 2.

Rationale: The model's representation at step 2 may be near the decision boundary
even when the ROUGE-L label calls it correct. Fisher proxy captures geometric
instability that surface labels miss.

Refutation condition: Mean Fisher proxy at step 2 for LATENT samples ≥ mean for
CORRECT samples. Refutation interpretation: Latent failures are indistinguishable
from correct samples at the representation level — governance cannot prevent them
without architectural changes.

---

## Amendment — 2026-03-28 (pre-experiment, same session)

Two additional predictions added after related-work analysis of Anthropic and
Google DeepMind 2024-2025 research. Added before Exp 01 GPU run. The original
P1–P3 are unchanged. P4 and P5 are added as separate experiment conditions.

### P4 — Context Contamination vs Model Behavior Collapse
Prediction: In a context-free ablation (Step 2 does NOT receive Step 1 output),
the AMPLIFIED failure rate will drop by ≥ 5 percentage points relative to the
standard pipeline.

Rationale: SCoRe (Kumar et al., Google DeepMind, NeurIPS 2024) shows that
self-correction failure is driven by distribution mismatch — the model cannot
escape wrong context from step 1. If AMPLIFIED drops in the context-free
condition, this confirms the context-contamination mechanism.

Refutation condition: AMPLIFIED drop < 2 percentage points.
Refutation interpretation: Cascade is a model behavior collapse intrinsic to
the model, not context-driven. GUARDIAN should prioritize ABSTAIN over
context-stripped REGENERATE.

### P5 — Eventual-Failure Direction Detects Latent Failures at Step 2
Prediction: A linear direction extracted from step 1 hidden states, trained on
step 3 outcome labels (y_s3), will predict LATENT failure class from step 2
hidden state alone with AUROC ≥ 0.60 (LATENT vs CORRECT, boundary samples excluded).

Design note: The direction is trained on y_s3 (did step 3 eventually fail?), NOT
y_s1 (was step 1 wrong?). LATENT samples have correct step 1 outputs by definition,
so a direction trained on y_s1 could never distinguish them from CORRECT samples.
Training on y_s3 surfaces the geometric signature of eventual failure already
present in h_s1, before any text output has gone wrong.

Rationale: Following the persona vector paradigm (Anthropic, arXiv:2507.21509),
hallucination propensity encodes as a linear direction. LATENT failures show correct
text at steps 1 and 2, but their h_s1 geometry may already encode the eventual
failure — the representation is drifting toward the wrong attractor before any
text-level signal appears. Boundary samples (ROUGE within ±0.05 of threshold)
are excluded from this analysis as their class labels are unreliable.

Refutation condition: Direction AUROC < 0.55 on clean (non-boundary) samples.
Refutation interpretation: Eventual failure is not linearly encoded in step-1
geometry — geometric signature only manifests at step 3 and cannot be detected
from earlier representations.

---

## Commitment

These predictions are fixed as of the commit timestamp on this file.
Results will be reported honestly in RESULTS.md regardless of whether they
confirm or refute these predictions. Negative results are documented as findings,
not hidden as failures.

The scientific value of this work depends on this commitment being real.
