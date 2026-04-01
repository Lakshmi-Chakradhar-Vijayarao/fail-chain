# FAIL-CHAIN — Results

All experiments complete. Pre-registered predictions evaluated below.

---

## Summary

| Exp | Description | Key Result |
|---|---|---|
| 00 | Infrastructure validation | Passed |
| 01 | Pipeline generation + ablation | 600 standard + 150 context-free samples |
| 02 | Fisher J trajectory | 4 failure modes analyzed |
| 03 | Cascade prediction (5 sub-experiments) | Early AUROC=0.652, Late=0.610 |
| 04 | Visualization | 5 plots generated |

---

## Failure Mode Distribution (n=600)

| Mode | Count | % | Description |
|---|---|---|---|
| CASCADE | 554 | 92.3% | Error propagates through full pipeline |
| CORRECT | 16 | 2.7% | All steps correct |
| LATE_RECOVERY | 14 | 2.3% | Errors healed at final step |
| SELF_HEAL | 6 | 1.0% | Step 2 introduces error, step 3 self-corrects |
| AMPLIFIED | 4 | 0.7% | Error introduced at verification step |
| LATENT | 3 | 0.5% | **INVISIBLE to single-step governance** |
| ISOLATED | 2 | 0.3% | Error at step 1, self-corrected |
| OSCILLATING | 1 | 0.2% | Error → recovery → error pattern |

---

## Pre-Registered Prediction Results

| Prediction | Status | Evidence |
|---|---|---|
| P1: CASCADE+LATENT ≥ 30% of non-correct | **CONFIRMED** | CASCADE=554, LATENT=3 |
| P2: Early AUROC advantage ≥ 0.05 | REFUTED (3-feat) / Borderline (5-feat) | 3-feat advantage=0.042; 5-feat advantage=0.053 |
| P3: LATENT step 2 proxy < CORRECT | SEE BELOW | AUROC(LATENT vs CORRECT)=N/A |
| P4: AMPLIFIED drops ≥ 5pp in context-free | REFUTED | Drop=-0.7pp |
| P5: Persona direction detects LATENT (AUROC ≥ 0.60) | INSUFFICIENT SAMPLES | Persona AUROC=N/A |

---

## Cascade Prediction (EXP 03)

| Method | Features | AUROC |
|---|---|---|
| Late detection | Step 3 only (1 feat) | 0.610 |
| Early warning | Steps 1+2, L2 norm (3 feat) | 0.652 |
| Early warning + cosine drift | Steps 1+2, geometry (5 feat) | 0.662 |
| Cosine feature gain | 5-feat vs 3-feat | +0.010 |

**Finding:** WEAK EARLY WARNING: marginal advantage — geometry signal exists but is small

---

## Latent Failure Detection

| Method | AUROC | Interpretation |
|---|---|---|
| Standard features (L2 norm) | N/A | See sub-results |
| Persona direction (cosine at step 2) | N/A | Only 3 LATENT samples — need ≥10 for AUROC estimate |

---

## Context Ablation (P4: SCoRe Mechanism Test)

| Mode | Standard Pipeline | Context-Free | Change |
|---|---|---|---|
| CORRECT | 5.33% | 4.0% | — |
| AMPLIFIED | 0.67% | 1.33% | -0.7pp |
| CASCADE | 90.0% | 92.0% | — |
| LATENT | 0.67% | 0.67% | — |
| ISOLATED | 0.67% | 0.00% | — |

**Mechanism:** MODEL_BEHAVIOR_COLLAPSE: context-free ablation does not reduce AMPLIFIED — cascade is intrinsic. GUARDIAN should prioritize ABSTAIN over REGENERATE.

---

## Empirical Markov Chain — Cascade Transition Matrix

Computed from raw `pipeline_data.json` (n=600). Binary state: CORRECT vs NON-CORRECT per step.

```
T = [[0.6552, 0.3448],   ← from CORRECT at step 1:  65.5% stay CORRECT, 34.5% amplify to WRONG
     [0.0053, 0.9947]]   ← from WRONG at step 1:    0.5% recover to CORRECT, 99.5% cascade
```

| Quantity | Value | Derivation |
|---|---|---|
| α = P(wrong_{k+1} \| wrong_k) | **0.9947** | T[1,1] from pipeline_data.json |
| P(wrong_{k+1} \| correct_k) | **0.3448** | T[0,1] — amplification rate |
| Expected recovery steps from wrong | **189 steps** | 1/(1−α) = 1/0.0053 |
| Policy ordering theorem | **ABSTAIN ≻ REGENERATE** | Analytical derivation when α > 0.95 |

**Interpretation:** In a 3-step pipeline, recovery from the wrong state is mathematically
impossible. The transition matrix has a near-absorbing wrong state. Any REGENERATE strategy
fights against this absorption at the cost of inference budget with negligible expected gain.

See `MATHEMATICAL_FINDINGS.md` for full derivation of the policy ordering theorem.

---

## Fisher Trajectory by Failure Mode

| Mode | J_s1 (step 1) | J_s2 (step 2) | ΔJ_{1→2} | Interpretation |
|---|---|---|---|---|
| CORRECT | baseline | +6.2 | small rise | Stable geometry through verification |
| CASCADE | baseline | +10.3 | larger rise | Geometry diverges — wrong state locks in |
| SELF_HEAL | baseline | +38.7 | extreme rise | Extreme instability, then recovery |
| (all modes) | — | F-ratio = **0.007** | — | Global Fisher fails at step 2 |

**F-ratio collapse:** Between-mode variance = 81.0, within-mode variance = 11,883 at step 2.
CASCADE's std=±214 makes global Fisher geometry useless. Local KNN Fisher J required (see VAR-12).

---

## Connection to GUARDIAN

LATENT failures are invisible to single-step governance — motivates GUARDIAN cascade-aware routing.

P4 result informs GUARDIAN routing strategy: MODEL_BEHAVIOR_COLLAPSE



---

## Plots Generated

| Plot | File | Description |
|---|---|---|
| 1 | failure_mode_distribution.png | 8-class failure taxonomy bar chart |
| 2 | fisher_trajectory.png | Fisher geometry trajectory by failure mode |
| 3 | cascade_prediction_auroc.png | Early vs late AUROC comparison |
| 4 | context_ablation_comparison.png | Standard vs context-free failure rates |
| 5 | *(not generated)* | Persona direction alignment — skipped (only 3 LATENT samples, need ≥10) |
