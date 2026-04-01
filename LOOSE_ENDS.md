# FAIL-CHAIN — Loose Ends for Future Research

This document records all deliberate experimental loose ends in FAIL-CHAIN.
Each loose end is a tractable extension that was identified but not implemented
in the current paper scope. They are preserved here so that GUARDIAN (Project 5)
and future PhD-level work can pick them up precisely.

Each entry has: a tag (for grep), the current finding it builds on, the exact
implementation path, and the estimated impact for publication positioning.

---

## How to Use This Document

- Tags are `[VAR-N]` matching the 10 experimental variations identified in the
  research planning phase.
- Each entry cites the paper that motivates it.
- `[GUARDIAN]` suffix = direct input to GUARDIAN's architecture design.
- `[PHD-YEAR-2]` suffix = scope for PhD Year 2 formal work.

---

## VAR-1: Thinking Token Windows (o1/o3 Extension)
**Tag:** `[VAR-1]`
**Current finding:** FAIL-CHAIN's 3-step pipeline uses explicit prompt-level steps.
Modern reasoning models (o1, o3, DeepSeek-R1) have implicit "thinking" steps
encoded in token windows before producing the final answer.

**What to do:**
1. Run the same ROUGE-L labeling on o1-mini or o3-mini outputs using the OpenAI API.
2. Extract hidden states at the boundary between thinking tokens and output tokens.
3. Apply `compute_trajectory()` treating each thinking-chunk boundary as a "step."
4. The failure taxonomy should transfer — LATENT failures in thinking token windows
   are particularly important: the model thinks correctly but outputs wrong.

**Why it matters:** Anthropic (2025) "Reasoning Models Don't Always Say What They
Think" showed CoT is omitted 59-81% of the time when unfaithful. LATENT failures
in thinking windows would be undetectable by CoT monitoring — only geometry catches them.

**Estimated effort:** 2 days (API calls + hidden state extraction from o3-mini)
**Publication impact:** Extends FAIL-CHAIN from open-source models to frontier models.
Positions work for NeurIPS 2027 or ICLR 2027. MUST cite arXiv:2025 Anthropic CoT paper.

---

## VAR-2A: Verbal vs Geometric Confidence Divergence
**Tag:** `[VAR-2A]`
**Current finding:** FAIL-CHAIN uses ROUGE-L as ground truth. The model's verbal
confidence (P(True) logit from "Is this correct? Yes/No") is not measured.

**What to do:**
1. After each step generation, append a binary confidence probe:
   "Is the above answer correct? Answer Yes or No." → extract logit of "Yes".
2. Compare per-step: verbal confidence vs Fisher proxy.
3. Key hypothesis: LATENT failures will show HIGH verbal confidence AND LOW
   Fisher proxy drift at step 2. This is the geometry-vs-language divergence.
4. Plot as a 2D scatter: Fisher proxy (x) vs P(True) (y), colored by failure mode.

**Why it matters:** Kadavath et al. 2022 (Anthropic) showed models "mostly know
what they know" via P(True). FAIL-CHAIN tests whether geometric uncertainty
(Fisher proxy) provides strictly more information than verbal self-knowledge.

**Estimated effort:** 0.5 days (add confidence probe to pipeline.py, re-run Exp 01)
**Warning:** Adds ~30% to Exp 01 GPU time. Run on a subset (n=200) if needed.
**Publication impact:** Creates a 2×2 table: (High/Low Fisher) × (High/Low P(True)).
The (Low Fisher, High P(True)) cell = LATENT failures with fake confidence.

---

## VAR-2B: Context-Free Ablation → GUARDIAN Routing (ALREADY IMPLEMENTED)
**Tag:** `[VAR-2B]` `[GUARDIAN]`
**Status:** Implemented in Exp 01 (context-free ablation) and Exp 03 (P4 test).
**What's left:** Use the P4 result to set GUARDIAN's routing logic:
- If P4 CONFIRMED: GUARDIAN's REGENERATE path should strip step 1 from step 2 context.
- If P4 REFUTED: GUARDIAN should prioritize ABSTAIN for AMPLIFIED-class samples.
  Context-stripping cannot fix a model behavior collapse.

**Implementation in GUARDIAN:** In `guardian/src/router.py`, add condition:
```python
if failure_mode == "AMPLIFIED" and P4_CONFIRMED:
    return Route.REGENERATE_NOCTX   # context-stripped regeneration
elif failure_mode == "AMPLIFIED":
    return Route.ABSTAIN            # intrinsic collapse, cannot route away
```

---

## VAR-3: Persona Direction → GUARDIAN Real-Time Monitor (ALREADY IMPLEMENTED)
**Tag:** `[VAR-3]` `[GUARDIAN]`
**Status:** Complete. Direction saved to `results/logs/hallucination_direction.npy` (shape: 2048, unit-normalized; LR on h_s1 → label_s3, all 600 samples).
**What's left for GUARDIAN:**
1. ~~Save direction vector~~ ✅ Done — `results/logs/hallucination_direction.npy`
2. In GUARDIAN's inference loop, compute:
   `cosine_alarm = h_s2 @ direction` at each step 2 inference call.
3. If `cosine_alarm > threshold_persona` → flag as potential LATENT failure
   before step 3 runs. Trigger REGENERATE or increased monitoring.
4. This is a <1ms overhead at inference time (3072-dim dot product).

**Why it's powerful:** Unlike FAISS KNN (GUARDIAN's primary mechanism), the
persona direction is a single pre-computed vector — zero latency overhead.
It complements FAISS: persona direction catches LATENT, FAISS catches CASCADE.

---

## VAR-4: Representation Steering at Cascade-Risk Steps
**Tag:** `[VAR-4]` `[PHD-YEAR-2]`
**Current finding:** FAIL-CHAIN identifies when cascade is about to happen
(at step 2, before step 3 runs). But it doesn't intervene — it only predicts.

**What to do:**
1. When Fisher proxy at step 2 exceeds cascade-risk threshold, apply activation
   steering: project h_s2 toward the "CORRECT" centroid and away from the
   "CASCADE" centroid. Use the direction from `compute_fisher_j()`.
2. Measure: does steered step 2 reduce the CASCADE rate at step 3?
3. This connects to GEOM-PROOF's steering experiments (HaRP Exp 22).

**Why it matters:** Connects mechanistic interpretability (read the geometry) to
mechanistic intervention (write the geometry). This is the full loop of
"uncertainty-based governance." First paper reads; second paper writes.

**Estimated effort:** 3 days (requires hook-based activation patching during generation)
**Warning:** Activation steering during autoregressive generation requires patching
the residual stream at L32 on every forward pass. Use `model.model.layers[32]` hook.
**Publication impact:** MAJOR — would be a standalone NeurIPS paper. Defer to PhD Year 2.

---

## VAR-5: Speculative Decoding as Early Exit Governance
**Tag:** `[VAR-5]` `[PHD-YEAR-2]`
**Idea:** Instead of running all 3 pipeline steps, use the Fisher proxy at step 1
as an "accept/reject" signal for speculative execution. If Fisher proxy at step 1
is above a threshold (high separability = confident answer), skip steps 2 and 3.

**Why it matters:** Reduces inference cost by 60-70% for CORRECT samples while
maintaining accuracy. This is a computational efficiency framing of the FAIL-CHAIN
geometry signal — "when can we stop early?"

**What to do:**
1. From your existing data, find the Fisher proxy threshold at step 1 that separates
   CORRECT from all other classes with precision ≥ 0.90.
2. Apply this threshold to your 100-sample holdout (saved in Exp 01).
3. Measure: step 1 accept rate, step 3 error rate for accepted samples.

**Note:** This is a pure CPU experiment — no new GPU runs needed.

---

## VAR-6: Beyond Fisher J for LATENT Detection (PARTIALLY IMPLEMENTED)
**Tag:** `[VAR-6]`
**Status:** Cosine similarity (h_s1, h_s2) added to `build_rich_feature_matrix()`.
Persona direction detection added in Exp 03.

**What's still left:**
1. **LID (Local Intrinsic Dimensionality):** For each sample, compute the local
   intrinsic dimensionality of h_s2's neighborhood using k-NN distances.
   LATENT failures hypothesis: h_s2 for LATENT samples lives in a lower-dimensional
   subspace than CORRECT (the representation is "collapsing" to a wrong attractor).
   ```python
   # LID estimator (Hill estimator)
   from sklearn.neighbors import NearestNeighbors
   def lid(h, X_pool, k=20):
       nn = NearestNeighbors(n_neighbors=k+1).fit(X_pool)
       dists, _ = nn.kneighbors([h])
       dists = dists[0][1:]  # exclude self
       return -k / np.sum(np.log(dists[:-1] / dists[-1] + 1e-10))
   ```
2. **Spectral rank of (h_s1, h_s2) pair:** Compute rank of the 2-column matrix
   [h_s1, h_s2] — if they are nearly identical (rank ≈ 1), the model is not
   updating its representation through verification. Low rank = ISOLATED or CORRECT.
   High rank + low ROUGE = LATENT signature.

**Estimated effort:** 1 day (pure numpy/scipy, CPU only)

---

## VAR-7: Sycophancy as 4th Pipeline Step
**Tag:** `[VAR-7]`
**Idea:** Add a Step 4 to the pipeline where the user "corrects" the model with
a wrong answer: "Actually, the answer is [WRONG ANSWER]. Revise your response."
Measure whether the model revises its step 3 output toward the wrong answer.

**Key question:** Are CASCADE-mode samples MORE susceptible to sycophancy pressure
than CORRECT samples? If yes, sycophancy is not a separate failure — it's a
terminal amplification of cascade risk.

**Why it matters:** Connects FAIL-CHAIN's cascade taxonomy to the governance problem
of adversarial users. GUARDIAN must detect both intrinsic cascade AND induced cascade.

**Estimated effort:** 1 day (add Step 4 prompt, measure ROUGE delta)
**Publication impact:** Medium — bridges cascade failure and sycophancy literatures.
Cite Anthropic 2025 sycophancy papers if published by submission deadline.

---

## VAR-8: Multi-Agent Handoff Failure (GUARDIAN Preview)
**Tag:** `[VAR-8]` `[GUARDIAN]`
**Idea:** Replace the 3-step single-agent pipeline with a 2-agent handoff:
- Agent A (Qwen 2.5 3B) generates the initial answer.
- Agent B (different model or same model, fresh context) verifies.
**Question:** Does multi-agent verification reduce cascade rates compared to
single-agent self-verification?

**Why it matters:** This is a minimal prototype of GUARDIAN's multi-agent routing.
If multi-agent reduces CASCADE but not LATENT, it confirms that LATENT is an
intrinsic model property, not a context-contamination artifact.

**Estimated effort:** 2-3 days (requires running two inference sessions per sample)
**Warning:** Doubles GPU time. Run on n=200 subset only.
**Note:** This is essentially the GUARDIAN paper's pilot study. Consider running
this as GUARDIAN Exp 00 (infrastructure) rather than FAIL-CHAIN Exp 06.

---

## VAR-9: Pipeline Depth Scaling
**Tag:** `[VAR-9]` `[PHD-YEAR-2]`
**Idea:** Does the failure taxonomy change with pipeline depth?
- 3-step: FAIL-CHAIN (current)
- 5-step: add Step 4 (re-verify) and Step 5 (final refinement)
- 7-step: iterative refinement until convergence

**Key question:** Do LATENT failures become detectable at deeper pipelines,
or do they compound? Is there a "detection horizon" for cascade?

**Estimated effort:** 2 days (modify pipeline.py for N steps, re-run Exp 01 on n=200)
**Publication impact:** HIGH — scaling law for failure mode rates vs pipeline depth.
Would produce a publication-quality figure alone.

---

## VAR-10: Geometry of Self-Knowledge (Metacognition via Fisher)
**Tag:** `[VAR-10]` `[PHD-YEAR-2]`
**Idea:** Define local Fisher separability at each step as a continuous
"metacognition signal" — how well can the model distinguish correct from wrong
at each step, just from its own hidden state?

**Implementation:**
1. For each step s ∈ {1, 2, 3}, compute Fisher J between:
   - h_s from CORRECT samples vs h_s from non-CORRECT samples.
2. Plot J_s1 vs J_s2 vs J_s3 — this is the "self-knowledge trajectory."
3. Hypothesis: J_s2 > J_s1 means verification improved self-knowledge (healthy).
   J_s2 < J_s1 means verification degraded separability (cascade risk).

**Why it matters:** This connects FAIL-CHAIN to Kadavath et al. 2022 ("Models Know
What They Know") and the GEOM-PROOF bound — local Fisher J is a moment-by-moment
certificate of self-knowledge. This is the unifying theory for the full 5-project arc.

**Estimated effort:** 0.5 days (pure CPU, uses compute_fisher_j() already implemented)
**Note:** This might be the most theoretically important experiment in FAIL-CHAIN.
It gives a clean narrative: "Fisher J is both a detection signal AND a metacognition
proxy — the model's geometry tells us whether it knows what it knows."

---

## Theoretical Limitations — Documented for Paper Honesty

### LIMIT-1: Causation vs Correlation in CASCADE
FAIL-CHAIN observes that when step 1 fails, steps 2 and 3 also fail. This is
labelled "CASCADE" (propagation). But it could equally be "the question is hard,
so all three independent generations fail." To confirm causation:
- Counterfactual needed: run step 3 with a corrected step 2 injection. If rouge_s3
  recovers, propagation is confirmed. This is tracked in VAR-2B (context ablation
  partially tests this for AMPLIFIED, not CASCADE).
- In the paper: state "we observe co-failure (CASCADE) which is consistent with
  propagation but does not rule out independent difficulty effects."

### LIMIT-2: L32 Prompt-Length Confound
Step 1 prompts are ~30 tokens; step 3 prompts are ~200 tokens. The L32 last-token
hidden state encodes different positional contexts at each step. The Fisher magnitude
trajectory may partially reflect changing prompt length, not changing factual state.
- Mitigation: in Exp 02, compare Fisher proxy at L32 vs earlier layers (L16, L24)
  to check whether the trajectory pattern is layer-specific or prompt-length-driven.

### LIMIT-3: Single Model on Adversarial Benchmark
All results are from Qwen 2.5 3B on TruthfulQA (designed to elicit hallucinations).
Cascade rates may be inflated vs neutral QA datasets.
- Mitigation: 100-sample holdout on NaturalQuestions (VAR-9 loose end).

---

## VAR-11: Spectral Rank as Zero-Cost Cascade Detector
**Tag:** `[VAR-11]`
**Status:** Identified. Computable from existing data — no GPU needed.
**Current finding:** CASCADE samples have hidden state std=±214 at step 2, suggesting the representation is not coherently updating through verification.

**What to do:**
For each sample i, compute `rank([h_s1[i], h_s2[i]].T)` — the rank of the 2-column matrix formed by stacking step-1 and step-2 hidden states.
- If rank ≈ 1: h_s2 ≈ c·h_s1 (same direction, just scaled) → model didn't update representation through verification → CORRECT or ISOLATED.
- If rank = 2: h_s2 is genuinely different from h_s1 → model updated representation → may signal CASCADE or LATENT.

**Data needed:** `results/hidden_states/pipeline_hidden_states.npz` (arrays h_s1, h_s2, shape 600×2048 each)
**Expected finding:** CASCADE rank > CORRECT rank at step 2. If true, rank is a zero-AUROC-overhead cascade alarm requiring only 1 SVD per sample.
**Estimated effort:** 0.5 days (pure numpy — `np.linalg.matrix_rank(np.column_stack([h_s1[i], h_s2[i]]))`)

---

## VAR-12: Local KNN Fisher J at Step 2 (Fix for F-ratio Collapse)
**Tag:** `[VAR-12]` `[GUARDIAN]`
**Status:** Identified. Computable from existing data — no GPU needed.
**Current finding:** Global Fisher F-ratio at step 2 = 0.007 — CASCADE's distribution is so non-Gaussian (std=±214) that global geometry tools fail completely.

**What to do:**
For each step-2 hidden state h_s2[i], compute Fisher J in the 50-NN neighborhood rather than globally:
1. Find 50 nearest neighbors of h_s2[i] in the full h_s2 pool (FAISS index on h_s2).
2. Split neighbors by label (CORRECT vs CASCADE within the neighborhood).
3. Compute local Fisher J = (μ_c − μ_cascade)² / (σ²_c + σ²_cascade) within the neighborhood.
4. Compare local J between CORRECT and CASCADE samples.

**Expected finding:** CORRECT samples have higher local J at step 2 than CASCADE samples — the geometry is locally coherent for correct responses even when globally indistinguishable. This is the proof-of-concept for GUARDIAN's local Fisher J routing.
**Estimated effort:** 1 day (FAISS for KNN, then Fisher J per neighborhood)
**Publication impact:** Validates the GUARDIAN design constraint: local J succeeds where global J fails.

---

## VAR-13: Intra-CASCADE Sub-Clustering
**Tag:** `[VAR-13]`
**Status:** Identified. Computable from existing data — no GPU needed.
**Current finding:** CASCADE (n=554) has step-2 std=±214, suggesting multiple distinct geometric sub-types within the class. This within-CASCADE variance is why F-ratio = 0.007 — the denominator (within-class variance) is inflated by mixing.

**What to do:**
Run K-means with k=3 on CASCADE's h_s2 vectors (554×2048 matrix). Expected sub-types:
- Sub-type A: consistent cascade (wrong from step 1, confirmed wrong at step 2 geometry)
- Sub-type B: near-LATENT cascade (h_s2 appears geometrically correct, but step 3 is wrong — this is the undetected LATENT subpopulation)
- Sub-type C: AMPLIFIED cascade (h_s1 was geometrically correct, h_s2 drifted into wrong space)

**Data needed:** `results/hidden_states/pipeline_hidden_states.npz` — filter by CASCADE label from `pipeline_data.json`
**Expected finding:** 3 geometric subtypes with distinct Fisher J trajectories, explaining the F-ratio collapse as a mixing artifact.
**Estimated effort:** 0.5 days (sklearn KMeans on 554×2048, then Fisher J per cluster)

---

## VAR-14: Lyapunov Threshold ΔJ* for Cascade Prediction
**Tag:** `[VAR-14]` `[GUARDIAN]`
**Status:** Identified. Computable from existing data — no GPU needed.
**Current finding (from MATHEMATICAL_FINDINGS.md):**
- CORRECT: ΔJ_{1→2} = +6.2 (small rise)
- CASCADE: ΔJ_{1→2} = +10.3 (larger rise)
- SELF_HEAL: ΔJ_{1→2} = +38.7 (extreme rise, then full recovery)

**Hypothesis:** ΔJ* ≈ +15 separates stable from unstable pipelines. Samples with ΔJ_{1→2} > 15 are at cascade risk regardless of step-2 output quality.

**What to do:**
1. Compute Fisher J at step 1 and step 2 for all 600 samples (using `compute_fisher_j()` from `src/cascade_predict.py`).
2. Compute ΔJ_{1→2} = J_s2 − J_s1 for all 600 samples.
3. Compute AUROC of ΔJ_{1→2} for predicting label_s3 (final cascade outcome).
4. Plot ROC curve and identify threshold ΔJ* that maximizes Youden's J.

**Expected finding:** ΔJ* ≈ 10–20 with AUROC ≥ 0.65 for cascade prediction. This becomes GUARDIAN's lightweight routing signal (cost: 2 Fisher J computations per query, no FAISS needed).
**Estimated effort:** 0.5 days (5-line computation + AUROC from sklearn)

---

## VAR-15: Cascade Absorption Coefficient α per Question Difficulty Quartile
**Tag:** `[VAR-15]`
**Status:** Identified. Computable from existing `pipeline_data.json` metadata — no GPU needed.
**Current finding:** α = 0.9947 on TruthfulQA (adversarial dataset). This may be dataset-specific.

**Hypothesis:** α decreases for easier questions. If α is a function of question difficulty (measurable via step-1 ROUGE score), then GUARDIAN can estimate per-query α from the step-1 output — and apply the policy ordering theorem (ABSTAIN ≻ REGENERATE only when α > 0.95) adaptively.

**What to do:**
1. From `pipeline_data.json`, extract rouge_s1 as a proxy for question difficulty (low rouge_s1 = hard question).
2. Split into quartiles: Q1 (easiest, rouge_s1 top 25%) through Q4 (hardest, rouge_s1 bottom 25%).
3. Compute α = P(wrong_s2 | wrong_s1) separately for each quartile.
4. Plot α vs difficulty quartile — expect α decreasing from Q4 (hardest) to Q1 (easiest).

**Expected finding:** α_Q4 ≈ 0.99 (near-absorbing for hard questions) vs α_Q1 ≈ 0.85 (some recovery for easy questions). This gives GUARDIAN a difficulty-conditional routing policy.
**Estimated effort:** 0.5 days (pure pandas/numpy on pipeline_data.json)

---

## Connection Summary: Loose Ends → GUARDIAN

| Loose End | GUARDIAN Component |
|---|---|
| VAR-2B (context ablation result) | Router.route() logic for AMPLIFIED class |
| VAR-3 (persona direction vector) | Step 2 real-time monitor (dot product alarm) |
| VAR-8 (multi-agent handoff) | GUARDIAN Exp 00 pilot design |
| VAR-6 LID (if implemented) | FAISS neighborhood enrichment for LATENT detection |
| VAR-10 (self-knowledge J_s) | GUARDIAN's adaptive threshold = local Fisher J |
| VAR-12 (local KNN Fisher J) | Proof-of-concept for GUARDIAN's local J routing (vs failed global J) |
| VAR-14 (Lyapunov ΔJ*) | Lightweight pre-FAISS routing signal (2 Fisher J computations) |
| VAR-15 (α per difficulty quartile) | Difficulty-conditional policy ordering for adaptive ABSTAIN threshold |

The persona direction vector (`results/logs/hallucination_direction.npy`) and
the ablation mechanism finding (`context_ablation.mechanism` in cascade_prediction.json)
are the two FAIL-CHAIN outputs that directly parameterize GUARDIAN's design.
Read these before designing GUARDIAN's `src/router.py`.
