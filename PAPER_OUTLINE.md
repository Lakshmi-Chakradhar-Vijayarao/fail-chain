# FAIL-CHAIN — Paper Outline

**Target:** NeurIPS 2026 Trustworthy ML Workshop / ICLR 2027 main track
**Working title:** "Failure Propagation Geometry in Self-Refinement Pipelines:
A Fisher-Information Analysis of Cascade, Latent, and Amplified Hallucination"

**Authors:** Lakshmi Chakradhar Vijayarao
**Arxiv preprint target:** Immediately after GUARDIAN Exp 01 completes

---

## Abstract (draft, fill after results)

Self-refinement pipelines — where language models iteratively verify and correct
their own outputs — are increasingly deployed in agentic AI systems. We demonstrate
that failure in these pipelines is not binary but exhibits seven geometrically
distinguishable modes, including LATENT failures that are invisible to
single-step governance mechanisms. Using Fisher separability geometry at layer 32
of Qwen 2.5 3B over 600 TruthfulQA samples, we show that: (1) cascade-risk failures
constitute **95.4%** of non-correct samples (CASCADE=92.3%, LATENT=0.5%), confirming
that self-refinement is not self-correcting for a substantial fraction of inputs;
(2) early-step hidden-state geometry predicts final-step failure with AUROC **0.662**
(5-feature: L2 norm + cosine drift) vs **0.610** for late detection (advantage **+0.053**),
providing a marginal but consistent early-warning signal before the final answer is
generated; (3) LATENT failures — where intermediate outputs pass correctness thresholds
but the final output fails — could not be evaluated due to insufficient LATENT samples
(n=3 of 600); a hallucination direction extracted from step-1 activations (AUROC **N/A**,
need ≥10 LATENT samples). A context-free ablation **refutes** that AMPLIFIED failures
are context-contamination artifacts: AMPLIFIED rate changes only -0.7pp (threshold ≥5pp),
indicating model behavior collapse intrinsic to the model, not context pollution. These findings directly inform cascade-aware governance
architectures for agentic AI pipelines.

---

## Section 1: Introduction (~1.5 pages)

**Opening hook:** Self-refinement pipelines are the backbone of modern agentic AI —
yet we lack a systematic understanding of how failures propagate through them.

**Key claims to establish:**
1. Self-refinement does not self-correct reliably (cite Huang et al. 2023, SCoRe 2024)
2. Existing work treats failure as binary — correct/incorrect at the final step
3. We propose a geometric multi-step failure taxonomy
4. LATENT failures are the critical gap: invisible to current governance

**Contributions (bullet form):**
- An 8-class failure taxonomy for 3-step self-refinement pipelines
- Fisher geometry as an unsupervised early-warning signal (vs supervised probes)
- Empirical evaluation of 5 pre-registered predictions (1 confirmed, 2 refuted, 2 insufficient samples)
- A context ablation establishing MODEL_BEHAVIOR_COLLAPSE as the dominant failure mechanism
- A hallucination direction vector extracted from step-1 activations (saved for GUARDIAN; could not be evaluated for LATENT detection due to n=3 LATENT samples)

**Related work paragraph positioning:**
- vs Self-Refine (Madaan, NeurIPS 2023): we study when it fails, not optimize it
- vs SCoRe (Kumar, NeurIPS 2024): we detect the collapse geometrically, not fix it
- vs Hidden States as Early Signals (arXiv:2601.09093): we use UNSUPERVISED geometry,
  detect novel failure modes (LATENT) not seen in supervised training
- vs MAST (arXiv:2503.13657): we provide the representation-level detection mechanism
  for MAST's "inter-agent communication failures" cluster
- vs Semantic Uncertainty (Kuhn, ICLR 2023): output-level; we are representation-level

---

## Section 2: Background (~0.75 pages)

**2.1 Fisher Separability Geometry**
- Define J = δᵀΣ⁻¹δ
- Connection to GEOM-PROOF (cite as companion work): J relates to AUROC via Φ(√J/2)
- Fisher proxy = L2 norm at L32 (89% depth); justify with HaRP and GEOM-PROOF findings
- Cite arXiv:2506.15830 (Fisher Information Metric for LLMs, 2025)

**2.2 Self-Refinement Pipelines**
- 3-step structure: Initial Answer → Self-Verify → Final Answer
- Why this structure: directly models ReAct (Yao 2022), Self-Refine (Madaan 2023)
- TruthfulQA as evaluation benchmark: contains known hallucination traps

**2.3 Failure Mode Taxonomy**
- Define all 8 modes with formal ROUGE-L criteria
- Table 1: taxonomy definition table
- Emphasis: LATENT is the critical novel class — introduce it formally here

---

## Section 3: Method (~1 page)

**3.1 Pipeline Setup**
- Qwen 2.5 3B Instruct, L32 hidden state extraction
- TruthfulQA n=600 + n=150 context-free ablation
- ROUGE-L threshold 0.3 for binary correctness labels
- Pre-registration of P1-P5 before any experiment (cite PREDICTION.md commit timestamp)

**3.2 Feature Engineering**
- Standard 3-feature matrix: [j_s1, j_s2, Δ_{s1→s2}]
- Rich 5-feature matrix: adds cosine(h_s1, h_s2) and L2(h_s2 - h_s1)
- Persona direction extraction: logistic regression on h_s1 → direction vector d
- LATENT detection: cosine(h_s2, d) as continuous alarm signal

**3.3 Evaluation Protocol**
- 5-fold stratified cross-validation for all AUROC estimates
- Context-free ablation: matched n=150 subset, same questions, different prompts
- Pre-registered prediction checking against PREDICTION.md

---

## Section 4: Results (~2.5 pages)

**4.1 Failure Mode Distribution (P1)**
Figure 1A — bar chart
- Report distribution across all 8 modes (n=600)
- Cascade-risk percentage: CASCADE(92.3%) + LATENT(0.5%) + AMPLIFIED(0.7%) + OSCILLATING(0.2%) = **93.7%**
- P1 result: **CONFIRMED** — (CASCADE+LATENT)/non-correct = 95.4% >> 30% threshold
- Unexpected finding: CASCADE dominates at 92.3%; minority modes (LATENT, AMPLIFIED, etc.) are too rare for per-mode statistics

**4.2 Geometric Trajectory Analysis**
Figure 1B — trajectory plot
- Per-mode Fisher proxy trajectory through steps 1→2→3
- Analyzed modes (n ≥ 5): CORRECT (16), CASCADE (554), SELF_HEAL (6), LATE_RECOVERY (14)
- **Note:** LATENT trajectory could not be plotted — only 3 samples (below 5-sample minimum). The narrative claim that LATENT has "CORRECT-like geometry until step 3" is a theoretical prediction, NOT confirmed empirically in this run.
- Observable finding: CASCADE shows higher step1→step2 Fisher proxy slope (+10.3) than CORRECT (+6.2), suggesting detectable early geometric divergence

**4.3 Early Warning vs Late Detection (P2)**
Figure 1C — AUROC bar chart
- Late detection (step 3 only, 1 feat): AUROC = **0.610**
- Early warning (steps 1+2, 3 feat L2 norm): AUROC = **0.652**, advantage = **+0.042**
- Early warning + cosine drift (5 feat): AUROC = **0.662**, advantage = **+0.053**
- Cosine feature gain (5-feat vs 3-feat): **+0.010**
- P2 result: **REFUTED** on pre-registered 3-feat threshold (0.042 < 0.05); borderline on 5-feat (0.053 ≥ 0.05)
- Interpretation: Geometry signal exists but is weak — CASCADE dominance may suppress signal; minority failure modes are too rare to separate

**4.4 LATENT Failure Detection (P3, P5)**
Figure 1D — LATENT detection panel (shows "Insufficient samples" message)
- P3 and P5 both **INSUFFICIENT** — only 3 LATENT samples, need ≥10 for AUROC estimate
- Persona direction extracted from h_s1 (LR on all 600 samples, step-3 label) and saved to `results/logs/hallucination_direction.npy` for GUARDIAN
- Direction quality (step-3 AUROC on all samples): 0.648 — confirms h_s1 carries final-step failure signal
- Cannot evaluate the core claim about LATENT geometric resemblance to CORRECT — flagged as open question for VAR-6 (larger n)

**4.5 Context Ablation: Mechanism of AMPLIFIED Failures (P4)**
Figure 2 — context ablation bar chart (n=150 per condition)
- AMPLIFIED: standard=0.67%, context-free=1.33% → **+0.67pp increase** (opposite of predicted direction)
- CASCADE: standard=90.0%, context-free=92.0%
- LATENT: standard=0.67%, context-free=0.67% (stable)
- ISOLATED: standard=0.67%, context-free=0.00%
- P4 result: **REFUTED** — AMPLIFIED does not drop in context-free condition
- Mechanism: **MODEL_BEHAVIOR_COLLAPSE** — cascade failures are intrinsic to the model, not caused by context pollution from step 1
- Implication for GUARDIAN: REGENERATE (context-stripping) will not reduce AMPLIFIED; use ABSTAIN instead

---

## Section 5: Implications for Pipeline Governance (~0.75 pages)

**5.1 Taxonomy-Conditioned Routing**
Based on empirical findings, propose the following governance rules:
| Failure Class | Detection Signal | Governance Action |
|---|---|---|
| CASCADE | High Fisher proxy slope step 1→2 | REGENERATE (restart from step 1) |
| LATENT | Persona direction alignment at step 2 | REGENERATE + escalate to ABSTAIN |
| AMPLIFIED (if P4 CONFIRMED) | High proxy at step 2 vs step 1 | REGENERATE with context-stripped step 2 |
| AMPLIFIED (if P4 REFUTED) | High proxy at step 2 | ABSTAIN (routing cannot fix this) |
| OSCILLATING | Non-monotonic proxy trajectory | ABSTAIN |

**5.2 Connection to GUARDIAN**
- FAIL-CHAIN provides the *what* (failure taxonomy + detection signals)
- GUARDIAN implements the *how* (adaptive routing + formal bounds)
- Two FAIL-CHAIN outputs that directly parameterize GUARDIAN:
  1. Persona direction vector → GUARDIAN's step-2 real-time alarm
  2. P4 mechanism result → GUARDIAN's REGENERATE path design

---

## Section 6: Discussion (~0.5 pages)

**Limitations:**
- Single model (Qwen 2.5 3B) — see VAR-1 (thinking token extension) for frontier models
- ROUGE-L as ground truth has known limitations — exact string matching misses
  semantically correct paraphrases
- n=600 on TruthfulQA — factoid QA, not open-ended reasoning

**Relation to reasoning model failures:**
- Anthropic (2025) showed CoT is unfaithful 59-81% of the time
- FAIL-CHAIN's LATENT mode is the pipeline-level analog: intermediate *outputs*
  appear correct but final output fails — the same phenomenon at the output level
  instead of the CoT level

**Negative results:**
- P2 REFUTED (3-feat): cascade failures have only marginal early geometric warning — motivates GUARDIAN's need for formal uncertainty bounds rather than relying on soft proxy signals
- P4 REFUTED: context-stripping does not reduce AMPLIFIED failures, confirming MODEL_BEHAVIOR_COLLAPSE; GUARDIAN must ABSTAIN rather than REGENERATE for this class
- P3/P5 INSUFFICIENT: LATENT failure geometry remains an open empirical question; LATENT rarity (0.5%) on TruthfulQA factoid QA may reflect dataset characteristics rather than model behavior — see VAR-6 (open-ended reasoning tasks)

---

## Section 7: Conclusion (~0.25 pages)

- Introduce the 8-class taxonomy and pre-registered results
- Core claim: Fisher geometry provides unsupervised early warning that supervised
  probes cannot provide for novel failure modes
- LATENT failures are the key governance challenge for agentic AI pipelines
- Open problems: thinking token extension, steering intervention, multi-agent handoff

---

## Appendix

**A. Prompt Templates** — STEP1, STEP2, STEP3, STEP2_NOCTX verbatim
**B. Full Trajectory Table** — Figure 3 (supplementary heatmap)
**C. Pre-Registered Predictions** — PREDICTION.md verbatim with commit hash
**D. Failure Mode Formal Definitions** — Table with ROUGE-L criteria
**E. Feature Engineering Details** — 5-feature matrix specification

---

## Writing Timeline

| Milestone | Target Date | Notes |
|---|---|---|
| Run all experiments (Exp 01-05) | After Kaggle GPU session | ~10 GPU hours |
| Draft Sections 3, 4 | After results available | Fill in P1-P5 confirmations |
| Draft Sections 1, 5, 6, 7 | +1 week | Can draft now with placeholder results |
| Internal review | +2 weeks | |
| Arxiv preprint | After GUARDIAN Exp 01 complete | Submit both simultaneously |
| NeurIPS 2026 workshop deadline | Check TML workshop CFP | ~June/July 2026 |
| ICLR 2027 submission | October 2026 | Main track target |

---

## Key Numbers — FILLED (from RESULTS.md, 2026-03-28)

| # | Prediction | Result | Value | Status |
|---|---|---|---|---|
| 1 | P1: CASCADE+LATENT ≥ 30% of non-correct | (554+3)/584 = **95.4%** | CASCADE=92.3%, LATENT=0.5% | **CONFIRMED** |
| 2 | P2: early AUROC (3-feat, steps 1+2) | **0.652** | vs late 0.610 → +0.042 | REFUTED (< 0.05 threshold) |
| 3 | P2: early AUROC (5-feat, +cosine drift) | **0.662** | vs late 0.610 → +0.053 | Borderline (0.053 ≥ 0.05) |
| 4 | P2: late detection AUROC (step 3 only) | **0.610** | 1-feature baseline | — |
| 5 | P2: cosine drift feature gain | **+0.010** | 5-feat vs 3-feat | — |
| 6 | P3: LATENT Fisher proxy vs CORRECT at step 2 | **N/A** | Only 3 LATENT (< 10 needed) | INSUFFICIENT |
| 7 | P4: AMPLIFIED drop in context-free ablation | **-0.7pp** | 0.67% → 1.33% (opposite direction) | REFUTED (< 5pp threshold) |
| 8 | P4: mechanism finding | MODEL_BEHAVIOR_COLLAPSE | Cascade intrinsic to model | REFUTED SCoRe hypothesis |
| 9 | P5: persona direction AUROC for LATENT | **N/A** | Only 3 LATENT (< 10 needed) | INSUFFICIENT |

**Key unexpected finding:** CASCADE dominates at 92.3% (expected ~40-50%). LATENT, AMPLIFIED, ISOLATED, OSCILLATING are all rare (≤ 1% each). This is itself a scientific finding: self-refinement with Qwen 2.5 3B on TruthfulQA almost never partially self-corrects — it either fully succeeds (2.7%) or fully fails in cascade (92.3%).

**Implication for GUARDIAN:** The dominant failure mode is CASCADE (not LATENT). GUARDIAN's primary detection target should be CASCADE, with ABSTAIN as the preferred action (P4 shows REGENERATE cannot fix it).
