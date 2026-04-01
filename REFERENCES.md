# References — FAIL-CHAIN

*Papers cited or directly compared against in this project.*

---

## Self-Refinement & Multi-Step Pipelines

**[1] Huang et al. — Large Language Models Cannot Self-Correct Reasoning Yet**
Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, et al.
*arXiv*, 2023.
> Shows self-refinement does not reliably self-correct. Primary motivation for studying cascade failure rates in 3-step pipelines.

**[2] Madaan et al. — Self-Refine: Iterative Refinement with Self-Feedback**
Aman Madaan, Niket Taneja, Prakhar Gupta, Skyler Hallinan, et al.
*NeurIPS*, 2023.
> Canonical self-refinement pipeline architecture. We study when it fails, not optimize it. Our 3-step pipeline mirrors this structure.

**[3] Kumar et al. — Training Language Models to Self-Correct via Reinforcement Learning (SCoRe)**
Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, et al.
*NeurIPS*, 2024.
> Reinforcement learning approach to self-correction. We detect the geometric collapse that makes self-correction fail, rather than training around it.

---

## Multi-Agent & Pipeline Failure

**[4] MAST — Multi-Agent System Failure Taxonomy**
*arXiv:2503.13657*, 2025.
> Inter-agent communication failures in multi-agent systems. FAIL-CHAIN provides the representation-level detection mechanism complementing this taxonomy.

**[5] Zhang et al. — How Language Models Hallucinate, and Why: The Snowball Effect**
Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith.
*arXiv:2305.13534*, 2023.
> Hallucination snowball effect: models double down on initial mistakes. Directly motivates our CASCADE failure mode (α = 0.9947).

**[6] From Spark to Fire — Cascade Errors in Multi-Agent LLM Pipelines**
*arXiv:2603.04474*, 2026.
> Empirical study of cascade errors in multi-agent settings. Read before any conference submission.

**[7] Yao et al. — ReAct: Synergizing Reasoning and Acting in Language Models**
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, et al.
*arXiv*, 2022. Published at *ICLR*, 2023.
> ReAct multi-step reasoning architecture. Our 3-step pipeline structure (query → step 1 → step 2 → step 3) directly models this.

---

## Uncertainty & Output-Level Detection

**[8] Kuhn et al. — Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation**
Lorenz Kuhn, Yarin Gal, Sebastian Farquhar.
*ICLR*, 2023.
> Output-level semantic uncertainty. Our approach operates at the representation level (not output) and detects collapse before it propagates.

---

## Hidden States as Early Signals

**[9] Hidden States as Early Warning Signals**
*arXiv:2601.09093*, 2025.
> Hidden state early warning signals for failure detection. Our work extends this with unsupervised geometry (Fisher J trajectory) and Markov chain formalization.

---

## Faithfulness of Reasoning

**[10] Anthropic — Reasoning Models Don't Always Say What They Think**
Anthropic.
*Technical Report*, 2025.
> CoT is unfaithful 59–81% of the time. Motivates pre-output geometry check rather than trusting chain-of-thought.

---

## Related Architectures

**[11] "Mamba Knockout" — Factual Flow in Mamba Architectures**
*ACL*, 2025.
> Mamba-1 and Mamba-2 have different factual information flow patterns. Strengthens architecture transfer implications of our Fisher J collapse result.

---

## Fisher Information Background

**[12] Fisher Information Metric for Large Language Models**
*arXiv:2506.15830*, 2025.
> Fisher Information applied to LLMs. Our F-ratio = 0.007 result (global J fails at step 2) connects to this theoretical framework.

---

## Dataset

**[13] Lin et al. — TruthfulQA: Measuring How Models Mimic Human Falsehoods**
Stephanie Lin, Jacob Hilton, Owain Evans.
*ACL*, 2022.
> Primary dataset. 600 three-step self-refinement pipelines constructed from TruthfulQA questions.

---

*Total: 13 references*
