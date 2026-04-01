"""
Failure mode taxonomy for FAIL-CHAIN.
Classifies each 3-step pipeline sample into one of 8 failure modes.

BOUNDARY ZONE:
ROUGE-L is a lexical overlap metric with high variance near the decision threshold.
A sample at rouge_s3=0.29 and rouge_s3=0.31 differ only in lexical phrasing,
not factual correctness — yet one is LATENT and the other CORRECT.

boundary_zone=0.05 defines a ±0.05 exclusion zone around threshold. Samples where
ANY step's ROUGE score falls within [threshold-boundary_zone, threshold+boundary_zone]
are flagged as boundary samples. These are excluded from LATENT/CORRECT
statistical comparisons (P3, P5) because their class assignment is unreliable.

This does NOT affect the taxonomy classification itself — boundary samples still
receive a failure mode label. The flag is used downstream in cascade_predict.py
to run "clean" analyses excluding boundary noise.
"""

BOUNDARY_ZONE = 0.05   # ±0.05 around threshold is unreliable assignment territory


def classify_failure_mode(rouge_s1: float, rouge_s2: float,
                           rouge_s3: float, threshold: float = 0.3) -> str:
    """
    Classify the failure mode of a 3-step self-refinement pipeline sample.

    All 8 boolean combinations of (wrong_s1, wrong_s2, wrong_s3) are named:
      CORRECT      (0,0,0)
      ISOLATED     (1,0,0)
      AMPLIFIED    (0,1,1)
      CASCADE      (1,1,1)
      LATE_RECOVERY(1,1,0)
      LATENT       (0,0,1)
      OSCILLATING  (1,0,1)
      SELF_HEAL    (0,1,0)  — step 2 introduces error that step 3 self-corrects

    Returns one of the 8 named modes.
    """
    wrong_s1 = rouge_s1 < threshold
    wrong_s2 = rouge_s2 < threshold
    wrong_s3 = rouge_s3 < threshold

    if not wrong_s1 and not wrong_s2 and not wrong_s3:
        return "CORRECT"
    if wrong_s1 and not wrong_s2 and not wrong_s3:
        return "ISOLATED"
    if not wrong_s1 and wrong_s2 and wrong_s3:
        return "AMPLIFIED"
    if wrong_s1 and wrong_s2 and wrong_s3:
        return "CASCADE"
    if wrong_s1 and wrong_s2 and not wrong_s3:
        return "LATE_RECOVERY"
    if not wrong_s1 and not wrong_s2 and wrong_s3:
        return "LATENT"
    if wrong_s1 and not wrong_s2 and wrong_s3:
        return "OSCILLATING"
    if not wrong_s1 and wrong_s2 and not wrong_s3:
        # Step 2 introduces error, step 3 self-corrects — SELF_HEAL
        # Previously named PARTIAL, but this is a fully determinate mode.
        return "SELF_HEAL"

    return "PARTIAL"   # unreachable given full boolean coverage above


def is_boundary_sample(rouge_s1: float, rouge_s2: float,
                        rouge_s3: float, threshold: float = 0.3,
                        zone: float = BOUNDARY_ZONE) -> bool:
    """
    Returns True if any step's ROUGE score falls within the boundary zone
    [threshold - zone, threshold + zone].

    Boundary samples have unreliable failure mode assignments because small
    lexical phrasing differences can flip the class label. Exclude these
    from LATENT/CORRECT statistical comparisons (P3, P5).

    Usage in pipeline_generation.py:
        "is_boundary": is_boundary_sample(rouge_s1, rouge_s2, rouge_s3)
    """
    lo, hi = threshold - zone, threshold + zone
    return any(lo <= r <= hi for r in (rouge_s1, rouge_s2, rouge_s3))


def is_cascade_risk(failure_mode: str) -> bool:
    """Returns True for failure modes that require pipeline-level governance."""
    return failure_mode in {"CASCADE", "LATENT", "AMPLIFIED", "OSCILLATING"}


def is_invisible_to_single_step(failure_mode: str) -> bool:
    """
    Returns True for failure modes where single-step governance at step 2 cannot
    catch the failure from output text alone (ROUGE-L passes at step 2).
    LATENT: steps 1+2 both pass threshold, step 3 fails.
    SELF_HEAL: step 2 fails but step 3 recovers — single-step at step 2 would
    trigger intervention unnecessarily.
    """
    return failure_mode in {"LATENT", "SELF_HEAL"}


FAILURE_MODE_DESCRIPTIONS = {
    "CORRECT":       "All three steps correct — no intervention needed",
    "ISOLATED":      "Error at step 1, self-corrected by step 2 — single step sufficient",
    "AMPLIFIED":     "Step 1 correct, step 2 introduces error, step 3 also wrong",
    "CASCADE":       "Error at step 1 propagates through full pipeline",
    "LATE_RECOVERY": "Errors at steps 1+2, correct at step 3 — pipeline self-heals late",
    "LATENT":        "Intermediate steps pass threshold, step 3 fails — INVISIBLE to output governance",
    "OSCILLATING":   "Error → recovery → error — unstable pipeline",
    "SELF_HEAL":     "Step 2 introduces error, step 3 self-corrects — single-step governance overtriggers",
}
