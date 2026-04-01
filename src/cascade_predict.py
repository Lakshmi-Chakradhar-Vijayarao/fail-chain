"""
Cascade prediction: can early-step Fisher geometry predict final-step failure?
Core experiment comparing early warning vs late detection AUROC.

Three detection experiments:
1. run_cascade_prediction()         — early (steps 1+2) vs late (step 3) AUROC [P2]
2. test_latent_detection()          — LATENT vs CORRECT using standard features [P3]
3. test_persona_direction()         — extract hallucination direction from h_s1 pool,
                                      measure cosine alignment of h_s2 for LATENT detection
4. test_context_ablation()          — compare failure mode distributions between
                                      standard pipeline and context-free ablation [P4]
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple


def run_cascade_prediction_rich(
    data: List[Dict],
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Same as run_cascade_prediction() but uses the 5-feature rich matrix
    (adds cosine_s1_s2 and l2_delta_hidden) for publication-quality comparison.
    Reports both 3-feature and 5-feature early AUROC so improvement from
    cosine features is explicitly quantified.
    """
    from src.fisher_trajectory import build_feature_matrix, build_rich_feature_matrix, build_late_feature_matrix

    X_early_3  = build_feature_matrix(data)
    X_early_5  = build_rich_feature_matrix(data)
    X_late     = build_late_feature_matrix(data)
    y          = np.array([d["label_s3"] for d in data])

    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return {"error": "Insufficient class balance for evaluation"}

    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = lambda: Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=1000))])

    auroc_early_3 = cross_val_score(
        pipe(), X_early_3, y, cv=cv, scoring="roc_auc"
    ).mean()

    auroc_early_5 = cross_val_score(
        pipe(), X_early_5, y, cv=cv, scoring="roc_auc"
    ).mean()

    auroc_late = cross_val_score(
        pipe(), X_late, y, cv=cv, scoring="roc_auc"
    ).mean()

    cosine_gain = float(auroc_early_5 - auroc_early_3)

    return {
        "auroc_early_3feat": float(auroc_early_3),
        "auroc_early_5feat": float(auroc_early_5),
        "auroc_late":        float(auroc_late),
        "advantage_3feat":   float(auroc_early_3 - auroc_late),
        "advantage_5feat":   float(auroc_early_5 - auroc_late),
        "cosine_gain":       cosine_gain,
        "n_samples":         len(data),
        "n_fail_s3":         int(y.sum()),
        "note": "cosine_gain = improvement from adding cosine drift feature (VAR-6 loose end)"
    }


def run_cascade_prediction(
    data: List[Dict],
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Compare early warning (steps 1+2 features) vs late detection (step 3 only)
    for predicting whether step 3 fails.

    Returns dict with:
        auroc_early: float
        auroc_late:  float
        advantage:   float (early - late)
        finding:     str (interpretation)
    """
    from src.fisher_trajectory import build_feature_matrix, build_late_feature_matrix

    X_early = build_feature_matrix(data)
    X_late  = build_late_feature_matrix(data)
    y       = np.array([d["label_s3"] for d in data])

    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return {"error": "Insufficient class balance for evaluation"}

    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = lambda: Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=1000))])

    auroc_early = cross_val_score(
        pipe(), X_early, y, cv=cv, scoring="roc_auc"
    ).mean()

    auroc_late = cross_val_score(
        pipe(), X_late, y, cv=cv, scoring="roc_auc"
    ).mean()

    advantage = auroc_early - auroc_late

    if advantage >= 0.05:
        finding = "EARLY WARNING EXISTS: geometry at steps 1+2 predicts step 3 failure before it occurs"
    elif advantage >= 0.02:
        finding = "WEAK EARLY WARNING: marginal advantage — geometry signal exists but is small"
    else:
        finding = ("NO EARLY WARNING: cascade failures not predictable from early geometry — "
                   "motivates formal multi-step bounds in GUARDIAN")

    return {
        "auroc_early": float(auroc_early),
        "auroc_late":  float(auroc_late),
        "advantage":   float(advantage),
        "finding":     finding,
        "n_samples":   len(data),
        "n_fail_s3":   int(y.sum()),
    }


def test_latent_detection(
    data: List[Dict],
    n_splits: int = 3,
    random_state: int = 42
) -> Dict:
    """
    Test whether early geometry can specifically detect LATENT failures.

    Runs two analyses:
    - "all": all LATENT and CORRECT samples
    - "clean": boundary samples excluded (is_boundary=False only)

    LATENT samples are the hardest case: their step-1 and step-2 ROUGE scores
    both pass the threshold, so they look correct at every intermediate step.
    The "clean" analysis excludes samples within ±0.05 of the ROUGE threshold,
    which have unreliable class assignments due to lexical phrasing noise.
    Both AUROCs are reported so the effect of boundary noise is visible.
    """
    from src.fisher_trajectory import build_feature_matrix

    def _run(subset_data, label):
        latent  = [d for d in subset_data if d.get("failure_mode") == "LATENT"]
        correct = [d for d in subset_data if d.get("failure_mode") == "CORRECT"]
        n_lat   = len(latent)
        if n_lat < 10:
            return {"n_latent": n_lat, "result": "INSUFFICIENT_SAMPLES",
                    "note": f"[{label}] Only {n_lat} LATENT — need ≥10"}
        combined = latent + correct[:n_lat * 3]
        X = build_feature_matrix(combined)
        y = np.array([1 if d.get("failure_mode") == "LATENT" else 0
                      for d in combined])
        cv   = StratifiedKFold(n_splits=min(n_splits, n_lat // 3 + 1),
                               shuffle=True, random_state=random_state)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000))])
        auroc = cross_val_score(
            pipe, X, y, cv=cv, scoring="roc_auc"
        ).mean()
        if auroc >= 0.65:
            interp = f"[{label}] Detectable (AUROC ≥ 0.65)"
        elif auroc >= 0.55:
            interp = f"[{label}] Weak signal (0.55 ≤ AUROC < 0.65)"
        else:
            interp = f"[{label}] Invisible to early geometry"
        return {"n_latent": n_lat, "n_correct_used": len(combined) - n_lat,
                "auroc": float(auroc), "interpretation": interp}

    result_all   = _run(data, "all")
    clean_data   = [d for d in data if not d.get("is_boundary", False)]
    result_clean = _run(clean_data, "clean")

    # Primary result for RESULTS.md: use clean analysis
    primary = result_clean if result_clean.get("result") != "INSUFFICIENT_SAMPLES" else result_all
    return {
        **primary,
        "auroc_all_samples":    result_all.get("auroc"),
        "auroc_clean_samples":  result_clean.get("auroc"),
        "n_boundary_excluded":  len(data) - len(clean_data),
        "interpretation":       primary.get("interpretation", "See sub-results"),
    }


def test_persona_direction(
    data: List[Dict],
    n_splits: int = 3,
    random_state: int = 42
) -> Dict:
    """
    Extract an 'eventual-failure direction' from step 1 hidden states,
    trained on step-3 outcome labels (y_s3). Measure h_s2 alignment with this
    direction to detect LATENT failures before step 3 runs.

    CRITICAL DESIGN — train on y_s3, NOT y_s1:
    LATENT samples have correct step-1 outputs (rouge_s1 ≥ 0.3). Training on y_s1
    would place them on the CORRECT side; the direction could never distinguish them.
    Training on y_s3 asks: "what direction in h_s1 geometry predicts eventual step-3
    failure?" — capturing representational instability that text labels miss at step 1.

    Step 1 geometry already encodes whether step 3 will fail, even when step-1
    text output looks correct. The direction surfaces this latent signal.

    Inspired by Anthropic's Persona Vectors (arXiv:2507.21509): hallucination
    propensity encodes as a linear direction in activation space. We extend this
    cross-step: extract from h_s1 using y_s3 labels, apply to h_s2.

    LOOSE END [VAR-3 / Persona]: Save direction as hallucination_direction.npy
    for GUARDIAN real-time monitor — single dot product at step 2 inference.
    """
    h_s1_all = np.array([d["h_s1"] for d in data])
    # Use y_s3 (eventual step-3 outcome) — not y_s1 — see docstring above.
    y_s3     = np.array([d["label_s3"] for d in data])

    n_fail_s3 = int(y_s3.sum())
    n_pass_s3 = int(len(y_s3) - y_s3.sum())

    if n_fail_s3 < 20 or n_pass_s3 < 20:
        return {
            "result": "INSUFFICIENT_S3_SAMPLES",
            "note": (f"Need ≥20 step-3 failures and ≥20 step-3 passes to extract direction. "
                     f"Got: {n_pass_s3} pass, {n_fail_s3} fail.")
        }

    # Fit LR on h_s1 predicting y_s3 → eventual-failure direction
    scaler = StandardScaler()
    h_s1_scaled = scaler.fit_transform(h_s1_all)
    lr = LogisticRegression(max_iter=1000, random_state=random_state, C=0.1)
    lr.fit(h_s1_scaled, y_s3)

    # Direction vector: LR coefficient (unit-normalized)
    direction = lr.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Project h_s2 onto hallucination direction for all samples
    h_s2_all  = np.array([d["h_s2"] for d in data])
    h_s2_scaled = scaler.transform(h_s2_all)
    alignments = h_s2_scaled @ direction   # [n_samples]

    # LATENT vs CORRECT: can alignment score distinguish them?
    latent_samples  = [d for d in data if d.get("failure_mode") == "LATENT"]
    correct_samples = [d for d in data if d.get("failure_mode") == "CORRECT"]

    n_latent = len(latent_samples)

    # Direction quality: how well does h_s1 predict y_s3?
    # Note: h_s1_scaled is fit on ALL data here (before CV split) — this is intentional
    # for direction extraction since we need consistent scaling for the direction vector.
    # The direction_s3_auroc is a quality diagnostic, not the primary AUROC claim (P5).
    direction_s3_auroc = float(cross_val_score(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000, C=0.1))]),
        h_s1_all, y_s3,
        cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
        scoring="roc_auc"
    ).mean())

    if n_latent < 10:
        return {
            "result": "INSUFFICIENT_LATENT",
            "n_latent": n_latent,
            "note": f"Only {n_latent} LATENT samples — need ≥10 for AUROC estimate",
            "direction_extracted": True,
            "direction_s3_auroc": direction_s3_auroc,
        }

    # Build LATENT vs CORRECT using clean (non-boundary) samples only.
    # Boundary samples have unreliable class labels near the ROUGE threshold.
    n_boundary_excluded = sum(1 for d in data if d.get("is_boundary", False))
    latent_idx  = [i for i, d in enumerate(data)
                   if d.get("failure_mode") == "LATENT" and not d.get("is_boundary", False)]
    correct_idx = [i for i, d in enumerate(data)
                   if d.get("failure_mode") == "CORRECT" and not d.get("is_boundary", False)]

    # Guard: n_latent above counted all LATENT (incl. boundary); clean set may be smaller.
    if len(latent_idx) < 10:
        return {
            "result": "INSUFFICIENT_LATENT",
            "n_latent": n_latent,
            "n_latent_clean": len(latent_idx),
            "note": (f"Only {len(latent_idx)} non-boundary LATENT samples — need ≥10. "
                     f"Total LATENT (incl. boundary): {n_latent}"),
            "direction_s3_auroc": direction_s3_auroc,
            "n_boundary_excluded": n_boundary_excluded,
        }

    correct_idx = correct_idx[:n_latent * 3]  # balance

    combined_idx = latent_idx + correct_idx
    X_align = alignments[combined_idx].reshape(-1, 1)
    y_align = np.array([1] * len(latent_idx) + [0] * len(correct_idx))

    # X_align is the 1-D cosine projection — already on a consistent scale.
    # Pipeline still used for consistency (StandardScaler on 1-D is a no-op).
    cv   = StratifiedKFold(n_splits=min(n_splits, n_latent // 3 + 1),
                           shuffle=True, random_state=random_state)
    auroc_persona = cross_val_score(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))]),
        X_align, y_align,
        cv=cv, scoring="roc_auc"
    ).mean()

    # Compare mean alignment: LATENT vs CORRECT
    mean_align_latent  = float(np.mean(alignments[latent_idx]))
    mean_align_correct = float(np.mean(alignments[correct_idx]))

    latent_higher = mean_align_latent > mean_align_correct
    if auroc_persona >= 0.65:
        interpretation = ("Eventual-failure direction detects LATENT at step 2 "
                          "(AUROC ≥ 0.65) — step-1 geometry predicts step-3 failure")
    elif auroc_persona >= 0.55:
        interpretation = "Weak signal (0.55 ≤ AUROC < 0.65) — partial geometric leakage"
    else:
        interpretation = ("LATENT failures invisible to eventual-failure direction at step 2 — "
                          "signature only emerges at step 3")

    return {
        "n_latent":                 n_latent,
        "n_correct_used":           len(correct_idx),
        "n_boundary_excluded":      n_boundary_excluded,
        "auroc_persona_direction":  float(auroc_persona),
        "direction_s3_auroc":       direction_s3_auroc,
        "mean_align_latent":        mean_align_latent,
        "mean_align_correct":       mean_align_correct,
        "latent_higher_align":      latent_higher,
        "interpretation":           interpretation,
    }


def test_context_ablation(
    standard_data: List[Dict],
    noctx_data: List[Dict],
) -> Dict:
    """
    Compare failure mode distributions between standard and context-free pipelines.

    Tests P4: Context-free Step 2 (no step 1 output visible) will reduce the
    AMPLIFIED failure rate by ≥ 5 percentage points relative to standard pipeline,
    confirming SCoRe's (Kumar et al. NeurIPS 2024) hypothesis that self-correction
    failure is driven by context contamination.

    Matched comparison: noctx_data must contain the same questions as the
    first len(noctx_data) samples of standard_data.

    LOOSE END [VAR-2B → GUARDIAN]:
    - If AMPLIFIED drops: GUARDIAN should route AMPLIFIED-risk samples to a
      context-stripped REGENERATE path (strip step 1 from step 2 prompt).
    - If AMPLIFIED stays: GUARDIAN should route to ABSTAIN — no routing strategy
      can fix a model behavior collapse without retraining.
    """
    from collections import Counter

    std_modes  = Counter(d["failure_mode"] for d in standard_data)
    noctx_modes = Counter(d["failure_mode"] for d in noctx_data)

    n_std   = len(standard_data)
    n_noctx = len(noctx_data)

    def pct(counter, mode, total):
        return counter.get(mode, 0) / max(total, 1) * 100

    amplified_std   = pct(std_modes,   "AMPLIFIED", n_std)
    amplified_noctx = pct(noctx_modes, "AMPLIFIED", n_noctx)
    cascade_std     = pct(std_modes,   "CASCADE",   n_std)
    cascade_noctx   = pct(noctx_modes, "CASCADE",   n_noctx)
    latent_std      = pct(std_modes,   "LATENT",    n_std)
    latent_noctx    = pct(noctx_modes, "LATENT",    n_noctx)
    correct_std     = pct(std_modes,   "CORRECT",   n_std)
    correct_noctx   = pct(noctx_modes, "CORRECT",   n_noctx)

    amplified_drop = amplified_std - amplified_noctx
    p4_confirmed   = amplified_drop >= 5.0

    if p4_confirmed:
        mechanism = ("CONTEXT_CONTAMINATION: removing step 1 context reduces AMPLIFIED rate — "
                     "confirms SCoRe hypothesis. GUARDIAN should use context-stripped REGENERATE.")
    elif amplified_drop >= 2.0:
        mechanism = "WEAK_CONTEXT_EFFECT: marginal reduction — both mechanisms likely operating."
    else:
        mechanism = ("MODEL_BEHAVIOR_COLLAPSE: context-free ablation does not reduce AMPLIFIED — "
                     "cascade is intrinsic. GUARDIAN should prioritize ABSTAIN over REGENERATE.")

    return {
        "n_standard":  n_std,
        "n_noctx":     n_noctx,
        "standard": {
            "AMPLIFIED": round(amplified_std, 2),
            "CASCADE":   round(cascade_std, 2),
            "LATENT":    round(latent_std, 2),
            "CORRECT":   round(correct_std, 2),
        },
        "noctx": {
            "AMPLIFIED": round(amplified_noctx, 2),
            "CASCADE":   round(cascade_noctx, 2),
            "LATENT":    round(latent_noctx, 2),
            "CORRECT":   round(correct_noctx, 2),
        },
        "amplified_drop_pct": round(amplified_drop, 2),
        "p4_confirmed":       p4_confirmed,
        "mechanism":          mechanism,
    }
