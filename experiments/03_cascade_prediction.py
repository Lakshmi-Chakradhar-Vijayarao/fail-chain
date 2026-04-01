"""
EXP 03 — Cascade Prediction Signal
KEY EXPERIMENT: Can Fisher geometry at steps 1+2 predict step 3 failure?

Four sub-experiments:
  A. Early warning vs late detection AUROC (3-feat and 5-feat)      [P2]
  B. Latent failure detection via standard features                  [P3]
  C. Persona direction detection — hallucination direction at step 2 [P5]
  D. Context ablation comparison — standard vs context-free          [P4]
  E. Cascade-risk vs safe AUROC                                      [supplementary]

Run on CPU after Exps 01 and 02 complete.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cascade_predict import (
    run_cascade_prediction,
    run_cascade_prediction_rich,
    test_latent_detection,
    test_persona_direction,
    test_context_ablation,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("FAIL-CHAIN — EXP 03: Cascade Prediction Signal")
print("=" * 60)

# ── Load standard pipeline data ───────────────────────────────────────────────
with open(RESULTS_DIR / "logs" / "pipeline_data.json") as f:
    data = json.load(f)
for d in data:
    d["h_s1"] = np.array(d["h_s1"])
    d["h_s2"] = np.array(d["h_s2"])
    d["h_s3"] = np.array(d["h_s3"])

print(f"Loaded {len(data)} standard pipeline samples")

# ── Load ablation data (if exists) ────────────────────────────────────────────
ablation_path = RESULTS_DIR / "logs" / "ablation_data.json"
ablation_data = []
if ablation_path.exists():
    with open(ablation_path) as f:
        ablation_data = json.load(f)
    for d in ablation_data:
        d["h_s1"] = np.array(d["h_s1"])
        d["h_s2"] = np.array(d["h_s2"])
        d["h_s3"] = np.array(d["h_s3"])
    print(f"Loaded {len(ablation_data)} context-free ablation samples")
else:
    print("WARNING: ablation_data.json not found — skipping P4 context ablation test")

print()

# ── A. Early warning vs late detection (P2) ───────────────────────────────────
print("A. Early Warning vs Late Detection (P2)")
print("-" * 60)

result = run_cascade_prediction(data, n_splits=5)
result_rich = run_cascade_prediction_rich(data, n_splits=5)

if "error" in result:
    print(f"ERROR: {result['error']}")
    p2_confirmed = False
else:
    print(f"3-feature early AUROC (steps 1+2): {result['auroc_early']:.3f}")
    print(f"5-feature early AUROC (+cosine):   {result_rich.get('auroc_early_5feat', 0):.3f}  "
          f"[cosine gain: {result_rich.get('cosine_gain', 0):+.3f}]")
    print(f"Late detection AUROC (step 3):     {result['auroc_late']:.3f}")
    print(f"Advantage (3-feat):                {result['advantage']:+.3f}")
    print(f"Advantage (5-feat):                {result_rich.get('advantage_5feat', 0):+.3f}")
    print(f"\nFinding: {result['finding']}")

    p2_confirmed = result["advantage"] >= 0.05
    print(f"\nP2: {'CONFIRMED ✓' if p2_confirmed else 'REFUTED ✗'} "
          f"(threshold: advantage ≥ 0.05)")

# ── B. Latent failure detection (P3) ─────────────────────────────────────────
print("\nB. Latent Failure Detection (P3)")
print("-" * 60)

latent_result = test_latent_detection(data, n_splits=3)

print(f"LATENT samples: {latent_result['n_latent']}")
if latent_result.get("result") not in ("INSUFFICIENT_SAMPLES", "INSUFFICIENT_LATENT"):
    print(f"AUROC (LATENT vs CORRECT, standard features): {latent_result['auroc']:.3f}")
    print(f"Interpretation: {latent_result['interpretation']}")
else:
    print(f"Note: {latent_result.get('note', 'insufficient samples')}")

# ── C. Persona direction detection (P5) ──────────────────────────────────────
print("\nC. Persona Direction Detection (P5)")
print("-" * 60)
print("Extracting hallucination direction from h_s1 correct/wrong split...")

persona_result = test_persona_direction(data, n_splits=3)

if "direction_s3_auroc" in persona_result:
    print(f"Step 1 direction AUROC (direction quality): {persona_result['direction_s3_auroc']:.3f}")

if persona_result.get("result") in ("INSUFFICIENT_S3_SAMPLES", "INSUFFICIENT_LATENT"):
    print(f"Note: {persona_result.get('note', 'insufficient samples')}")
    p5_confirmed = None
else:
    print(f"LATENT samples: {persona_result['n_latent']}")
    print(f"AUROC (persona direction → LATENT detection): {persona_result['auroc_persona_direction']:.3f}")
    print(f"Mean alignment — LATENT: {persona_result['mean_align_latent']:.4f}  |  "
          f"CORRECT: {persona_result['mean_align_correct']:.4f}")
    print(f"Interpretation: {persona_result['interpretation']}")

    p5_confirmed = persona_result["auroc_persona_direction"] >= 0.60
    print(f"\nP5: {'CONFIRMED ✓' if p5_confirmed else 'REFUTED ✗'} "
          f"(threshold: persona direction AUROC ≥ 0.60)")

# ── D. Context ablation (P4) ──────────────────────────────────────────────────
print("\nD. Context Ablation — Standard vs Context-Free Pipeline (P4)")
print("-" * 60)

ablation_result = {}
p4_confirmed = None

if ablation_data:
    # Match ablation to same questions as first N samples of standard
    n_abl = len(ablation_data)
    std_matched = [d for d in data if d["id"] < n_abl]

    ablation_result = test_context_ablation(std_matched, ablation_data)

    print(f"Comparing {ablation_result['n_standard']} standard vs "
          f"{ablation_result['n_noctx']} context-free samples")
    print(f"\n  {'Mode':<12} {'Standard %':>12} {'Context-Free %':>16}")
    print(f"  {'-'*42}")
    for mode in ("CORRECT", "AMPLIFIED", "CASCADE", "LATENT"):
        std_p  = ablation_result["standard"].get(mode, 0)
        noctx_p = ablation_result["noctx"].get(mode, 0)
        delta  = noctx_p - std_p
        print(f"  {mode:<12} {std_p:>11.1f}%  {noctx_p:>14.1f}%  ({delta:+.1f}pp)")

    print(f"\nAMPLIFIED drop: {ablation_result['amplified_drop_pct']:+.1f}pp")
    print(f"Mechanism: {ablation_result['mechanism']}")

    p4_confirmed = ablation_result["p4_confirmed"]
    print(f"\nP4: {'CONFIRMED ✓' if p4_confirmed else 'REFUTED ✗'} "
          f"(threshold: AMPLIFIED drops ≥ 5pp in context-free condition)")
else:
    print("Skipped — ablation_data.json not found")

# ── E. Cascade-risk vs safe AUROC ─────────────────────────────────────────────
print("\nE. Cascade-Risk vs Safe Samples")
print("-" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.fisher_trajectory import build_feature_matrix, build_rich_feature_matrix
from src.failure_modes import is_cascade_risk

cascade_samples = [d for d in data if is_cascade_risk(d["failure_mode"])]
safe_samples    = [d for d in data if d["failure_mode"] in ("CORRECT", "ISOLATED")]
auroc_cascade   = None

if len(cascade_samples) >= 20 and len(safe_samples) >= 20:
    combined = cascade_samples + safe_samples
    X = build_rich_feature_matrix(combined)
    y = np.array([1 if is_cascade_risk(d["failure_mode"]) else 0 for d in combined])

    auroc_cascade = cross_val_score(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))]),
        X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc"
    ).mean()

    print(f"Cascade-risk samples: {len(cascade_samples)}")
    print(f"Safe samples:         {len(safe_samples)}")
    print(f"AUROC (cascade vs safe, 5-feat): {auroc_cascade:.3f}")
else:
    print(f"Skipped — cascade_samples={len(cascade_samples)}, safe={len(safe_samples)}")

# ── Save all results ──────────────────────────────────────────────────────────
final_results = {
    "cascade_prediction":      result,
    "cascade_prediction_rich": result_rich,
    "latent_detection":        latent_result,
    "persona_direction":       persona_result,
    "context_ablation":        ablation_result,
    "cascade_vs_safe_auroc":   float(auroc_cascade) if auroc_cascade is not None else None,
    "prediction_checks": {
        "P2_confirmed": p2_confirmed if "p2_confirmed" in dir() else None,
        "P4_confirmed": p4_confirmed,
        "P5_confirmed": p5_confirmed if "p5_confirmed" in dir() else None,
    },
}

out_path = RESULTS_DIR / "logs" / "cascade_prediction.json"
with open(out_path, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\nSaved: {out_path}")
print("Run experiments/04_visualization.py next")
