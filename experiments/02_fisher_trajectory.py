"""
EXP 02 — Fisher J Trajectory Analysis
Compute per-step Fisher proxy and analyze trajectory by failure mode.
Run on CPU after EXP 01 completes.
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fisher_trajectory import fisher_proxy, compute_fisher_j, compute_trajectory

RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("FAIL-CHAIN — EXP 02: Fisher J Trajectory")
print("=" * 60)

# ── Load pipeline data ────────────────────────────────────────────────────────
data_path = RESULTS_DIR / "logs" / "pipeline_data.json"
with open(data_path) as f:
    data = json.load(f)

# Convert lists back to numpy
for d in data:
    d["h_s1"] = np.array(d["h_s1"])
    d["h_s2"] = np.array(d["h_s2"])
    d["h_s3"] = np.array(d["h_s3"])

print(f"Loaded {len(data)} samples")

# ── Failure mode distribution ─────────────────────────────────────────────────
modes = Counter(d["failure_mode"] for d in data)
print("\nFailure mode distribution:")
for mode, count in modes.most_common():
    print(f"  {mode:<16}: {count:>4} ({count/len(data)*100:.1f}%)")

# ── Fisher proxy trajectory per failure mode ──────────────────────────────────
print("\nFisher proxy trajectory by failure mode:")
print(f"{'Mode':<16} {'Step1':<12} {'Step2':<12} {'Step3':<12} {'Slope1→2':<12} {'Slope2→3':<12}")
print("-" * 76)

trajectory_results = {}
MODES_TO_ANALYZE = ["CORRECT", "CASCADE", "LATENT", "AMPLIFIED",
                     "ISOLATED", "OSCILLATING", "SELF_HEAL", "LATE_RECOVERY"]

for mode in MODES_TO_ANALYZE:
    mode_data = [d for d in data if d["failure_mode"] == mode]
    if len(mode_data) < 5:
        print(f"  {mode:<16}: insufficient samples ({len(mode_data)})")
        continue

    traj = compute_trajectory(mode_data)
    trajectory_results[mode] = traj

    print(f"  {mode:<16}: "
          f"{traj['step1']['mean']:.2f}±{traj['step1']['std']:.2f}  "
          f"{traj['step2']['mean']:.2f}±{traj['step2']['std']:.2f}  "
          f"{traj['step3']['mean']:.2f}±{traj['step3']['std']:.2f}  "
          f"{traj['slope_s1_s2']:+.2f}       "
          f"{traj['slope_s2_s3']:+.2f}")

# ── Latent vs Correct comparison (P3 prediction check) ───────────────────────
# Use boundary-excluded clean samples — consistent with cascade_predict.py analysis.
# Boundary samples (ROUGE within ±0.05 of threshold) have unreliable class labels.
clean_data = [d for d in data if not d.get("is_boundary", False)]
latent     = [d for d in clean_data if d["failure_mode"] == "LATENT"]
correct    = [d for d in clean_data if d["failure_mode"] == "CORRECT"]
n_boundary_excluded = len(data) - len(clean_data)

if len(latent) >= 5 and len(correct) >= 5:
    mean_j_s2_latent  = np.mean([fisher_proxy(d["h_s2"]) for d in latent])
    mean_j_s2_correct = np.mean([fisher_proxy(d["h_s2"]) for d in correct])

    print(f"\nP3 Prediction Check (Latent vs Correct at step 2, clean samples only):")
    print(f"  Boundary samples excluded: {n_boundary_excluded}")
    print(f"  Mean Fisher proxy step 2 — LATENT:  {mean_j_s2_latent:.3f}  (n={len(latent)})")
    print(f"  Mean Fisher proxy step 2 — CORRECT: {mean_j_s2_correct:.3f}  (n={len(correct)})")
    print(f"  Difference: {mean_j_s2_latent - mean_j_s2_correct:+.3f}")

    p3_confirmed = mean_j_s2_latent < mean_j_s2_correct
    print(f"  P3: {'CONFIRMED' if p3_confirmed else 'REFUTED'} "
          f"(prediction: LATENT step 2 proxy < CORRECT step 2 proxy)")
else:
    print(f"\nP3 check skipped: LATENT n={len(latent)}, CORRECT n={len(correct)} "
          f"(after excluding {n_boundary_excluded} boundary samples)")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "n_samples": len(data),
    "failure_mode_distribution": dict(modes),
    "trajectory_by_mode": trajectory_results,
    "latent_vs_correct_step2": {
        "latent_mean":          float(mean_j_s2_latent)  if len(latent)  >= 5 else None,
        "correct_mean":         float(mean_j_s2_correct) if len(correct) >= 5 else None,
        "n_latent_clean":       len(latent),
        "n_correct_clean":      len(correct),
        "n_boundary_excluded":  n_boundary_excluded,
    } if len(latent) >= 5 and len(correct) >= 5 else {
        "n_latent_clean":       len(latent),
        "n_correct_clean":      len(correct),
        "n_boundary_excluded":  n_boundary_excluded,
        "note":                 "insufficient clean samples for P3 check"
    }
}

out_path = RESULTS_DIR / "logs" / "fisher_trajectory.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {out_path}")
print("Run experiments/03_cascade_prediction.py next")
