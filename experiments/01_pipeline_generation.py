"""
EXP 01 — Pipeline Generation
Run 3-step self-refinement pipeline on 600 TruthfulQA samples.
Extract hidden states at L32 for each step.

Run on Kaggle GPU (~4 hours on T4).
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import run_pipeline, run_pipeline_noctx
from src.failure_modes import classify_failure_mode, is_boundary_sample

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_LAYER    = 32
N_PIPELINE      = 600
N_HOLDOUT       = 100       # last 100 samples reserved for GUARDIAN test
N_ABLATION      = 150       # context-free ablation: matched subset of first 150 questions
ROUGE_THRESHOLD = 0.3
MAX_NEW_TOKENS  = 80
SAVE_EVERY      = 50        # checkpoint frequency
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "hidden_states").mkdir(exist_ok=True)
(RESULTS_DIR / "logs").mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Target layer: L{TARGET_LAYER}")
print(f"Pipeline samples: {N_PIPELINE}")


# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def compute_rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score  = scorer.score(reference, prediction)
    return score["rougeL"].fmeasure


# ── Load dataset ──────────────────────────────────────────────────────────────
print("\nLoading TruthfulQA...")
dataset  = load_dataset("truthful_qa", "generation")
samples  = list(dataset["validation"])
# Use first N_PIPELINE + N_HOLDOUT samples
samples  = samples[:N_PIPELINE + N_HOLDOUT]
pipeline_samples = samples[:N_PIPELINE]
holdout_samples  = samples[N_PIPELINE:]

print(f"Pipeline samples:  {len(pipeline_samples)}")
print(f"Holdout samples:   {len(holdout_samples)}")
print(f"Ablation samples:  {N_ABLATION} (context-free, matched to first {N_ABLATION} pipeline questions)")


# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading Qwen 2.5 3B...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded.")


# ── Pipeline generation ───────────────────────────────────────────────────────
pipeline_data = []
skipped = 0

for i, sample in enumerate(tqdm(pipeline_samples, desc="Pipeline")):
    question    = sample["question"]
    best_answer = sample["best_answer"]

    try:
        result = run_pipeline(
            model, tokenizer, question,
            target_layer=TARGET_LAYER,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE
        )

        rouge_s1 = compute_rouge_l(result["step1_output"], best_answer)
        rouge_s2 = compute_rouge_l(result["step2_output"], best_answer)
        rouge_s3 = compute_rouge_l(result["step3_output"], best_answer)

        failure_mode = classify_failure_mode(rouge_s1, rouge_s2, rouge_s3,
                                              ROUGE_THRESHOLD)
        boundary     = is_boundary_sample(rouge_s1, rouge_s2, rouge_s3,
                                          ROUGE_THRESHOLD)

        pipeline_data.append({
            "id":            i,
            "question":      question,
            "best_answer":   best_answer,
            "step1_output":  result["step1_output"],
            "step2_output":  result["step2_output"],
            "step3_output":  result["step3_output"],
            "h_s1":          result["h_s1"].tolist(),
            "h_s2":          result["h_s2"].tolist(),
            "h_s3":          result["h_s3"].tolist(),
            "rouge_s1":      rouge_s1,
            "rouge_s2":      rouge_s2,
            "rouge_s3":      rouge_s3,
            "label_s1":      int(rouge_s1 < ROUGE_THRESHOLD),
            "label_s2":      int(rouge_s2 < ROUGE_THRESHOLD),
            "label_s3":      int(rouge_s3 < ROUGE_THRESHOLD),
            "failure_mode":  failure_mode,
            "is_boundary":   boundary,   # exclude from LATENT/CORRECT analysis if True
        })

    except Exception as e:
        print(f"\nSample {i} failed: {e}")
        skipped += 1
        continue

    # Checkpoint
    if (i + 1) % SAVE_EVERY == 0:
        ckpt_path = RESULTS_DIR / "logs" / "pipeline_data_checkpoint.json"
        with open(ckpt_path, "w") as f:
            json.dump(pipeline_data, f)
        print(f"\n[{i+1}/{N_PIPELINE}] Checkpoint saved. Skipped: {skipped}")

# ── Save standard pipeline results ───────────────────────────────────────────
out_path = RESULTS_DIR / "logs" / "pipeline_data.json"
with open(out_path, "w") as f:
    json.dump(pipeline_data, f, indent=2)

hs = {
    "h_s1": np.array([d["h_s1"] for d in pipeline_data]),
    "h_s2": np.array([d["h_s2"] for d in pipeline_data]),
    "h_s3": np.array([d["h_s3"] for d in pipeline_data]),
    "labels_s1": np.array([d["label_s1"] for d in pipeline_data]),
    "labels_s2": np.array([d["label_s2"] for d in pipeline_data]),
    "labels_s3": np.array([d["label_s3"] for d in pipeline_data]),
}
np.savez(RESULTS_DIR / "hidden_states" / "pipeline_hidden_states.npz", **hs)

from collections import Counter
modes = Counter(d["failure_mode"] for d in pipeline_data)
total = len(pipeline_data)

print("\n" + "=" * 60)
print("EXP 01 STANDARD PIPELINE COMPLETE")
print(f"Samples processed: {total} (skipped: {skipped})")
print("\nFailure mode distribution:")
for mode, count in modes.most_common():
    print(f"  {mode:<16}: {count:>4} ({count/total*100:.1f}%)")
print("=" * 60)


# ── Context-free ablation (P4 test) ───────────────────────────────────────────
# Run run_pipeline_noctx() on first N_ABLATION questions.
# These are matched to the standard pipeline — same questions, same order.
# Comparison in Exp 03 measures AMPLIFIED rate drop (SCoRe hypothesis test).
print(f"\nRunning context-free ablation on first {N_ABLATION} questions...")
print("(Step 2 does NOT see step 1 output — tests context contamination hypothesis)")
print("-" * 60)

ablation_data  = []
ablation_skipped = 0

for i, sample in enumerate(tqdm(pipeline_samples[:N_ABLATION], desc="Ablation (noctx)")):
    question    = sample["question"]
    best_answer = sample["best_answer"]

    try:
        result = run_pipeline_noctx(
            model, tokenizer, question,
            target_layer=TARGET_LAYER,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE
        )

        rouge_s1 = compute_rouge_l(result["step1_output"], best_answer)
        rouge_s2 = compute_rouge_l(result["step2_output"], best_answer)
        rouge_s3 = compute_rouge_l(result["step3_output"], best_answer)

        failure_mode = classify_failure_mode(rouge_s1, rouge_s2, rouge_s3,
                                             ROUGE_THRESHOLD)
        boundary     = is_boundary_sample(rouge_s1, rouge_s2, rouge_s3,
                                          ROUGE_THRESHOLD)

        ablation_data.append({
            "id":           i,
            "question":     question,
            "best_answer":  best_answer,
            "step1_output": result["step1_output"],
            "step2_output": result["step2_output"],
            "step3_output": result["step3_output"],
            "h_s1":         result["h_s1"].tolist(),
            "h_s2":         result["h_s2"].tolist(),
            "h_s3":         result["h_s3"].tolist(),
            "rouge_s1":     rouge_s1,
            "rouge_s2":     rouge_s2,
            "rouge_s3":     rouge_s3,
            "label_s1":     int(rouge_s1 < ROUGE_THRESHOLD),
            "label_s2":     int(rouge_s2 < ROUGE_THRESHOLD),
            "label_s3":     int(rouge_s3 < ROUGE_THRESHOLD),
            "failure_mode": failure_mode,
            "is_boundary":  boundary,
            "pipeline_type": "noctx",
        })

    except Exception as e:
        print(f"\nAblation sample {i} failed: {e}")
        ablation_skipped += 1
        continue

    if (i + 1) % SAVE_EVERY == 0:
        ckpt = RESULTS_DIR / "logs" / "ablation_data_checkpoint.json"
        with open(ckpt, "w") as f:
            json.dump(ablation_data, f)
        print(f"\n[{i+1}/{N_ABLATION}] Ablation checkpoint saved.")

abl_path = RESULTS_DIR / "logs" / "ablation_data.json"
with open(abl_path, "w") as f:
    json.dump(ablation_data, f, indent=2)

abl_modes = Counter(d["failure_mode"] for d in ablation_data)
abl_total = len(ablation_data)

print("\n" + "=" * 60)
print("EXP 01 ABLATION COMPLETE")
print(f"Samples processed: {abl_total} (skipped: {ablation_skipped})")
print("\nContext-free failure mode distribution:")
for mode, count in abl_modes.most_common():
    print(f"  {mode:<16}: {count:>4} ({count/abl_total*100:.1f}%)")

# Quick AMPLIFIED comparison
std_matched = [d for d in pipeline_data if d["id"] < N_ABLATION]
std_amp_pct  = sum(1 for d in std_matched if d["failure_mode"] == "AMPLIFIED") / max(len(std_matched), 1) * 100
noctx_amp_pct = sum(1 for d in ablation_data if d["failure_mode"] == "AMPLIFIED") / max(abl_total, 1) * 100

print(f"\nAMPLIFIED — Standard: {std_amp_pct:.1f}%  |  Context-free: {noctx_amp_pct:.1f}%")
print(f"Drop: {std_amp_pct - noctx_amp_pct:+.1f}pp  (P4 threshold: ≥5pp drop to CONFIRM)")
print("=" * 60)
print(f"\nSaved standard data: {out_path}")
print(f"Saved ablation data:  {abl_path}")
print("Run experiments/02_fisher_trajectory.py next")
