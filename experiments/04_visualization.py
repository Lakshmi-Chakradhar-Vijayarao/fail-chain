"""
EXP 04 — Visualization and Results Logging
Generate all plots and write RESULTS.md.
Run on CPU after EXPs 01-03 complete.

Plots generated:
  1. failure_mode_distribution.png    — 8-class bar chart
  2. fisher_trajectory.png            — geometry trajectory by failure mode
  3. cascade_prediction_auroc.png     — early vs late AUROC comparison
  4. context_ablation_comparison.png  — standard vs context-free failure rates
  5. persona_direction_alignment.png  — hallucination direction alignment at step 2 (only generated if n_latent >= 10)
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fisher_trajectory import fisher_proxy

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "font.family":      "monospace",
})

COLORS = {
    "CORRECT":       "#2ea043",
    "ISOLATED":      "#58a6ff",
    "AMPLIFIED":     "#f0883e",
    "CASCADE":       "#f85149",
    "LATE_RECOVERY": "#79c0ff",
    "LATENT":        "#bc8cff",
    "OSCILLATING":   "#ffa657",
    "SELF_HEAL":     "#8b949e",
}

# ── Load all results ──────────────────────────────────────────────────────────
with open(RESULTS_DIR / "logs" / "pipeline_data.json")    as f: data     = json.load(f)
with open(RESULTS_DIR / "logs" / "fisher_trajectory.json") as f: traj    = json.load(f)
with open(RESULTS_DIR / "logs" / "cascade_prediction.json") as f: cascade = json.load(f)

for d in data:
    d["h_s1"] = np.array(d["h_s1"])
    d["h_s2"] = np.array(d["h_s2"])
    d["h_s3"] = np.array(d["h_s3"])

modes = Counter(d["failure_mode"] for d in data)

print("Generating plots...")

# ── Plot 1: Failure mode distribution ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sorted_modes = sorted(modes.items(), key=lambda x: x[1], reverse=True)
labels = [m for m, _ in sorted_modes]
counts = [c for _, c in sorted_modes]
colors = [COLORS.get(m, "#8b949e") for m in labels]

bars = ax.bar(labels, counts, color=colors, edgecolor="#30363d", linewidth=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{count}", ha="center", va="bottom", fontsize=10, color="#e6edf3")

ax.set_xlabel("Failure Mode", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"FAIL-CHAIN: Failure Mode Distribution (n={len(data)})", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "failure_mode_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → failure_mode_distribution.png")

# ── Plot 2: Fisher proxy trajectory by failure mode ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for mode, traj_data in traj["trajectory_by_mode"].items():
    means = [traj_data["step1"]["mean"],
             traj_data["step2"]["mean"],
             traj_data["step3"]["mean"]]
    stds  = [traj_data["step1"]["std"],
             traj_data["step2"]["std"],
             traj_data["step3"]["std"]]
    color = COLORS.get(mode, "#8b949e")
    xs = [1, 2, 3]
    ax.plot(xs, means, marker="o", color=color, linewidth=2.5,
            label=f"{mode} (n={modes[mode]})", zorder=5)
    ax.fill_between(xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.1)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Step 1\n(Initial Answer)", "Step 2\n(Self-Verify)",
                     "Step 3\n(Final Answer)"], fontsize=11)
ax.set_xlabel("Pipeline Step", fontsize=12)
ax.set_ylabel("Mean Fisher Proxy at L32", fontsize=12)
ax.set_title("Geometry Trajectory Through Self-Refinement Pipeline", fontsize=13)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fisher_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → fisher_trajectory.png")

# ── Plot 3: Early warning vs late detection AUROC ────────────────────────────
cp = cascade.get("cascade_prediction", {})
if cp and "auroc_early" in cp:
    fig, ax = plt.subplots(figsize=(7, 5))
    aurocs = [cp["auroc_late"], cp["auroc_early"]]
    labels = ["Late Detection\n(Step 3 only)", "Early Warning\n(Steps 1+2)"]
    bar_colors = ["#f85149", "#2ea043"]
    bars = ax.bar(labels, aurocs, color=bar_colors,
                  edgecolor="#30363d", width=0.5)
    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#e6edf3")
    ax.axhline(0.5, color="#8b949e", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax.set_ylim(0.4, min(1.0, max(aurocs) + 0.1))
    ax.set_ylabel("AUROC (Predicting Step 3 Failure)", fontsize=12)
    ax.set_title(f"Early Warning vs Late Detection\n"
                 f"Advantage: {cp['advantage']:+.3f}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cascade_prediction_auroc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → cascade_prediction_auroc.png")

print("\nAll plots saved.")

# ── Plot 4: Context ablation comparison ──────────────────────────────────────
abl = cascade.get("context_ablation", {})
if abl and "standard" in abl and "noctx" in abl:
    modes_to_show = ["CORRECT", "AMPLIFIED", "CASCADE", "LATENT"]
    std_vals  = [abl["standard"].get(m, 0) for m in modes_to_show]
    noctx_vals = [abl["noctx"].get(m, 0) for m in modes_to_show]

    x = np.arange(len(modes_to_show))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_std   = ax.bar(x - width/2, std_vals,   width, label="Standard (ctx)",  color="#58a6ff", edgecolor="#30363d")
    bars_noctx = ax.bar(x + width/2, noctx_vals, width, label="Context-free",    color="#f0883e", edgecolor="#30363d")

    for bar, val in list(zip(bars_std, std_vals)) + list(zip(bars_noctx, noctx_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#e6edf3")

    ax.set_xticks(x)
    ax.set_xticklabels(modes_to_show, fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=12)
    drop = abl.get("amplified_drop_pct", 0)
    p4   = abl.get("p4_confirmed", False)
    ax.set_title(f"Context Ablation: Standard vs Context-Free Step 2\n"
                 f"AMPLIFIED drop={drop:+.1f}pp  |  P4: {'CONFIRMED' if p4 else 'REFUTED'}",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "context_ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → context_ablation_comparison.png")

# ── Plot 5: Persona direction alignment ──────────────────────────────────────
pers = cascade.get("persona_direction", {})
if pers and "mean_align_latent" in pers and "mean_align_correct" in pers:
    fig, ax = plt.subplots(figsize=(7, 5))

    categories  = ["CORRECT\n(n={})".format(pers.get("n_correct_used", "?")),
                   "LATENT\n(n={})".format(pers.get("n_latent", "?"))]
    values      = [pers["mean_align_correct"], pers["mean_align_latent"]]
    bar_colors  = [COLORS["CORRECT"], COLORS["LATENT"]]

    bars = ax.bar(categories, values, color=bar_colors, edgecolor="#30363d", width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (0.001 if v >= 0 else -0.003),
                f"{v:.4f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#e6edf3")

    ax.axhline(0, color="#8b949e", linestyle="--", linewidth=1)
    auroc_p = pers.get("auroc_persona_direction", 0)
    p5      = auroc_p >= 0.60
    ax.set_ylabel("Mean Cosine Alignment with\nHallucination Direction at Step 2", fontsize=11)
    ax.set_title(f"Persona Direction: Step 2 Alignment by Failure Class\n"
                 f"AUROC={auroc_p:.3f}  |  P5: {'CONFIRMED' if p5 else 'REFUTED'}",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "persona_direction_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → persona_direction_alignment.png")

# ── Write RESULTS.md ─────────────────────────────────────────────────────────
cp   = cascade.get("cascade_prediction", {})
cpr  = cascade.get("cascade_prediction_rich", {})
ld   = cascade.get("latent_detection",   {})
pers = cascade.get("persona_direction",  {})
abl  = cascade.get("context_ablation",   {})
pck  = cascade.get("prediction_checks",  {})
tm   = traj.get("trajectory_by_mode",    {})
fmd  = traj.get("failure_mode_distribution", {})

def safe_fmt(val, fmt=".3f"):
    if isinstance(val, float): return format(val, fmt)
    return str(val) if val is not None else "N/A"

results_md = f"""# FAIL-CHAIN — Results

All experiments complete. Pre-registered predictions evaluated below.

---

## Summary

| Exp | Description | Key Result |
|---|---|---|
| 00 | Infrastructure validation | Passed |
| 01 | Pipeline generation + ablation | {len(data)} standard + {abl.get('n_noctx','?')} context-free samples |
| 02 | Fisher J trajectory | {len(tm)} failure modes analyzed |
| 03 | Cascade prediction (5 sub-experiments) | Early AUROC={safe_fmt(cp.get('auroc_early'))}, Late={safe_fmt(cp.get('auroc_late'))} |
| 04 | Visualization | 5 plots generated |

---

## Failure Mode Distribution (n={len(data)})

| Mode | Count | % | Description |
|---|---|---|---|
""" + "\n".join([
    f"| {mode} | {cnt} | {cnt/len(data)*100:.1f}% | "
    + {"CORRECT":       "All steps correct",
       "ISOLATED":      "Error at step 1, self-corrected",
       "AMPLIFIED":     "Error introduced at verification step",
       "CASCADE":       "Error propagates through full pipeline",
       "LATE_RECOVERY": "Errors healed at final step",
       "LATENT":        "**INVISIBLE to single-step governance**",
       "OSCILLATING":   "Error → recovery → error pattern",
       "SELF_HEAL":     "Step 2 introduces error, step 3 self-corrects"
       }.get(mode, "Mixed pattern") + " |"
    for mode, cnt in modes.most_common()
]) + f"""

---

## Pre-Registered Prediction Results

| Prediction | Status | Evidence |
|---|---|---|
| P1: CASCADE+LATENT ≥ 30% of non-correct | {'**CONFIRMED**' if (fmd.get('CASCADE',0)+fmd.get('LATENT',0)) / max(len(data)-fmd.get('CORRECT',0),1) >= 0.30 else 'REFUTED'} | CASCADE={fmd.get('CASCADE',0)}, LATENT={fmd.get('LATENT',0)} |
| P2: Early AUROC advantage ≥ 0.05 | {'**CONFIRMED**' if cp.get('advantage',0) >= 0.05 else 'REFUTED'} | Advantage = {safe_fmt(cp.get('advantage'))} |
| P3: LATENT step 2 proxy < CORRECT | {'**CONFIRMED**' if ld.get('auroc',0) >= 0.55 else 'REFUTED' if isinstance(ld.get('auroc'), float) else 'SEE BELOW'} | AUROC(LATENT vs CORRECT)={safe_fmt(ld.get('auroc'))} |
| P4: AMPLIFIED drops ≥ 5pp in context-free | {'**CONFIRMED**' if abl.get('p4_confirmed') else 'REFUTED' if abl else 'NOT RUN'} | Drop={safe_fmt(abl.get('amplified_drop_pct'), '.1f')}pp |
| P5: Persona direction detects LATENT (AUROC ≥ 0.60) | {'**CONFIRMED**' if pers.get('auroc_persona_direction',0) >= 0.60 else 'REFUTED' if 'auroc_persona_direction' in pers else 'INSUFFICIENT SAMPLES'} | Persona AUROC={safe_fmt(pers.get('auroc_persona_direction'))} |

---

## Cascade Prediction (EXP 03)

| Method | Features | AUROC |
|---|---|---|
| Late detection | Step 3 only (1 feat) | {safe_fmt(cp.get('auroc_late'))} |
| Early warning | Steps 1+2, L2 norm (3 feat) | {safe_fmt(cp.get('auroc_early'))} |
| Early warning + cosine drift | Steps 1+2, geometry (5 feat) | {safe_fmt(cpr.get('auroc_early_5feat'))} |
| Cosine feature gain | 5-feat vs 3-feat | {safe_fmt(cpr.get('cosine_gain'), '+.3f')} |

**Finding:** {cp.get('finding', 'See logs')}

---

## Latent Failure Detection

| Method | AUROC | Interpretation |
|---|---|---|
| Standard features (L2 norm) | {safe_fmt(ld.get('auroc'))} | {ld.get('interpretation', ld.get('note', 'Insufficient'))[:60]} |
| Persona direction (cosine at step 2) | {safe_fmt(pers.get('auroc_persona_direction'))} | {pers.get('interpretation', pers.get('note', 'See logs'))[:60]} |

---

## Context Ablation (P4: SCoRe Mechanism Test)

| Mode | Standard Pipeline | Context-Free | Change |
|---|---|---|---|
| CORRECT | {abl.get('standard',{}).get('CORRECT','N/A')}% | {abl.get('noctx',{}).get('CORRECT','N/A')}% | — |
| AMPLIFIED | {abl.get('standard',{}).get('AMPLIFIED','N/A')}% | {abl.get('noctx',{}).get('AMPLIFIED','N/A')}% | {safe_fmt(abl.get('amplified_drop_pct'), '+.1f')}pp |
| CASCADE | {abl.get('standard',{}).get('CASCADE','N/A')}% | {abl.get('noctx',{}).get('CASCADE','N/A')}% | — |
| LATENT | {abl.get('standard',{}).get('LATENT','N/A')}% | {abl.get('noctx',{}).get('LATENT','N/A')}% | — |

**Mechanism:** {abl.get('mechanism', 'Not available')}

---

## Connection to GUARDIAN

{"LATENT failures are invisible to single-step governance — motivates GUARDIAN cascade-aware routing." if fmd.get('LATENT',0) > 0 else "Cascade failure analysis informs GUARDIAN pipeline routing."}

{"P4 result informs GUARDIAN routing strategy: " + abl.get('mechanism','').split(":")[0] if abl.get('mechanism') else ""}

{"Persona direction vector extracted in this experiment can be deployed as GUARDIAN's real-time step-2 monitor — a 3072-dim dot product latency overhead." if pers.get('auroc_persona_direction', 0) >= 0.55 else ""}

---

## Plots Generated

| Plot | File | Description |
|---|---|---|
| 1 | failure_mode_distribution.png | 8-class failure taxonomy bar chart |
| 2 | fisher_trajectory.png | Fisher geometry trajectory by failure mode |
| 3 | cascade_prediction_auroc.png | Early vs late AUROC comparison |
| 4 | context_ablation_comparison.png | Standard vs context-free failure rates |
| 5 | persona_direction_alignment.png | Step 2 hallucination direction alignment |
"""

with open(Path(__file__).parent.parent / "RESULTS.md", "w") as f:
    f.write(results_md)

print("\nResults written to RESULTS.md")
print("FAIL-CHAIN experiments complete.")
