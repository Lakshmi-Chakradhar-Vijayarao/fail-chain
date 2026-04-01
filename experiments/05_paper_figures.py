"""
EXP 05 — Publication-Quality Figures for Paper Submission
Generates the main 4-panel figure and supporting figures for the paper.
Run on CPU after all other experiments complete.

Output files (results/plots/paper/):
  fig1_main_4panel.pdf       — Main result figure (for paper body; Panel D shows "insufficient samples" if LATENT n<10)
  fig1_main_4panel.png       — PNG version (for README / preprint)
  fig2_ablation.pdf          — Context ablation (for paper Section 4.5)
  fig2_ablation.png          — PNG version
  fig_supp_trajectory.pdf    — Full trajectory table (supplementary)
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fisher_trajectory import fisher_proxy

RESULTS_DIR = Path(__file__).parent.parent / "results"
PAPER_DIR   = RESULTS_DIR / "plots" / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication-quality style (white background, paper-ready) ─────────────────
plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#444444",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "text.color":           "#111111",
    "axes.labelcolor":      "#111111",
    "xtick.color":          "#444444",
    "ytick.color":          "#444444",
    "grid.color":           "#dddddd",
    "grid.linewidth":       0.6,
    "font.family":          "serif",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "legend.fontsize":      9,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
})

COLORS = {
    "CORRECT":       "#2d8a4e",
    "ISOLATED":      "#2b7cbf",
    "AMPLIFIED":     "#d4701a",
    "CASCADE":       "#c0392b",
    "LATE_RECOVERY": "#5dade2",
    "LATENT":        "#7d3c98",
    "OSCILLATING":   "#e67e22",
    "SELF_HEAL":     "#888888",
}

# ── Load all results ──────────────────────────────────────────────────────────
print("Loading results...")
with open(RESULTS_DIR / "logs" / "pipeline_data.json")     as f: data     = json.load(f)
with open(RESULTS_DIR / "logs" / "fisher_trajectory.json") as f: traj     = json.load(f)
with open(RESULTS_DIR / "logs" / "cascade_prediction.json") as f: cascade = json.load(f)

for d in data:
    d["h_s1"] = np.array(d["h_s1"])
    d["h_s2"] = np.array(d["h_s2"])
    d["h_s3"] = np.array(d["h_s3"])

modes     = Counter(d["failure_mode"] for d in data)
cp        = cascade.get("cascade_prediction", {})
cpr       = cascade.get("cascade_prediction_rich", {})
ld        = cascade.get("latent_detection", {})
pers      = cascade.get("persona_direction", {})
abl       = cascade.get("context_ablation", {})
traj_mode = traj.get("trajectory_by_mode", {})
fmd       = traj.get("failure_mode_distribution", {})


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main 4-panel figure
# Panel A: Failure mode distribution
# Panel B: Fisher proxy trajectory by failure mode
# Panel C: Early vs late AUROC (3-feat vs 5-feat vs late)
# Panel D: LATENT detection summary (persona direction vs standard)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1 (main 4-panel)...")

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax_a = fig.add_subplot(gs[0, 0])   # Panel A — failure mode distribution
ax_b = fig.add_subplot(gs[0, 1])   # Panel B — trajectory
ax_c = fig.add_subplot(gs[1, 0])   # Panel C — AUROC comparison
ax_d = fig.add_subplot(gs[1, 1])   # Panel D — LATENT detection

# ── Panel A: Failure mode distribution ───────────────────────────────────────
sorted_modes = sorted(modes.items(), key=lambda x: x[1], reverse=True)
labels_a = [m for m, _ in sorted_modes]
counts_a = [c for _, c in sorted_modes]
colors_a = [COLORS.get(m, "#888888") for m in labels_a]

bars = ax_a.bar(labels_a, counts_a, color=colors_a, edgecolor="white",
                linewidth=0.8, width=0.7)
for bar, count in zip(bars, counts_a):
    ax_a.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 1,
              str(count), ha="center", va="bottom", fontsize=8)

ax_a.set_ylabel("Count")
ax_a.set_title(f"(A) Failure Mode Distribution (n={len(data)})")
ax_a.grid(axis="y", alpha=0.5)
plt.setp(ax_a.get_xticklabels(), rotation=35, ha="right", fontsize=8)

# Annotate cascade-risk zone
cascade_risk_pct = sum(counts_a[i] for i, m in enumerate(labels_a)
                        if m in ("CASCADE","LATENT","AMPLIFIED","OSCILLATING")) / len(data) * 100
ax_a.text(0.97, 0.97, f"Cascade-risk:\n{cascade_risk_pct:.1f}%",
          transform=ax_a.transAxes, ha="right", va="top",
          fontsize=8, color="#c0392b",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde8e8", edgecolor="#c0392b", alpha=0.9))

# ── Panel B: Fisher proxy trajectory ─────────────────────────────────────────
TRAJ_MODES = ["CORRECT", "CASCADE", "LATENT", "AMPLIFIED", "ISOLATED"]
xs = [1, 2, 3]
xlabels = ["Step 1\nInitial", "Step 2\nVerify", "Step 3\nFinal"]

for mode in TRAJ_MODES:
    if mode not in traj_mode:
        continue
    td = traj_mode[mode]
    means = [td["step1"]["mean"], td["step2"]["mean"], td["step3"]["mean"]]
    stds  = [td["step1"]["std"],  td["step2"]["std"],  td["step3"]["std"]]
    color = COLORS.get(mode, "#888888")
    n     = fmd.get(mode, modes.get(mode, 0))
    lw    = 2.5 if mode in ("CASCADE", "LATENT") else 1.5
    ls    = "--" if mode == "LATENT" else "-"

    ax_b.plot(xs, means, marker="o", color=color, linewidth=lw,
              linestyle=ls, label=f"{mode} (n={n})", zorder=5, markersize=5)
    ax_b.fill_between(xs,
                      [m - s for m, s in zip(means, stds)],
                      [m + s for m, s in zip(means, stds)],
                      color=color, alpha=0.08)

ax_b.set_xticks(xs)
ax_b.set_xticklabels(xlabels)
ax_b.set_ylabel("Mean Fisher Proxy (L2 norm at L32)")
ax_b.set_title("(B) Geometry Trajectory Through Pipeline")
ax_b.legend(loc="best", fontsize=8)
ax_b.grid(alpha=0.4)

# ── Panel C: AUROC comparison bar chart ──────────────────────────────────────
auroc_values = [
    cp.get("auroc_late",        0),
    cp.get("auroc_early",       0),
    cpr.get("auroc_early_5feat", 0),
]
bar_labels = [
    "Late\n(Step 3, 1 feat)",
    "Early\n(Steps 1+2, 3 feat)",
    "Early+Cosine\n(Steps 1+2, 5 feat)",
]
bar_colors_c = ["#c0392b", "#2b7cbf", "#2d8a4e"]

bars_c = ax_c.bar(bar_labels, auroc_values, color=bar_colors_c,
                   edgecolor="white", linewidth=0.8, width=0.55)
for bar, v in zip(bars_c, auroc_values):
    ax_c.text(bar.get_x() + bar.get_width()/2,
              v + 0.005, f"{v:.3f}",
              ha="center", va="bottom", fontsize=10, fontweight="bold")

ax_c.axhline(0.5, color="#888888", linestyle="--", linewidth=1, label="Chance (0.50)")
ax_c.set_ylim(0.4, min(1.0, max(auroc_values) + 0.12))
ax_c.set_ylabel("AUROC (Predicting Step 3 Failure)")
adv = cp.get("advantage", 0)
adv_rich = cpr.get("advantage_5feat", 0)
ax_c.set_title(f"(C) Early Warning vs Late Detection\n"
               f"3-feat advantage: {adv:+.3f}  |  5-feat advantage: {adv_rich:+.3f}")
ax_c.legend(fontsize=9)
ax_c.grid(axis="y", alpha=0.4)

# ── Panel D: LATENT detection summary ────────────────────────────────────────
latent_methods  = []
latent_aurocs   = []
latent_colors   = []

if isinstance(ld.get("auroc"), float):
    latent_methods.append("Standard\nFeatures\n(L2 norm)")
    latent_aurocs.append(ld["auroc"])
    latent_colors.append("#2b7cbf")

if isinstance(pers.get("auroc_persona_direction"), float):
    latent_methods.append("Persona\nDirection\n(cosine)")
    latent_aurocs.append(pers["auroc_persona_direction"])
    latent_colors.append("#7d3c98")

if latent_methods:
    bars_d = ax_d.bar(latent_methods, latent_aurocs, color=latent_colors,
                       edgecolor="white", linewidth=0.8, width=0.45)
    for bar, v in zip(bars_d, latent_aurocs):
        ax_d.text(bar.get_x() + bar.get_width()/2,
                  v + 0.005, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax_d.axhline(0.5, color="#888888", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax_d.axhline(0.65, color="#7d3c98", linestyle=":", linewidth=1, alpha=0.5,
                 label="Detection threshold (0.65)")
    ax_d.set_ylim(0.4, min(1.0, max(latent_aurocs) + 0.15))
    ax_d.set_ylabel("AUROC (LATENT vs CORRECT)")
    n_lat = ld.get("n_latent", pers.get("n_latent", "?"))
    ax_d.set_title(f"(D) LATENT Failure Detection\n"
                   f"n_latent={n_lat} — invisible to single-step governance")
    ax_d.legend(fontsize=9)
    ax_d.grid(axis="y", alpha=0.4)
else:
    ax_d.text(0.5, 0.5, "Insufficient LATENT\nsamples for AUROC",
              ha="center", va="center", transform=ax_d.transAxes,
              fontsize=12, color="#888888")
    ax_d.set_title("(D) LATENT Failure Detection")

# ── Save Figure 1 ─────────────────────────────────────────────────────────────
for fmt in ("pdf", "png"):
    out = PAPER_DIR / f"fig1_main_4panel.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  → {out.name}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Context Ablation (Section 4.4)
# ═══════════════════════════════════════════════════════════════════════════════
if abl and "standard" in abl:
    print("Generating Figure 2 (context ablation)...")
    modes_fig2 = ["CORRECT", "AMPLIFIED", "CASCADE", "LATENT", "ISOLATED"]
    std_v   = [abl["standard"].get(m, 0) for m in modes_fig2]
    noctx_v = [abl["noctx"].get(m, 0)    for m in modes_fig2]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(modes_fig2))
    w = 0.35

    b1 = ax2.bar(x - w/2, std_v,   w, label="Standard (context-conditioned)",
                 color="#2b7cbf", edgecolor="white")
    b2 = ax2.bar(x + w/2, noctx_v, w, label="Context-free (ablation)",
                 color="#d4701a", edgecolor="white")

    for bar, v in list(zip(b1, std_v)) + list(zip(b2, noctx_v)):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(modes_fig2)
    ax2.set_ylabel("Frequency (%)")
    drop = abl.get("amplified_drop_pct", 0)
    p4   = abl.get("p4_confirmed", False)
    ax2.set_title(f"Context Ablation: Standard vs Context-Free Step 2\n"
                  f"AMPLIFIED rate drop = {drop:+.1f}pp  |  "
                  f"P4: {'Confirmed' if p4 else 'Refuted'}  |  "
                  f"n_std={abl['n_standard']}, n_noctx={abl['n_noctx']}")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.4)

    # Annotate mechanism
    mech_short = abl.get("mechanism", "").split(":")[0]
    ax2.text(0.97, 0.97, f"Mechanism:\n{mech_short}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                       edgecolor="#d4701a", alpha=0.9))

    for fmt in ("pdf", "png"):
        fig2.savefig(PAPER_DIR / f"fig2_ablation.{fmt}", dpi=300,
                     bbox_inches="tight", facecolor="white")
        print(f"  → fig2_ablation.{fmt}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Supplementary: Full trajectory table as heatmap
# ═══════════════════════════════════════════════════════════════════════════════
if traj_mode:
    print("Generating Figure 3 (trajectory heatmap — supplementary)...")

    all_modes   = list(traj_mode.keys())
    step_labels = ["Step 1 (Initial)", "Step 2 (Verify)", "Step 3 (Final)"]
    step_keys   = ["step1", "step2", "step3"]

    matrix = np.array([
        [traj_mode[m][sk]["mean"] for sk in step_keys]
        for m in all_modes
    ])

    # Normalize per-mode to show within-mode trajectory shape
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    matrix_norm = (matrix - row_min) / (row_max - row_min + 1e-10)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))

    # Raw values heatmap
    im1 = ax3a.imshow(matrix, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax3a.set_xticks(range(3))
    ax3a.set_xticklabels(step_labels, rotation=20, ha="right")
    ax3a.set_yticks(range(len(all_modes)))
    ax3a.set_yticklabels(all_modes)
    plt.colorbar(im1, ax=ax3a, shrink=0.8)
    ax3a.set_title("Fisher Proxy: Raw Mean Values")

    for i in range(len(all_modes)):
        for j in range(3):
            ax3a.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                      fontsize=8, color="black")

    # Normalized trajectory (shape)
    im2 = ax3b.imshow(matrix_norm, aspect="auto", cmap="RdYlGn",
                       interpolation="nearest", vmin=0, vmax=1)
    ax3b.set_xticks(range(3))
    ax3b.set_xticklabels(step_labels, rotation=20, ha="right")
    ax3b.set_yticks(range(len(all_modes)))
    ax3b.set_yticklabels(all_modes)
    plt.colorbar(im2, ax=ax3b, shrink=0.8, label="Normalized (0=min, 1=max per mode)")
    ax3b.set_title("Fisher Proxy: Normalized Trajectory Shape\n(reveals within-mode patterns)")

    fig3.suptitle("Supplementary: Fisher Geometry Trajectory by Failure Mode", fontsize=12)
    plt.tight_layout()

    for fmt in ("pdf", "png"):
        fig3.savefig(PAPER_DIR / f"fig_supp_trajectory.{fmt}", dpi=300,
                     bbox_inches="tight", facecolor="white")
        print(f"  → fig_supp_trajectory.{fmt}")
    plt.close()


print(f"\nAll paper figures saved to: {PAPER_DIR}")
print("\nFiles for paper submission:")
for f in sorted(PAPER_DIR.glob("*.pdf")):
    print(f"  {f.name}")
print("\nFor README/preprint (PNG versions):")
for f in sorted(PAPER_DIR.glob("*.png")):
    print(f"  {f.name}")
