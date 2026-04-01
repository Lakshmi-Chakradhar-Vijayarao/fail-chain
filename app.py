"""
FAIL-CHAIN: Geometric Signatures of Failure Propagation in Multi-Step LLM Pipelines
=====================================================================================
Research dashboard — walks through the complete FAIL-CHAIN pipeline:
failure taxonomy, Fisher geometry trajectory, cascade prediction, and
pre-registered prediction outcomes.

Run:   streamlit run app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAIL-CHAIN · Failure Propagation Geometry",
    page_icon="⛓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared style (same design system as HaRP / GEOM-PROOF) ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary:    #b71c1c;
    --primary-lt: #c62828;
    --primary-bg: #ffebee;
    --green:      #2e7d32;
    --green-bg:   #e8f5e9;
    --amber:      #bf360c;
    --amber-bg:   #fbe9e7;
    --red:        #b71c1c;
    --red-bg:     #ffebee;
    --teal:       #00695c;
    --teal-bg:    #e0f2f1;
    --purple:     #4527a0;
    --purple-bg:  #ede7f6;
    --slate:      #37474f;
    --slate-bg:   #eceff1;
    --bg:         #fdf5f5;
    --surface:    #ffffff;
    --text:       #1a0a0a;
    --text-muted: #6b7280;
    --border:     #f0d8d8;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.finding-box {
    background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
    border-left: 5px solid var(--primary);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(183,28,28,0.10);
    font-size: 0.93rem;
}
.good-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 5px solid var(--green);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(46,125,50,0.10);
    font-size: 0.93rem;
}
.warn-box {
    background: linear-gradient(135deg, #fbe9e7 0%, #fff8e1 100%);
    border-left: 5px solid var(--amber);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(191,54,12,0.10);
    font-size: 0.93rem;
}
.null-box {
    background: linear-gradient(135deg, #eceff1 0%, #f5f5f5 100%);
    border-left: 5px solid var(--slate);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(55,71,79,0.08);
    font-size: 0.93rem;
}
.analogy-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #e0f2f1 100%);
    border-left: 5px solid var(--teal);
    padding: 0.9rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    font-style: italic;
    box-shadow: 0 2px 8px rgba(0,105,92,0.10);
    font-size: 0.90rem;
}
.limit-box {
    background: linear-gradient(135deg, #eceff1 0%, #f5f5f5 100%);
    border-left: 5px solid var(--slate);
    padding: 1.0rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(55,71,79,0.08);
    font-size: 0.91rem;
}

.step-pill {
    display: inline-block;
    background: linear-gradient(90deg, #b71c1c 0%, #c62828 100%);
    color: white;
    padding: 0.28rem 0.9rem;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 6px rgba(183,28,28,0.28);
}
.exp-pill {
    display: inline-block;
    background: linear-gradient(90deg, #283593 0%, #3949ab 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    margin-right: 0.3rem;
    box-shadow: 0 2px 6px rgba(40,53,147,0.25);
}
.novel-pill {
    display: inline-block;
    background: linear-gradient(90deg, #00695c 0%, #00897b 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(0,105,92,0.25);
}
.honest-pill {
    display: inline-block;
    background: linear-gradient(90deg, #2e7d32 0%, #388e3c 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(46,125,50,0.25);
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(183,28,28,0.07);
    transition: box-shadow 0.2s, transform 0.15s;
}
.stat-card:hover {
    box-shadow: 0 8px 28px rgba(183,28,28,0.13);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-number-red    { color: var(--red); }
.stat-number-green  { color: var(--green); }
.stat-number-indigo { color: #283593; }
.stat-number-amber  { color: var(--amber); }
.stat-number-purple { color: var(--purple); }
.stat-label {
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.35rem;
    line-height: 1.3;
}
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, #3e0000 0%, #b71c1c 100%);
    color: #ffe8e8;
    border-radius: 8px;
    padding: 0.38rem 1rem;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(183,28,28,0.22);
}
.feature-chip {
    display: inline-block;
    background: var(--primary-bg);
    color: var(--primary);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
    font-family: 'SFMono-Regular', Consolas, monospace;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0000 0%, #3e0000 40%, #7b1212 75%, #b71c1c 100%);
}
section[data-testid="stSidebar"] * { color: #ffe8e8 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #ffe8e8 !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 0.35rem 0.7rem !important;
    margin-bottom: 0.1rem !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
h1 { color: #3e0000; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; }
h2 { color: #b71c1c; font-weight: 700; }
h3 { color: #00695c; font-weight: 600; }
h1::after {
    content: '';
    display: block;
    width: 56px;
    height: 4px;
    background: linear-gradient(90deg, #b71c1c, #ef5350);
    border-radius: 2px;
    margin-top: 0.4rem;
}
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
LOGS_DIR  = _HERE / "results" / "logs"
PLOTS_DIR = _HERE / "results" / "plots"
PAPER_DIR = PLOTS_DIR / "paper"

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_json(name):
    p = LOGS_DIR / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

@st.cache_data
def load_pipeline_data():
    return load_json("pipeline_data.json")

def plot_img(path, caption=None, width=None):
    p = Path(path) if isinstance(path, str) else path
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=(width is None))
    else:
        st.info(f"Plot not yet generated: {p.name}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def pill(text):
    st.markdown(f'<span class="step-pill">{text}</span>', unsafe_allow_html=True)

def exp_pill(text):
    st.markdown(f'<span class="exp-pill">{text}</span>', unsafe_allow_html=True)

def novel_pill(text):
    st.markdown(f'<span class="novel-pill">{text}</span>', unsafe_allow_html=True)

def finding(text):
    st.markdown(f'<div class="finding-box">{text}</div>', unsafe_allow_html=True)

def good(text):
    st.markdown(f'<div class="good-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def null(text):
    st.markdown(f'<div class="null-box">{text}</div>', unsafe_allow_html=True)

def analogy(text):
    st.markdown(f'<div class="analogy-box"><b>Analogy:</b> {text}</div>', unsafe_allow_html=True)

def limit(text):
    st.markdown(f'<div class="limit-box"><b>⚠ Limitation:</b> {text}</div>', unsafe_allow_html=True)

def honest_pill():
    st.markdown('<span class="honest-pill">✓ PRE-REGISTERED</span>', unsafe_allow_html=True)

def stat(number, label, color=""):
    st.markdown(
        f'<div class="stat-card"><div class="stat-number {color}">{number}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def phase(text):
    st.markdown(f'<div class="phase-badge">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_brief():
    st.title("Research Brief — Why Failure Propagation Matters")
    st.markdown(
        "##### For decision-makers interested in trustworthy AI pipelines — "
        "no prior technical knowledge required."
    )

    st.markdown("---")
    st.markdown("## The Problem in One Sentence")
    finding(
        "Modern AI systems don't answer questions in a single step — they reason, verify, "
        "and refine. A hallucination at step 1 doesn't stay at step 1. It becomes the "
        "<b>context</b> for steps 2 and 3 — and current governance systems are blind to this."
    )

    st.markdown("## What This Project Does")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### The Pipeline")
        st.markdown("""
        We study a 3-step self-refinement pipeline on Qwen 2.5 3B over 600 TruthfulQA questions:

        1. **Step 1 — Initial Answer:** Model answers the question
        2. **Step 2 — Self-Verification:** Model reviews its own answer
        3. **Step 3 — Final Answer:** Model produces a refined final answer

        At each step, we measure the model's **hidden-state geometry** at layer 32
        (identified in HaRP as the optimal detection layer).
        """)
    with col2:
        st.markdown("### The Question")
        st.markdown("""
        > *Can we detect a cascade failure **before** the final answer is generated?*

        If Fisher separability geometry at steps 1+2 predicts step 3 failure,
        we have an **early warning signal** — a way to route or abort before the
        wrong answer is delivered.

        If it can't, the failure is **geometry-invisible** — motivating GUARDIAN's
        formal multi-step bounds.

        **Both answers are publishable. We pre-registered which we expected.**
        """)

    st.markdown("## The Key Finding")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        stat("92.3%", "CASCADE rate", "stat-number-red")
    with col_b:
        stat("0.662", "Early AUROC", "stat-number-indigo")
    with col_c:
        stat("+0.053", "Advantage vs late", "stat-number-amber")
    with col_d:
        stat("MODEL\nCOLLAPSE", "P4 mechanism", "stat-number-purple")

    good(
        "<b>P1 CONFIRMED:</b> CASCADE + LATENT account for <b>95.4%</b> of all non-correct samples — "
        "self-refinement is not self-correcting for the vast majority of failure cases. "
        "This directly motivates cascade-aware governance in GUARDIAN."
    )
    warn(
        "<b>P2 BORDERLINE:</b> Early-step geometry predicts final failure with AUROC <b>0.662</b> — "
        "only a <b>+0.053</b> advantage over late detection (0.610). The signal exists but is weak, "
        "suggesting CASCADE dominance may suppress inter-mode geometric variation."
    )
    finding(
        "<b>P4 REFUTED:</b> Context-free ablation shows AMPLIFIED failures <i>increase</i> when step 1 "
        "context is removed (+0.7pp vs predicted −5pp). Cascade is intrinsic to model behavior "
        "(<b>MODEL_BEHAVIOR_COLLAPSE</b>), not driven by context pollution. "
        "GUARDIAN should ABSTAIN, not REGENERATE, for AMPLIFIED-class queries."
    )

    st.markdown("## Connection to the PhD Arc")
    st.markdown("""
    | Project | Contribution |
    |---|---|
    | MECH-INT (1) | Located hallucination signal at L8-L9 in GPT-2 — causal, FFN over-retrieval |
    | HaRP (2) | L32 at 89% depth carries hallucination signal in Qwen 2.5 3B; AUROC 0.775 |
    | GEOM-PROOF (3) | Fisher J bounds AUROC within 1.5%; conformal coverage α*=0.07 |
    | **FAIL-CHAIN (4)** | **Geometry trajectory through 3-step pipeline; 8-class failure taxonomy** |
    | GUARDIAN (5) | Uses FAIL-CHAIN cascade risk + persona direction for pipeline-aware routing |
    """)


def page_overview():
    st.title("Project Overview")
    exp_pill("EXP 00") ; exp_pill("EXP 01")
    st.markdown("**Infrastructure validation + 600-sample pipeline generation on Kaggle T4 GPU**")
    st.markdown("---")

    st.markdown("## Pipeline Architecture")
    st.code("""
Question ──► [STEP 1: Initial Answer]    ──► Extract H_L32_s1, ROUGE label_s1
                     │
                     ▼
            [STEP 2: Self-Verification]  ──► Extract H_L32_s2, ROUGE label_s2
                     │
                     ▼
            [STEP 3: Final Answer]       ──► Extract H_L32_s3, ROUGE label_s3

Fisher proxy at each step = L2 norm of hidden state at L32 (last generated token)
    """, language="text")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Experimental Setup")
        st.markdown("""
        - **Model:** Qwen 2.5 3B Instruct (36 layers, 2048 hidden dim)
        - **Layer:** L32 (89% depth — optimal per HaRP)
        - **Dataset:** TruthfulQA, n=600 pipeline + n=150 context-free ablation
        - **Threshold:** ROUGE-L < 0.3 → hallucinated
        - **Boundary zone:** ±0.05 around threshold (excluded from LATENT/CORRECT stats)
        - **GPU:** Kaggle T4 (~4.5h for 600 samples)
        """)
    with col2:
        st.markdown("### What We Measure")
        st.markdown("""
        At each of the 3 steps, for each sample:
        - Hidden state **H_L32** (last generated token, shape 2048)
        - **Fisher proxy** = L2 norm of H_L32
        - **ROUGE-L label** (correct/hallucinated)
        - **Failure mode** (8-class taxonomy)

        Across samples:
        - **Fisher trajectory** = mean proxy per mode across steps 1→2→3
        - **Cascade prediction AUROC** (early steps vs late step)
        - **Context ablation** (step 2 with/without step 1 context)
        """)

    st.markdown("## Key Design Choices")
    analogy(
        "Why self-refinement and not RAG? Self-refinement is fully controlled (one model, no retrieval "
        "randomness), realistic (deployed agentic systems use it), and lets us isolate geometry at each "
        "step cleanly. RAG adds retrieval noise that confounds the geometry signal."
    )
    limit(
        "ROUGE-L as ground truth has well-known limitations — it measures surface-form word overlap, "
        "not factual accuracy. A semantically correct paraphrase may score below 0.3. "
        "See GEOM-PROOF Exp 07 (LLM-as-judge Cohen's κ = -0.010) for context on this limitation."
    )


def page_taxonomy():
    st.title("8-Class Failure Taxonomy")
    exp_pill("EXP 01") ; exp_pill("EXP 02")
    st.markdown("**Formal definitions for all 8 failure modes — pre-registered before experiments**")
    st.markdown("---")

    honest_pill()
    st.markdown("All failure mode definitions were committed in `PREDICTION.md` before any GPU run.")

    st.markdown("## Taxonomy Definition")

    MODES = [
        ("CORRECT",       "#2d8a4e", "correct_s1 ∧ correct_s2 ∧ correct_s3",
         "All 3 steps correct — baseline", "No intervention needed", 16, 2.7),
        ("CASCADE",       "#c0392b", "wrong_s1 ∧ wrong_s2 ∧ wrong_s3",
         "Error propagates through entire pipeline", "Must catch at step 1; ABSTAIN recommended", 554, 92.3),
        ("LATE_RECOVERY", "#5dade2", "wrong_s1 ∧ wrong_s2 ∧ correct_s3",
         "Errors at steps 1+2, pipeline self-heals at step 3", "Watchful wait — recovery may succeed", 14, 2.3),
        ("SELF_HEAL",     "#888888", "correct_s1 ∧ wrong_s2 ∧ correct_s3",
         "Step 2 introduces error, step 3 self-corrects", "Partial self-recovery observed", 6, 1.0),
        ("AMPLIFIED",     "#d4701a", "correct_s1 ∧ wrong_s2 ∧ wrong_s3",
         "Step 1 correct but verification step introduces error", "ABSTAIN (context-stripping doesn't help — P4)", 4, 0.7),
        ("LATENT",        "#7d3c98", "correct_s1 ∧ correct_s2 ∧ wrong_s3",
         "Intermediate steps appear correct — failure only at final output",
         "⚠ INVISIBLE to single-step governance", 3, 0.5),
        ("ISOLATED",      "#2b7cbf", "wrong_s1 ∧ correct_s2 ∧ correct_s3",
         "Error at step 1 — self-corrected by pipeline", "Self-correction works here", 2, 0.3),
        ("OSCILLATING",   "#e67e22", "wrong_s1 ∧ correct_s2 ∧ wrong_s3",
         "Error → recovery → error again", "Unstable pipeline — ABSTAIN", 1, 0.2),
    ]

    for mode, color, condition, description, governance, count, pct in MODES:
        border_style = "border-left: 5px solid " + color
        bg = "#fff8f8" if mode == "LATENT" else "#fafafa"
        st.markdown(
            f'<div style="{border_style};background:{bg};padding:0.9rem 1.2rem;'
            f'border-radius:0 8px 8px 0;margin:0.5rem 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:800;font-size:1.0rem;color:{color};">{mode}</span>'
            f'<span style="font-size:0.80rem;color:#666;font-family:monospace;">'
            f'n={count} ({pct}%)</span></div>'
            f'<div style="font-size:0.85rem;margin-top:0.3rem;color:#333;">'
            f'<b>Condition:</b> <code>{condition}</code></div>'
            f'<div style="font-size:0.83rem;color:#555;margin-top:0.2rem;">{description}</div>'
            f'<div style="font-size:0.80rem;color:#777;margin-top:0.15rem;">'
            f'<b>Governance:</b> {governance}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    finding(
        "<b>LATENT is the critical novel class.</b> Both intermediate step outputs pass the ROUGE-L "
        "correctness threshold, yet the final answer fails. A single-step governance system evaluating "
        "step 2 would declare this query SAFE — and be wrong. This class directly motivates "
        "GUARDIAN's cascade-aware routing architecture."
    )

    st.markdown("## Observed Distribution")
    plot_img(PLOTS_DIR / "failure_mode_distribution.png",
             caption="Failure mode distribution across 600 pipeline samples")


def page_trajectory():
    st.title("Fisher Geometry Trajectory")
    exp_pill("EXP 02")
    st.markdown("**Per-mode Fisher proxy (L2 norm at L32) trajectory through steps 1→2→3**")
    st.markdown("---")

    traj = load_json("fisher_trajectory.json")
    traj_mode = traj.get("trajectory_by_mode", {})

    if not traj_mode:
        st.warning("fisher_trajectory.json not loaded — run `python experiments/02_fisher_trajectory.py` first.")
        return

    st.markdown("## Fisher Proxy by Failure Mode")
    plot_img(PLOTS_DIR / "fisher_trajectory.png",
             caption="Fisher proxy (L2 norm at L32) across 3 pipeline steps, by failure mode")

    st.markdown("## Trajectory Table")
    rows = []
    for mode, td in traj_mode.items():
        rows.append({
            "Mode": mode,
            "Step 1 Mean": f"{td['step1']['mean']:.1f} ± {td['step1']['std']:.1f}",
            "Step 2 Mean": f"{td['step2']['mean']:.1f} ± {td['step2']['std']:.1f}",
            "Step 3 Mean": f"{td['step3']['mean']:.1f} ± {td['step3']['std']:.1f}",
            "Slope 1→2":   f"{td['slope_s1_s2']:+.1f}",
            "Slope 2→3":   f"{td['slope_s2_s3']:+.1f}",
            "n": td["n"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## Interpretation")
    col1, col2 = st.columns(2)
    with col1:
        good(
            "<b>CASCADE has the highest step 1→2 slope (+10.3)</b> compared to CORRECT (+6.2). "
            "This suggests CASCADE samples show greater geometric instability during the "
            "verification step — a weak but detectable early signal."
        )
    with col2:
        null(
            "<b>LATENT trajectory not computable</b> — only 3 LATENT samples in 600 (below 5-sample "
            "minimum). The theoretical prediction (LATENT geometry resembles CORRECT at steps 1+2, "
            "then diverges at step 3) could not be empirically verified. See VAR-6 in LOOSE_ENDS.md."
        )

    limit(
        "Only 4 of 8 modes had sufficient samples for trajectory analysis: CORRECT (16), "
        "CASCADE (554), LATE_RECOVERY (14), SELF_HEAL (6). The taxonomy's minority modes — "
        "LATENT (3), AMPLIFIED (4), ISOLATED (2), OSCILLATING (1) — are too rare on "
        "TruthfulQA factoid QA for per-mode statistics."
    )


def page_cascade_prediction():
    st.title("Cascade Prediction Signal")
    exp_pill("EXP 03")
    honest_pill()
    st.markdown("**Pre-registered P2: Can early-step geometry predict final-step failure?**")
    st.markdown("---")

    cp = load_json("cascade_prediction.json")
    if not cp:
        st.warning("cascade_prediction.json not found — run `python experiments/03_cascade_prediction.py` first.")
        return

    pred = cp.get("cascade_prediction", {})
    rich = cp.get("cascade_prediction_rich", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat(f"{pred.get('auroc_late', 0):.3f}", "Late AUROC\n(Step 3 only, 1 feat)", "stat-number-amber")
    with col2:
        stat(f"{pred.get('auroc_early', 0):.3f}", "Early AUROC\n(Steps 1+2, 3 feat)", "stat-number-indigo")
    with col3:
        stat(f"{rich.get('auroc_early_5feat', 0):.3f}", "Early+Cosine AUROC\n(5 feat)", "stat-number-green")
    with col4:
        adv = rich.get("advantage_5feat", 0)
        col_class = "stat-number-green" if adv >= 0.05 else "stat-number-amber"
        stat(f"{adv:+.3f}", "5-feat Advantage\nvs Late Detection", col_class)

    st.markdown("---")
    plot_img(PLOTS_DIR / "cascade_prediction_auroc.png",
             caption="Early warning vs late detection AUROC (3-feat and 5-feat early, 1-feat late)")

    st.markdown("## P2 Prediction Outcome")
    p2 = cp.get("prediction_checks", {}).get("P2_confirmed", False)
    adv_3feat = pred.get("advantage", 0)
    adv_5feat = rich.get("advantage_5feat", 0)

    if p2:
        good(f"<b>P2 CONFIRMED</b> — Early AUROC advantage {adv_3feat:+.3f} ≥ 0.05 threshold.")
    else:
        warn(
            f"<b>P2 REFUTED</b> (3-feat advantage {adv_3feat:+.3f} &lt; 0.05 threshold) / "
            f"<b>Borderline</b> (5-feat advantage {adv_5feat:+.3f} ≥ 0.05). "
            "The geometry signal exists but is weak — CASCADE dominance (92.3%) may suppress "
            "inter-mode geometric variation, compressing the separability advantage."
        )

    st.markdown("## Feature Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 3-Feature Matrix")
        st.markdown("""
        | Feature | Description |
        |---|---|
        | `j_s1` | Fisher proxy at step 1 (L2 norm of H_L32) |
        | `j_s2` | Fisher proxy at step 2 |
        | `Δ_{s1→s2}` | Proxy slope (step 2 − step 1) |
        """)
    with col2:
        st.markdown("### 5-Feature Matrix (Rich)")
        st.markdown("""
        | Feature | Description |
        |---|---|
        | `j_s1` | Fisher proxy at step 1 |
        | `j_s2` | Fisher proxy at step 2 |
        | `Δ_{s1→s2}` | Proxy slope |
        | `cosine(h_s1, h_s2)` | Geometric drift direction |
        | `L2(h_s2 − h_s1)` | Drift magnitude |

        Cosine feature gain: **{:.3f}** (5-feat vs 3-feat)
        """.format(rich.get("advantage_5feat", 0) - pred.get("advantage", 0)))

    st.markdown("### Evaluation Protocol")
    good(
        "All AUROC values computed with <b>5-fold stratified cross-validation</b> using "
        "<b>sklearn Pipeline</b> (StandardScaler → LogisticRegression) to prevent data leakage "
        "from scaler fitting on full dataset. Random seed 42 for reproducibility."
    )


def page_latent_detection():
    st.title("Latent Failure Detection")
    exp_pill("EXP 03")
    honest_pill()
    st.markdown("**Pre-registered P3 and P5: Can geometry detect LATENT failures before step 3?**")
    st.markdown("---")

    cp = load_json("cascade_prediction.json")
    ld = cp.get("latent_detection", {})
    pers = cp.get("persona_direction", {})

    finding(
        "<b>LATENT is the most dangerous failure mode.</b> Both step 1 and step 2 outputs "
        "pass the ROUGE-L correctness threshold — yet the final answer at step 3 is wrong. "
        "A single-step governance system sees two correct outputs and declares the query SAFE. "
        "Only step-3 evaluation catches it — after the wrong answer has already been generated."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### P3 — Fisher Proxy Comparison")
        st.markdown("""
        **Prediction:** LATENT samples will show lower mean Fisher proxy at step 2 than
        CORRECT samples, despite passing the ROUGE-L threshold.

        **Rationale:** The model's representation at step 2 may be near the decision boundary
        even when the surface label calls it correct.
        """)
        n_latent = ld.get("n_latent", 3)
        null(
            f"<b>P3 INSUFFICIENT SAMPLES</b> — only <b>{n_latent}</b> LATENT samples "
            f"(need ≥10 for AUROC). After boundary exclusion (±0.05 ROUGE zone), 0 clean "
            f"LATENT samples remain. P3 cannot be evaluated on this run."
        )
    with col2:
        st.markdown("### P5 — Persona Direction")
        st.markdown("""
        **Prediction:** A linear direction extracted from step-1 hidden states
        (trained on step-3 outcome labels `y_s3`) will detect LATENT failures at
        step 2 with AUROC ≥ 0.60.

        **Direction quality (diagnostic):** Step-3 AUROC on all 600 samples =
        **{:.3f}** — confirms h_s1 carries eventual-failure signal.
        """.format(pers.get("direction_s3_auroc", 0)))
        null(
            f"<b>P5 INSUFFICIENT SAMPLES</b> — only <b>{pers.get('n_latent', 3)}</b> LATENT "
            f"samples. The persona direction was extracted and saved as "
            f"`results/logs/hallucination_direction.npy` for GUARDIAN, but the primary "
            f"LATENT-vs-CORRECT AUROC cannot be estimated with n=3."
        )

    st.markdown("---")
    st.markdown("## Hallucination Direction — Saved for GUARDIAN")
    good(
        "The persona direction vector (shape: 2048, unit-normalized) is saved at "
        "<code>results/logs/hallucination_direction.npy</code>. "
        "GUARDIAN's step-2 real-time monitor computes "
        "<code>cosine_alarm = h_s2 @ direction</code> as a &lt;1ms inference-time signal. "
        "This is FAIL-CHAIN's primary contribution to GUARDIAN's design."
    )

    warn(
        "<b>Why only 3 LATENT samples?</b> TruthfulQA is adversarially designed to elicit "
        "step-3 failures — but its factoid QA format means the model either fails immediately "
        "(CASCADE, 92.3%) or doesn't fail at all (CORRECT, 2.7%). The intermediate failure modes "
        "that LATENT requires (steps 1+2 correct, step 3 wrong) are extremely rare. "
        "Open-ended reasoning tasks (VAR-6 in LOOSE_ENDS.md) should produce more LATENT samples."
    )


def page_ablation():
    st.title("Context Ablation — P4")
    exp_pill("EXP 03")
    honest_pill()
    st.markdown("**Pre-registered P4: Is cascade caused by context contamination or model behavior collapse?**")
    st.markdown("---")

    cp = load_json("cascade_prediction.json")
    abl = cp.get("context_ablation", {})

    if not abl:
        st.warning("Context ablation data not found.")
        return

    st.markdown("## Experimental Design")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Standard Pipeline (n=150)**
        - Step 2 receives step 1 output as context
        - Standard STEP2_PROMPT with `{step1_answer}`
        """)
    with col2:
        st.markdown("""
        **Context-Free Pipeline (n=150)**
        - Step 2 does **NOT** receive step 1 output
        - STEP2_NOCTX_PROMPT — isolated self-verification
        - Tests SCoRe (Kumar et al., NeurIPS 2024) mechanism
        """)

    st.markdown("## Results")
    plot_img(PLOTS_DIR / "context_ablation_comparison.png",
             caption="Standard vs context-free pipeline: failure mode frequency comparison")

    std  = abl.get("standard", {})
    noctx = abl.get("noctx", {})
    modes_abl = ["CORRECT", "AMPLIFIED", "CASCADE", "LATENT", "ISOLATED"]

    rows = []
    for m in modes_abl:
        s = std.get(m, 0)
        n = noctx.get(m, 0)
        rows.append({"Mode": m, "Standard %": f"{s:.2f}%", "Context-Free %": f"{n:.2f}%",
                     "Change": f"{n-s:+.2f}pp"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    drop = abl.get("amplified_drop_pct", 0)
    p4   = abl.get("p4_confirmed", False)

    st.markdown("## P4 Prediction Outcome")
    if p4:
        good(f"<b>P4 CONFIRMED</b> — AMPLIFIED dropped {abs(drop):.1f}pp ≥ 5pp threshold.")
    else:
        finding(
            f"<b>P4 REFUTED</b> — AMPLIFIED <i>increased</i> by {abs(drop):.1f}pp in context-free "
            f"condition (expected drop ≥ 5pp). AMPLIFIED failures are <b>not</b> caused by context "
            f"contamination from step 1. The cascade is intrinsic to the model's behavior "
            f"(<b>MODEL_BEHAVIOR_COLLAPSE</b>)."
        )

    st.markdown("## Implication for GUARDIAN")
    good(
        "<b>GUARDIAN routing consequence:</b> Since context-stripping (REGENERATE_NOCTX) "
        "does not reduce AMPLIFIED failures, GUARDIAN should route AMPLIFIED-class queries to "
        "<b>ABSTAIN</b> rather than REGENERATE. The model cannot fix its own cascade by "
        "seeing less context — the failure is architectural, not contextual."
    )

    analogy(
        "A doctor who gives a wrong diagnosis doesn't improve by being shown less patient history — "
        "the error is in the reasoning, not the context. Same for AMPLIFIED cascade failures."
    )


def page_predictions():
    st.title("Pre-Registered Prediction Outcomes")
    honest_pill()
    st.markdown("**All 5 predictions committed before any GPU run — results reported honestly**")
    st.markdown("---")

    cp = load_json("cascade_prediction.json")
    checks = cp.get("prediction_checks", {})

    PREDICTIONS = [
        ("P1", "CASCADE + LATENT ≥ 30% of non-correct",
         checks.get("P1_confirmed", True), "CONFIRMED",
         f"(CASCADE=554 + LATENT=3) / (600−16) = {checks.get('P1_cascade_latent_pct_noncorrect', 95.4):.1f}% >> 30%",
         "green"),
        ("P2", "Early AUROC advantage ≥ 0.05 (steps 1+2 vs step 3 only)",
         checks.get("P2_confirmed", False), "REFUTED / BORDERLINE",
         "3-feat advantage = +0.042 (< 0.05) | 5-feat advantage = +0.053 (borderline ≥ 0.05). "
         "Signal exists but is weak.",
         "amber"),
        ("P3", "LATENT Fisher proxy at step 2 < CORRECT proxy",
         None, "INSUFFICIENT SAMPLES",
         "Only 3 LATENT samples (need ≥10). All 3 are boundary-zone samples. "
         "Cannot evaluate.",
         "slate"),
        ("P4", "AMPLIFIED drops ≥ 5pp in context-free ablation",
         checks.get("P4_confirmed", False), "REFUTED",
         "AMPLIFIED increased by +0.7pp (opposite direction). "
         "Mechanism: MODEL_BEHAVIOR_COLLAPSE — cascade is intrinsic.",
         "red"),
        ("P5", "Persona direction AUROC ≥ 0.60 for LATENT detection",
         checks.get("P5_confirmed", None), "INSUFFICIENT SAMPLES",
         "Only 3 LATENT samples. Direction extracted and saved for GUARDIAN "
         "(direction quality AUROC = 0.648 on all samples), but primary LATENT-vs-CORRECT "
         "AUROC cannot be estimated.",
         "slate"),
    ]

    for pid, desc, confirmed, status, evidence, color in PREDICTIONS:
        color_map = {
            "green": ("#e8f5e9", "#2e7d32"),
            "amber": ("#fff8e1", "#bf360c"),
            "red":   ("#ffebee", "#b71c1c"),
            "slate": ("#eceff1", "#37474f"),
        }
        bg, border = color_map.get(color, ("#f9f9f9", "#999"))
        icon = "✅" if confirmed is True else ("❌" if confirmed is False else "⬜")
        st.markdown(
            f'<div style="background:{bg};border-left:5px solid {border};'
            f'padding:1rem 1.3rem;border-radius:0 8px 8px 0;margin:0.6rem 0;">'
            f'<div style="font-weight:800;font-size:1.0rem;color:{border};">'
            f'{icon} {pid} — {status}</div>'
            f'<div style="font-size:0.88rem;font-style:italic;margin:0.3rem 0;color:#444;">'
            f'{desc}</div>'
            f'<div style="font-size:0.83rem;color:#555;">{evidence}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    good(
        "<b>Scientific integrity note:</b> These 5 predictions were committed to git "
        "(<code>PREDICTION.md</code>) before the Kaggle GPU run (Exp 01) started. "
        "The timestamp on the commit proves predictions precede results. "
        "Negative results are reported honestly as findings, not hidden as failures. "
        "P2 REFUTED, P4 REFUTED, P3/P5 INSUFFICIENT — all documented."
    )


def page_paper_figures():
    st.title("Publication-Quality Figures")
    exp_pill("EXP 05")
    st.markdown("**Paper-ready figures for NeurIPS 2026 workshop / ICLR 2027 submission**")
    st.markdown("---")

    st.markdown("## Figure 1 — Main 4-Panel Result")
    plot_img(PAPER_DIR / "fig1_main_4panel.png",
             caption="Fig 1: (A) Failure mode distribution, (B) Fisher proxy trajectory, "
                     "(C) Early vs late AUROC, (D) LATENT detection status")

    st.markdown("## Figure 2 — Context Ablation")
    plot_img(PAPER_DIR / "fig2_ablation.png",
             caption="Fig 2: Standard vs context-free pipeline failure rates (P4 test)")

    st.markdown("## Figure S1 — Supplementary: Full Trajectory Heatmap")
    plot_img(PAPER_DIR / "fig_supp_trajectory.png",
             caption="Fig S1: Fisher proxy trajectory heatmap (raw + normalized) for all analyzed modes")

    st.markdown("---")
    st.markdown("## Submission Files")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**For paper body (PDF):**")
        for f in sorted(PAPER_DIR.glob("*.pdf")):
            st.markdown(f"- `{f.name}`")
    with col2:
        st.markdown("**For README / preprint (PNG):**")
        for f in sorted(PAPER_DIR.glob("*.png")):
            st.markdown(f"- `{f.name}`")


def page_full_results():
    st.title("Full Results")
    st.markdown("**Complete results log — all numbers in one place**")
    st.markdown("---")

    cp = load_json("cascade_prediction.json")
    ft = load_json("fisher_trajectory.json")

    if not cp or not ft:
        st.warning("Run all experiments first (Exp 02–05).")
        return

    st.markdown("## Failure Mode Distribution")
    fmd = ft.get("failure_mode_distribution", {})
    total = sum(fmd.values())
    if fmd:
        rows = [{"Mode": m, "Count": c, "Percent": f"{c/total*100:.1f}%"}
                for m, c in sorted(fmd.items(), key=lambda x: -x[1])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("## Cascade Prediction AUROC")
    pred = cp.get("cascade_prediction", {})
    rich = cp.get("cascade_prediction_rich", {})
    rows = [
        {"Method": "Late detection (step 3 only, 1 feat)", "AUROC": f"{pred.get('auroc_late', 0):.3f}", "Advantage": "—"},
        {"Method": "Early warning (steps 1+2, 3 feat)", "AUROC": f"{pred.get('auroc_early', 0):.3f}",
         "Advantage": f"{pred.get('advantage', 0):+.3f}"},
        {"Method": "Early + cosine drift (5 feat)", "AUROC": f"{rich.get('auroc_early_5feat', 0):.3f}",
         "Advantage": f"{rich.get('advantage_5feat', 0):+.3f}"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("## Context Ablation")
    abl = cp.get("context_ablation", {})
    std = abl.get("standard", {}); noctx = abl.get("noctx", {})
    if std and noctx:
        rows = [{"Mode": m, "Standard %": f"{std.get(m, 0):.2f}%",
                 "Context-Free %": f"{noctx.get(m, 0):.2f}%",
                 "Change": f"{noctx.get(m, 0) - std.get(m, 0):+.2f}pp"}
                for m in ["CORRECT", "AMPLIFIED", "CASCADE", "LATENT", "ISOLATED"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(f"**Mechanism:** `{abl.get('mechanism', 'N/A')}`")

    st.markdown("## Pre-Registered Prediction Summary")
    checks = cp.get("prediction_checks", {})
    rows = [
        {"Prediction": "P1: CASCADE+LATENT ≥ 30% non-correct",
         "Result": f"{checks.get('P1_cascade_latent_pct_noncorrect', 95.4):.1f}%",
         "Status": "✅ CONFIRMED"},
        {"Prediction": "P2: Early AUROC advantage ≥ 0.05",
         "Result": f"{pred.get('advantage', 0):+.3f} (3-feat) / {rich.get('advantage_5feat', 0):+.3f} (5-feat)",
         "Status": "❌ REFUTED / borderline"},
        {"Prediction": "P3: LATENT step 2 proxy < CORRECT",
         "Result": "N/A (n=3 LATENT)", "Status": "⬜ INSUFFICIENT"},
        {"Prediction": "P4: AMPLIFIED drops ≥ 5pp context-free",
         "Result": f"{abl.get('amplified_drop_pct', 0):+.1f}pp", "Status": "❌ REFUTED"},
        {"Prediction": "P5: Persona direction AUROC ≥ 0.60",
         "Result": "N/A (n=3 LATENT)", "Status": "⬜ INSUFFICIENT"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("## Output Files")
    st.markdown("""
    | File | Description |
    |---|---|
    | `results/logs/pipeline_data.json` | 600 pipeline samples (hidden states + labels + failure modes) |
    | `results/logs/ablation_data.json` | 150 context-free ablation samples |
    | `results/hidden_states/pipeline_hidden_states.npz` | H_L32 arrays shape (600, 2048) × 3 steps |
    | `results/logs/fisher_trajectory.json` | Per-mode Fisher proxy means/stds/slopes |
    | `results/logs/cascade_prediction.json` | All AUROC values + prediction_checks |
    | `results/logs/hallucination_direction.npy` | Persona direction vector (2048,) for GUARDIAN |
    | `results/plots/paper/fig1_main_4panel.pdf` | Main paper figure (PDF) |
    | `results/plots/paper/fig2_ablation.pdf` | Context ablation figure (PDF) |
    | `results/plots/paper/fig_supp_trajectory.pdf` | Supplementary trajectory heatmap (PDF) |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "📋 Research Brief":           page_brief,
    "Overview":                    page_overview,
    "Failure Taxonomy":            page_taxonomy,
    "Fisher Trajectory":           page_trajectory,
    "Cascade Prediction (P2)":     page_cascade_prediction,
    "Latent Detection (P3 / P5)":  page_latent_detection,
    "Context Ablation (P4)":       page_ablation,
    "Pre-Registered Predictions":  page_predictions,
    "Paper Figures":               page_paper_figures,
    "Full Results":                page_full_results,
}

SECTIONS = {
    "Start Here":        ["📋 Research Brief"],
    "Pipeline & Data":   ["Overview", "Failure Taxonomy"],
    "Results":           ["Fisher Trajectory", "Cascade Prediction (P2)",
                          "Latent Detection (P3 / P5)", "Context Ablation (P4)"],
    "Evaluation":        ["Pre-Registered Predictions"],
    "Output":            ["Paper Figures", "Full Results"],
}

with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.3rem;">
      <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.01em;color:#ffe8e8;">
        ⛓ FAIL-CHAIN</div>
      <div style="font-size:0.72rem;opacity:0.65;color:#ffcccc;margin-top:0.1rem;">
        Failure Propagation Geometry in LLM Pipelines</div>
      <div style="font-size:0.75rem;color:#ffcccc;margin-top:0.35rem;opacity:0.85;">
        Lakshmi Chakradhar Vijayarao</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:0.75rem 0.9rem;margin:0.6rem 0;">
      <div style="font-size:0.70rem;font-weight:700;opacity:0.6;text-transform:uppercase;
      letter-spacing:0.09em;margin-bottom:0.5rem;">Model &amp; Dataset</div>
      <div style="font-size:0.80rem;color:#ffd4d4;line-height:1.7;">
        <b>Model:</b> Qwen 2.5 3B Instruct<br>
        <b>Arch:</b> 36 layers · 2048 hidden dim<br>
        <b>Layer:</b> L32 (89% depth)<br>
        <b>Dataset:</b> TruthfulQA<br>
        <b>Samples:</b> 600 pipeline + 150 ablation<br>
        <b>Experiments:</b> 6 (Exp 00–05)
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:0.75rem 0.9rem;margin:0.4rem 0 0.8rem;">
      <div style="font-size:0.70rem;font-weight:700;opacity:0.6;text-transform:uppercase;
      letter-spacing:0.09em;margin-bottom:0.5rem;">Key Results</div>
      <div style="font-size:0.80rem;color:#ffd4d4;line-height:1.7;">
        <b>CASCADE rate:</b> 92.3% (554/600)<br>
        <b>Early AUROC (5-feat):</b> 0.662<br>
        <b>Late AUROC:</b> 0.610<br>
        <b>P1:</b> ✅ CONFIRMED (95.4%)<br>
        <b>P2:</b> ❌ REFUTED / borderline<br>
        <b>P4 mechanism:</b> MODEL_BEHAVIOR_COLLAPSE
      </div>
    </div>
    """, unsafe_allow_html=True)

    for section, page_list in SECTIONS.items():
        icon = {"Start Here": "📋", "Pipeline & Data": "🔬",
                "Results": "📊", "Evaluation": "✅", "Output": "📁"}.get(section, "•")
        st.markdown(
            f'<div style="font-size:0.70rem;font-weight:700;opacity:0.50;'
            f'text-transform:uppercase;letter-spacing:0.10em;margin:0.7rem 0 0.3rem;">'
            f'{icon} {section}</div>',
            unsafe_allow_html=True,
        )
        for pname in page_list:
            if st.button(pname, key=f"nav_{pname}", use_container_width=True):
                st.session_state["page"] = pname

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.70rem;opacity:0.45;text-align:center;padding:0.3rem 0;">
      Lakshmi Chakradhar Vijayarao · FAIL-CHAIN · 2026<br>
      Project 4 of 5 in the PhD arc
    </div>
    """, unsafe_allow_html=True)

# ── Route ──────────────────────────────────────────────────────────────────────
active = st.session_state.get("page", "📋 Research Brief")
if active not in PAGES:
    active = "📋 Research Brief"
PAGES[active]()
