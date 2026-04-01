"""
3-step self-refinement pipeline for FAIL-CHAIN.
Prompt templates and generation utilities for Qwen 2.5 3B.

Two pipeline variants:
  run_pipeline()       — standard: step 2 sees step 1 output (context-conditioned)
  run_pipeline_noctx() — ablation: step 2 does NOT see step 1 output (context-free)

The ablation tests SCoRe's hypothesis that self-correction fails due to context
contamination from step 1. If AMPLIFIED rate drops in context-free condition,
cascade is context-driven. If not, it is a model behavior collapse.
"""
import torch
import numpy as np
from typing import Tuple, Optional


# ── Prompt templates ──────────────────────────────────────────────────────────

STEP1_PROMPT = """Answer the following question concisely and factually.
Question: {question}
Answer:"""

STEP2_PROMPT = """You previously answered a question. Review your answer carefully.
Question: {question}
Your previous answer: {step1_answer}
Is this answer accurate? Provide a verified or corrected answer.
Verified answer:"""

# Context-free Step 2: IDENTICAL instruction and format to STEP2_PROMPT,
# but {step1_answer} is removed. Only the context changes — not the task.
# This is the correct ablation for SCoRe's context-contamination hypothesis.
#
# DESIGN NOTE: The original STEP2_NOCTX changed the instruction framing
# ("pause and verify..."), which confounds the ablation — the task changes
# alongside the context. This version uses the exact same "Review your answer"
# framing but omits the answer, so any difference in failure rates is attributable
# purely to the absence of step-1 context, not to different instruction phrasing.
STEP2_NOCTX_PROMPT = """Review your answer to the following question carefully.
Question: {question}
Is your answer accurate? Provide a verified or corrected answer.
Verified answer:"""

STEP3_PROMPT = """Based on your analysis, provide your final definitive answer.
Question: {question}
Initial answer: {step1_answer}
Verified answer: {step2_answer}
Final answer:"""


def format_step1(question: str) -> str:
    return STEP1_PROMPT.format(question=question)


def format_step2(question: str, step1_answer: str) -> str:
    return STEP2_PROMPT.format(question=question, step1_answer=step1_answer)


def format_step2_noctx(question: str) -> str:
    """Context-free Step 2 prompt — no step 1 answer exposed."""
    return STEP2_NOCTX_PROMPT.format(question=question)


def format_step3(question: str, step1_answer: str, step2_answer: str) -> str:
    return STEP3_PROMPT.format(
        question=question,
        step1_answer=step1_answer,
        step2_answer=step2_answer
    )


# ── Generation with hidden state extraction ───────────────────────────────────

def generate_with_hidden_state(
    model,
    tokenizer,
    prompt: str,
    target_layer: int = 32,
    max_new_tokens: int = 80,
    device: str = "cuda"
) -> Tuple[str, np.ndarray]:
    """
    Generate a response and extract the last-token hidden state at target_layer.

    Returns:
        (generated_text, hidden_state_at_layer)
        hidden_state shape: [hidden_dim]  (last token position, target layer)

    NOTE — hook capture behavior:
        The forward hook fires on EVERY autoregressive generation step (one per new
        token). Each step overwrites hidden_state_capture["h"]. The final captured
        state therefore corresponds to the LAST generated token, not the prompt's
        last token. This is intentional: we want the representation after the model
        has finished generating its answer, encoding the full context + answer.
        For analysis of prompt-boundary states, re-run with max_new_tokens=1.
    """
    hidden_state_capture = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # Last token, detached, float32
        hidden_state_capture["h"] = h[0, -1, :].detach().float().cpu().numpy()

    # Register hook on target layer
    hook = model.model.layers[target_layer].register_forward_hook(hook_fn)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    hook.remove()

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    hidden_state = hidden_state_capture.get("h", np.zeros(model.config.hidden_size))

    return generated_text, hidden_state


def run_pipeline(
    model,
    tokenizer,
    question: str,
    target_layer: int = 32,
    max_new_tokens: int = 80,
    device: str = "cuda"
) -> dict:
    """
    Standard 3-step self-refinement pipeline.
    Step 2 receives step 1 output as context (context-conditioned).

    Returns dict with:
        step1_output, step2_output, step3_output: generated text
        h_s1, h_s2, h_s3: hidden states at target_layer  [hidden_dim]
        pipeline_type: "standard"
    """
    p1 = format_step1(question)
    s1_out, h_s1 = generate_with_hidden_state(
        model, tokenizer, p1, target_layer, max_new_tokens, device)

    p2 = format_step2(question, s1_out)
    s2_out, h_s2 = generate_with_hidden_state(
        model, tokenizer, p2, target_layer, max_new_tokens, device)

    p3 = format_step3(question, s1_out, s2_out)
    s3_out, h_s3 = generate_with_hidden_state(
        model, tokenizer, p3, target_layer, max_new_tokens, device)

    return {
        "step1_output": s1_out,
        "step2_output": s2_out,
        "step3_output": s3_out,
        "h_s1": h_s1,
        "h_s2": h_s2,
        "h_s3": h_s3,
        "pipeline_type": "standard",
    }


def run_pipeline_noctx(
    model,
    tokenizer,
    question: str,
    target_layer: int = 32,
    max_new_tokens: int = 80,
    device: str = "cuda"
) -> dict:
    """
    Context-free ablation pipeline.
    Step 2 does NOT receive step 1 output — tests whether cascade propagation
    is driven by context contamination or intrinsic model behavior collapse.

    SCoRe (Kumar et al., NeurIPS 2024) hypothesis: self-correction fails because
    the model cannot escape its own wrong step 1 context. This pipeline tests that.

    LOOSE END [VAR-2B]: If AMPLIFIED rate drops significantly vs standard pipeline,
    the cascade is context-driven → GUARDIAN should strip context before REGENERATE.
    If AMPLIFIED rate stays the same, cascade is a model behavior collapse →
    GUARDIAN's ABSTAIN zone becomes the primary intervention.

    Returns dict with pipeline_type: "noctx"
    """
    p1 = format_step1(question)
    s1_out, h_s1 = generate_with_hidden_state(
        model, tokenizer, p1, target_layer, max_new_tokens, device)

    # Step 2 without step 1 context — fresh verification
    p2 = format_step2_noctx(question)
    s2_out, h_s2 = generate_with_hidden_state(
        model, tokenizer, p2, target_layer, max_new_tokens, device)

    p3 = format_step3(question, s1_out, s2_out)
    s3_out, h_s3 = generate_with_hidden_state(
        model, tokenizer, p3, target_layer, max_new_tokens, device)

    return {
        "step1_output": s1_out,
        "step2_output": s2_out,
        "step3_output": s3_out,
        "h_s1": h_s1,
        "h_s2": h_s2,
        "h_s3": h_s3,
        "pipeline_type": "noctx",
    }
