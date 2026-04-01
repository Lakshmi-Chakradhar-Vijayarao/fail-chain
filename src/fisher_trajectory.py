"""
Fisher separability proxy computation for each pipeline step.
Used in FAIL-CHAIN to measure geometric trajectory through the pipeline.
"""
import numpy as np
from typing import List, Dict


def fisher_proxy(h: np.ndarray) -> float:
    """
    Per-sample Fisher magnitude proxy: L2 norm of the hidden state vector.

    IMPORTANT — what this is and what it is not:
    This is NOT Fisher separability J = δᵀΣ⁻¹δ, which is a distributional
    quantity requiring two populations (correct vs wrong). That is computed
    in compute_fisher_j() for population-level analysis.

    This is ||h||₂ — the representation magnitude at a single step for a
    single sample. It serves as a proxy because:
    1. In HaRP and GEOM-PROOF, higher representation magnitude at L32 correlated
       with higher class separability across the dataset.
    2. Low ||h||₂ at intermediate steps is empirically associated with samples
       that will fail at the final step — the representation is "compressed"
       toward a degenerate attractor.

    In the paper, refer to this as "representation magnitude proxy" or
    "Fisher magnitude proxy" — not "Fisher J" or "Fisher separability."
    The population-level J is reported separately from compute_trajectory().

    Args:
        h: hidden state vector [hidden_dim] at layer L32
    Returns:
        scalar ≥ 0 — representation magnitude
    """
    return float(np.linalg.norm(h))


def compute_fisher_j(h_correct: np.ndarray, h_wrong: np.ndarray) -> float:
    """
    Full Fisher separability J = delta^T Sigma^{-1} delta
    between correct and wrong hidden state distributions.

    Args:
        h_correct: [n_correct, hidden_dim]
        h_wrong:   [n_wrong, hidden_dim]
    Returns:
        Fisher J scalar
    """
    mu_c = h_correct.mean(0)
    mu_w = h_wrong.mean(0)
    delta = mu_c - mu_w

    sigma = np.cov(np.vstack([h_correct, h_wrong]).T)

    try:
        sigma_inv = np.linalg.pinv(sigma)
        J = float(delta @ sigma_inv @ delta)
    except np.linalg.LinAlgError:
        # Fallback: normalized squared distance
        trace_sigma = np.trace(sigma) + 1e-8
        J = float(np.dot(delta, delta) / trace_sigma)

    return max(J, 0.0)  # Fisher J is non-negative


def compute_trajectory(data: List[Dict]) -> Dict:
    """
    Compute Fisher proxy trajectory stats for a list of pipeline samples.

    Args:
        data: list of dicts with h_s1, h_s2, h_s3 keys

    Returns:
        dict with mean/std proxy at each step
    """
    j_s1 = [fisher_proxy(d["h_s1"]) for d in data]
    j_s2 = [fisher_proxy(d["h_s2"]) for d in data]
    j_s3 = [fisher_proxy(d["h_s3"]) for d in data]

    return {
        "step1": {"mean": float(np.mean(j_s1)), "std": float(np.std(j_s1))},
        "step2": {"mean": float(np.mean(j_s2)), "std": float(np.std(j_s2))},
        "step3": {"mean": float(np.mean(j_s3)), "std": float(np.std(j_s3))},
        "slope_s1_s2": float(np.mean(j_s2) - np.mean(j_s1)),
        "slope_s2_s3": float(np.mean(j_s3) - np.mean(j_s2)),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def build_feature_matrix(data: List[Dict]) -> np.ndarray:
    """
    Build early-warning feature matrix from steps 1 and 2.
    Used for cascade prediction before step 3 runs.

    Features per sample: [proxy_s1, proxy_s2, delta_s1_s2]

    Returns:
        X: [n_samples, 3]
    """
    rows = []
    for d in data:
        j1 = fisher_proxy(d["h_s1"])
        j2 = fisher_proxy(d["h_s2"])
        rows.append([j1, j2, j2 - j1])
    return np.array(rows)


def build_rich_feature_matrix(data: List[Dict]) -> np.ndarray:
    """
    Extended early-warning feature matrix — 5 features per sample.
    Used for publication-quality AUROC comparison.

    Features:
        [proxy_s1, proxy_s2, delta_s1_s2, cosine_s1_s2, l2_delta_hidden]

    - proxy_s1, proxy_s2: L2 norm at each step (Fisher J proxy)
    - delta_s1_s2: norm change step1→step2 (trajectory slope)
    - cosine_s1_s2: cosine similarity h_s1 vs h_s2 (representation drift)
      LOW cosine = large representational shift between steps (cascade risk signal)
    - l2_delta_hidden: L2 norm of (h_s2 - h_s1) in raw hidden space
      Captures direction + magnitude of representation drift jointly

    LOOSE END [VAR-6]: cosine_s1_s2 is the primary LATENT failure detector —
    LATENT samples will show normal proxy values (hence passing surface governance)
    but anomalous cosine drift indicating the representation is drifting away from
    the correct attractor despite appearing textually correct at step 2.
    This is the key finding for Paper Section 4.3.

    Returns:
        X: [n_samples, 5]
    """
    rows = []
    for d in data:
        h1 = np.array(d["h_s1"])
        h2 = np.array(d["h_s2"])
        j1 = fisher_proxy(h1)
        j2 = fisher_proxy(h2)
        cos_12 = cosine_similarity(h1, h2)
        l2_delta = float(np.linalg.norm(h2 - h1))
        rows.append([j1, j2, j2 - j1, cos_12, l2_delta])
    return np.array(rows)


def build_late_feature_matrix(data: List[Dict]) -> np.ndarray:
    """
    Build late-detection feature matrix from step 3 only.
    Baseline for comparison with early warning.

    Returns:
        X: [n_samples, 1]
    """
    return np.array([[fisher_proxy(d["h_s3"])] for d in data])
