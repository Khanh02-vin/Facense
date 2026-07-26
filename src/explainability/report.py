"""Report Generator — gom evidence + counterfactual + stability + cluster.

Produces the final structured JSON report consumed by `serve.py` and
downstream consumers.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Sequence

from .evidence import compute_evidence, FEATURE_NAMES
from .counterfactual import counterfactual_analysis
from .stability import stability_check


# ==========================================================================
# Cluster explanation helper
# ==========================================================================


def _cluster_explanation(
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    top_frac: float = 0.3,
    max_clusters: int = 3,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[dict]:
    """Cluster top-scored clips and describe each cluster's feature profile.

    Takes the top ``top_frac`` of clips by BT score, runs KMeans,
    and returns per-cluster feature means with a concise label.
    """
    if len(bt_scores) < 10:
        return []

    # Sort clips by BT score descending
    sorted_clips = sorted(bt_scores.items(), key=lambda x: -x[1])
    n_top = max(3, int(len(sorted_clips) * top_frac))
    top_clips = [cn for cn, _ in sorted_clips[:n_top]]

    # Build matrix
    matrix = np.array(
        [[clip_features.get(cn, {}).get(f, 0.0) for f in feature_names]
         for cn in top_clips],
        dtype=np.float64,
    )

    if len(matrix) < 3:
        return []

    # Determine K
    n_clusters = min(max_clusters, len(matrix) - 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    clusters = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        center = scaler.inverse_transform(km.cluster_centers_[k].reshape(1, -1))[0]

        # Top-3 distinguishing features (highest absolute deviation from global mean)
        global_mean = np.mean(matrix, axis=0)
        deviation = np.abs(center - global_mean)
        top_idx = np.argsort(-deviation)[:3]

        features = {
            feature_names[i]: round(float(center[i]), 3)
            for i in top_idx
        }

        # Generate a concise label from the top feature
        top_feat = feature_names[top_idx[0]]
        val = center[top_idx[0]]
        direction = "cao" if val > global_mean[top_idx[0]] else "thấp"
        label = f"{top_feat}_{direction}"

        clusters.append({
            "name": f"Type {chr(65 + k)}",
            "label": label,
            "n_clips": int(mask.sum()),
            "features": features,
        })

    clusters.sort(key=lambda x: -x["n_clips"])
    return clusters


# ==========================================================================
# NL summary (LLM-free, template-based)
# ==========================================================================


def _summary_line(consensus: list[dict], n_pairs: int) -> str:
    """Template-based NL summary from consensus evidence."""
    if not consensus:
        return "Chưa có đủ dữ liệu để phân tích sở thích."

    top = consensus[:3]
    parts = []
    for t in top:
        feat = t["feature"].replace("_", " ")
        parts.append(feat)
    return (
        f"Bạn có xu hướng ưu tiên {', '.join(parts[:-1])} và {parts[-1]}. "
        f"Phân tích dựa trên {n_pairs} cặp so sánh."
    )


# ==========================================================================
# Top-level API
# ==========================================================================


def generate_report(
    pairs: list,
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    n_bootstrap: int = 100,
) -> dict:
    """Generate complete explainability report.

    Args:
        pairs: List of PairwiseSample (or any object with ``image_A``,
            ``image_B``, ``winner`` attrs).
        clip_features: Dict mapping clip_name → {feature_name: value}.
        bt_scores: Dict mapping clip_name → BT score.
        feature_names: Ordered list of feature names matching the values
            in ``clip_features``.
        n_bootstrap: Number of bootstrap iterations for stability.

    Returns:
        Structured report dict (JSON-serialisable).
    """
    # ── Prepare data ────────────────────────────────────────────────
    from src.preference_learning.preference import PairwiseSample

    # Ensure pairs are PairwiseSample objects
    pair_objects: list[PairwiseSample] = []
    for p in pairs:
        if isinstance(p, PairwiseSample):
            pair_objects.append(p)
        elif hasattr(p, "image_A") and hasattr(p, "image_B") and hasattr(p, "winner"):
            pair_objects.append(PairwiseSample(
                user_id=getattr(p, "user_id", ""),
                image_A=p.image_A,
                image_B=p.image_B,
                winner=p.winner,
            ))

    # ── Evidence ────────────────────────────────────────────────────
    evidence = compute_evidence(pair_objects, clip_features, bt_scores, feature_names)

    # ── Counterfactual ──────────────────────────────────────────────
    cf = counterfactual_analysis(clip_features, bt_scores, feature_names)

    # ── Stability ───────────────────────────────────────────────────
    stab = stability_check(clip_features, bt_scores, feature_names, n_bootstrap=n_bootstrap)

    # ── Cluster explanation ─────────────────────────────────────────
    clusters = _cluster_explanation(clip_features, bt_scores, feature_names=feature_names)

    # ── Confidence ──────────────────────────────────────────────────
    n_pairs = len(pair_objects)
    n_clips = len(clip_features)

    # ── Summary ─────────────────────────────────────────────────────
    consensus = evidence.get("consensus", [])
    summary = _summary_line(consensus, n_pairs)

    return {
        "summary": summary,
        "evidence": evidence,
        "counterfactual": cf,
        "stability": stab,
        "clusters": clusters,
        "confidence": {
            "n_pairs": n_pairs,
            "n_clips": n_clips,
        },
    }
