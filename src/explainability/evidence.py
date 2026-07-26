"""Evidence Layer — feature importance from pairwise preference data.

Three methods, all data-driven (no LLM):

* **PairwiseDelta** — for each pair (A > B), compute feature-wise
  difference.  Aggregate mean difference across all pairs.  A positive
  weight means the feature tends to be higher in the winning clip.

* **LinearModel** — fit Ridge regression (features → BT score),
  interpret abs(coef_) as importance.

* **Permutation** — shuffle each feature across clips, refit Ridge
  on features → BT score, measure R² drop.  Larger drop = more
  important.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import Sequence

from src.preference_learning.preference import PairwiseSample

# ── Public feature list (L1 + L2, numeric only) ──────────────────────────
FEATURE_NAMES: list[str] = [
    # Layer 1
    "motion_energy",
    "motion_peak",
    "motion_variance",
    "blur_score",
    "brightness",
    "brightness_std",
    "face_visibility",
    # Layer 2
    "smile",
    "mouth_open",
    "eye_contact",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_symmetry",
    "face_clarity",
]

# pupil_left, pupil_right excluded (always 0.0 in stored data).
# face_detected excluded (boolean, not numeric).


# ==========================================================================
# Helpers
# ==========================================================================


def _clip_feature_dict(
    clip_features: dict[str, dict[str, float]],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, list[str]]:
    """Convert clip feature dicts to (matrix, clip_names) sorted by name."""
    clip_names = sorted(clip_features.keys())
    if not clip_names:
        return np.empty((0, len(feature_names)), dtype=np.float64), []
    matrix = np.array(
        [[clip_features[cn].get(f, 0.0) for f in feature_names] for cn in clip_names],
        dtype=np.float64,
    )
    return matrix, clip_names


def _normalize_weights(raw: dict[str, float]) -> list[dict]:
    """Normalise raw weights to [-1, 1] and return sorted list."""
    if not raw:
        return []
    vals = np.array(list(raw.values()))
    max_abs = float(np.max(np.abs(vals))) or 1.0
    items = [(f, float(v) / max_abs) for f, v in raw.items()]
    items.sort(key=lambda x: -abs(x[1]))
    return [
        {"feature": f, "weight": round(w, 4)}
        for f, w in items
        if abs(w) > 1e-6
    ]


# ==========================================================================
# Method 1 — PairwiseDelta
# ==========================================================================


def pairwise_delta(
    pairs: list[PairwiseSample],
    clip_features: dict[str, dict[str, float]],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[dict]:
    """Feature importance via winner-loser feature difference.

    Returns sorted list of {"feature", "weight", "n_pairs"}.
    """
    accum = {f: 0.0 for f in feature_names}
    counts = {f: 0 for f in feature_names}

    for p in pairs:
        if p.winner not in ("A", "B"):
            continue
        winner_id = p.image_A if p.winner == "A" else p.image_B
        loser_id = p.image_B if p.winner == "A" else p.image_A
        if winner_id not in clip_features or loser_id not in clip_features:
            continue

        fw = clip_features[winner_id]
        fl = clip_features[loser_id]

        for f in feature_names:
            delta = fw.get(f, 0.0) - fl.get(f, 0.0)
            if abs(delta) > 1e-9:
                accum[f] += delta
                counts[f] += 1

    # Mean delta per feature
    mean_delta = {}
    for f in feature_names:
        if counts[f] > 0:
            mean_delta[f] = accum[f] / counts[f]

    # Normalise
    if not mean_delta:
        return []

    max_abs = max(abs(v) for v in mean_delta.values()) or 1.0
    result = []
    for f in feature_names:
        if f in mean_delta and abs(mean_delta[f]) > 1e-6:
            result.append({
                "feature": f,
                "weight": round(mean_delta[f] / max_abs, 4),
                "n_pairs": counts[f],
            })
    result.sort(key=lambda x: -abs(x["weight"]))
    return result


# ==========================================================================
# Method 2 — LinearModel (Ridge regressor)
# ==========================================================================


def linear_model_importance(
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    alpha: float = 1.0,
) -> list[dict]:
    """Feature importance via Ridge regression coefficients.

    Returns sorted list of {"feature", "weight", "coef", "r2"}.
    """
    matrix, clip_names = _clip_feature_dict(clip_features, feature_names)
    if len(clip_names) == 0:
        return []

    y = np.array([bt_scores.get(cn, 0.0) for cn in clip_names], dtype=np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))

    coefs = np.abs(model.coef_)
    max_c = float(np.max(coefs)) or 1.0
    result = [
        {
            "feature": feature_names[i],
            "weight": round(float(coefs[i] / max_c), 4),
            "coef": round(float(model.coef_[i]), 4),
            "r2": round(r2, 4),
        }
        for i in range(len(feature_names))
        if coefs[i] > 1e-8
    ]
    result.sort(key=lambda x: -abs(x["weight"]))
    return result


# ==========================================================================
# Method 3 — Permutation importance (on Ridge → BT score)
# ==========================================================================


def permutation_importance(
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    n_repeats: int = 5,
    alpha: float = 1.0,
) -> list[dict]:
    """Permutation importance — shuffle each feature, measure R² drop.

    Returns sorted list of {"feature", "weight", "drop", "std"}.
    """
    matrix, clip_names = _clip_feature_dict(clip_features, feature_names)
    if len(clip_names) == 0:
        return []

    y = np.array([bt_scores.get(cn, 0.0) for cn in clip_names], dtype=np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    # Baseline
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    baseline_r2 = r2_score(y, model.predict(X))

    result = []
    rng = np.random.default_rng(42)

    for i, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            model_perm = Ridge(alpha=alpha)
            model_perm.fit(X_perm, y)
            perm_r2 = r2_score(y, model_perm.predict(X_perm))
            drops.append(baseline_r2 - perm_r2)

        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))
        if mean_drop > 1e-8:
            result.append({
                "feature": fname,
                "weight": round(mean_drop, 4),
                "drop": round(mean_drop, 4),
                "std": round(std_drop, 4),
            })

    result.sort(key=lambda x: -abs(x["weight"]))

    # Also include R² drop as fraction of baseline
    if baseline_r2 > 1e-6:
        for r in result:
            r["drop_pct"] = round(r["drop"] / baseline_r2 * 100, 1)

    return result


# ==========================================================================
# Consensus aggregation
# ==========================================================================


def _consensus(
    *methods: list[dict],
    method_labels: list[str],
) -> list[dict]:
    """Average weights across methods, track rank stability."""
    from collections import defaultdict

    all_features = set()
    for m in methods:
        for item in m:
            all_features.add(item["feature"])

    scores: dict[str, list[float]] = defaultdict(list)
    for method, label in zip(methods, method_labels):
        for item in method:
            w = item.get("weight", 0.0)
            scores[item["feature"]].append(w)

    result = []
    for f in all_features:
        vals = scores.get(f, [0.0])
        avg_w = float(np.mean(vals))
        if abs(avg_w) < 1e-6:
            continue
        result.append({
            "feature": f,
            "avg_weight": round(avg_w, 4),
            "n_methods": len(vals),
        })

    result.sort(key=lambda x: -abs(x["avg_weight"]))
    return result


# ==========================================================================
# Top-level API
# ==========================================================================


def compute_evidence(
    pairs: list[PairwiseSample],
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict:
    """Run all three evidence methods and produce consensus.

    Returns dict with keys:
      pairwise_delta, regression, permutation, consensus
    """
    pd = pairwise_delta(pairs, clip_features, feature_names)
    lm = linear_model_importance(clip_features, bt_scores, feature_names)
    pm = permutation_importance(clip_features, bt_scores, feature_names)

    cs = _consensus(pd, lm, pm, method_labels=["pairwise_delta", "regression", "permutation"])

    return {
        "pairwise_delta": pd,
        "regression": lm,
        "permutation": pm,
        "consensus": cs,
    }
