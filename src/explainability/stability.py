"""Stability Check — top features có đổi khi thêm dữ liệu?

Bootstrap resampling: subsample clips, recompute Ridge importance,
measure rank stability of top-K features via Kendall Tau.

Verdict:
  tau ≥ 0.7 → stable
  tau ≥ 0.4 → moderately stable
  tau < 0.4 → unstable
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
from typing import Sequence

from .evidence import FEATURE_NAMES


def _ridge_importances(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    alpha: float = 1.0,
) -> np.ndarray:
    """Return absolute coefficients from fitted Ridge."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return np.abs(model.coef_)


def stability_check(
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    n_bootstrap: int = 100,
    sample_frac: float = 0.8,
    top_k: int = 5,
    alpha: float = 1.0,
) -> dict:
    """Bootstrap stability of feature importance rankings.

    Returns:
    ```python
    {
      "kendall_tau": 0.89,
      "verdict": "stable",
      "top_k": 5,
      "top_features": ["eye_contact", "smile", ...],
      "n_bootstrap": 100,
    }
    ```
    """
    clip_names = sorted(clip_features.keys())
    matrix = np.array(
        [[clip_features[cn].get(f, 0.0) for f in feature_names] for cn in clip_names],
        dtype=np.float64,
    )
    y = np.array([bt_scores.get(cn, 0.0) for cn in clip_names], dtype=np.float64)

    n = len(clip_names)
    if n < 10:
        return {
            "kendall_tau": 1.0,
            "verdict": "insufficient_data",
            "top_k": top_k,
            "top_features": [],
            "n_bootstrap": 0,
        }

    # Full-data importance ranking
    full_imp = _ridge_importances(matrix, y, feature_names, alpha)
    full_rank = np.argsort(-full_imp)  # descending

    # Bootstrap
    rng = np.random.default_rng(42)
    tau_values = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=int(n * sample_frac), replace=True)
        X_bs = matrix[idx]
        y_bs = y[idx]

        bs_imp = _ridge_importances(X_bs, y_bs, feature_names, alpha)
        bs_rank = np.argsort(-bs_imp)

        # Kendall Tau on top-K features
        full_topk = set(full_rank[:top_k])
        bs_topk = set(bs_rank[:top_k])
        all_idx = list(full_topk | bs_topk)

        if len(all_idx) < 2:
            continue

        # Map to 1..len for kendalltau
        rel_full = np.array([full_rank.tolist().index(i) for i in all_idx])
        rel_bs = np.array([bs_rank.tolist().index(i) for i in all_idx])

        tau, _ = kendalltau(rel_full, rel_bs)
        if not np.isnan(tau):
            tau_values.append(tau)

    if not tau_values:
        return {
            "kendall_tau": 1.0,
            "verdict": "no_variation",
            "top_k": top_k,
            "top_features": [feature_names[i] for i in full_rank[:top_k]],
            "n_bootstrap": n_bootstrap,
        }

    mean_tau = float(np.mean(tau_values))
    if mean_tau >= 0.7:
        verdict = "stable"
    elif mean_tau >= 0.4:
        verdict = "moderately_stable"
    else:
        verdict = "unstable"

    return {
        "kendall_tau": round(mean_tau, 4),
        "verdict": verdict,
        "top_k": top_k,
        "top_features": [feature_names[i] for i in full_rank[:top_k]],
        "n_bootstrap": n_bootstrap,
    }
