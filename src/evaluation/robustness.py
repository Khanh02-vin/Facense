"""Robustness — thêm dữ liệu từ từ, đo độ ổn định của top features.

Sample progressively larger fractions of clips, recompute Ridge
feature importance, measure Kendall Tau against full-data ranking.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Sequence

from .feature_validation import _ridge_importances


def robustness_check(
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] | None = None,
    fractions: list[float] | None = None,
    n_repeat: int = 3,
) -> dict:
    """Measure feature-rank stability as data grows.

    Args:
        clip_features: {clip_name: {feature: value}}
        bt_scores: {clip_name: score}
        fractions: Fractions of data to sample (default
            ``[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]``).
        n_repeat: Number of random samples per fraction (averaged).

    Returns:
    ```python
    {
      "kendall_tau_progression": [0.82, 0.88, 0.92, 0.95, 0.98, 1.0],
      "verdict": "stable",
      "fractions": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    ```
    """
    if feature_names is None:
        from src.explainability.evidence import FEATURE_NAMES
        feature_names = FEATURE_NAMES
    if fractions is None:
        fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    clip_names = sorted(clip_features.keys())
    n = len(clip_names)
    if n < 10:
        return {
            "kendall_tau_progression": [],
            "verdict": "insufficient_data",
            "fractions": fractions,
        }

    matrix = np.array(
        [[clip_features[cn].get(f, 0.0) for f in feature_names]
         for cn in clip_names],
        dtype=np.float64,
    )
    y = np.array([bt_scores.get(cn, 0.0) for cn in clip_names], dtype=np.float64)

    # Full-data ranking
    full_imp = _ridge_importances(clip_features, bt_scores, feature_names)
    full_rank = np.argsort(-np.abs(full_imp))

    rng = np.random.default_rng(42)
    tau_progression = []

    for frac in fractions:
        taus = []
        for _ in range(n_repeat):
            k = max(5, int(n * frac))
            idx = rng.choice(n, size=k, replace=False)
            X_sub = matrix[idx]
            y_sub = y[idx]

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_sub)
            model = Ridge(alpha=1.0)
            model.fit(Xs, y_sub)
            sub_imp = np.abs(model.coef_)
            sub_rank = np.argsort(-sub_imp)

            # Kendall Tau on union of top-5 features
            full_top5 = set(full_rank[:5])
            sub_top5 = set(sub_rank[:5])
            union = list(full_top5 | sub_top5)
            if len(union) >= 2:
                rf = [full_rank.tolist().index(i) for i in union]
                rs = [sub_rank.tolist().index(i) for i in union]
                tau, _ = kendalltau(rf, rs)
                if not np.isnan(tau):
                    taus.append(tau)
        tau_progression.append(float(np.mean(taus)) if taus else 0.0)

    tau_progression[-1] = 1.0  # last fraction = full data → perfect

    # Verdict: stable if all tau (except last) >= 0.7
    if len(tau_progression) <= 1:
        verdict = "stable"
    else:
        stable_count = sum(1 for t in tau_progression[:-1] if t >= 0.7)
        verdict = "stable" if stable_count == len(tau_progression) - 1 else "unstable"

    return {
        "kendall_tau_progression": [round(t, 4) for t in tau_progression],
        "verdict": verdict,
        "fractions": fractions,
    }
