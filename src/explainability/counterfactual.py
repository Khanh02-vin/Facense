"""Counterfactual Analysis — "bỏ feature X → accuracy giảm Y%".

Fits a Ridge regressor (features → BT score) as a proxy model for the
preference signal.  Counterfactual = remove one feature (set to mean),
refit, measure R² drop.

The drop percentage answers: "how much does this feature contribute to
predicting the user's preference score?"
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import Sequence

from .evidence import FEATURE_NAMES


def counterfactual_analysis(
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    alpha: float = 1.0,
) -> dict:
    """Compute counterfactual impact of each feature.

    For each feature:
      1. Fit Ridge on all features → BT score (baseline R²).
      2. Replace feature column with its mean, refit, measure R².
      3. drop = baseline_R² - ablation_R².

    Returns dict:
    ```python
    {
      "baseline_r2": 0.42,
      "counterfactuals": [
        {"feature": "eye_contact", "r2_drop": 0.18, "drop_pct": 42.9, "importance": "critical"},
        ...
      ]
    }
    ```

    Importance labels: critical (drop_pct > 30), notable (10-30),
    minor (1-10), negligible (< 1).
    """
    clip_names = sorted(clip_features.keys())
    matrix = np.array(
        [[clip_features[cn].get(f, 0.0) for f in feature_names] for cn in clip_names],
        dtype=np.float64,
    )
    y = np.array([bt_scores.get(cn, 0.0) for cn in clip_names], dtype=np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    # Baseline
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    baseline_r2 = r2_score(y, model.predict(X))

    # Per-feature ablation
    cf_list = []
    for i, fname in enumerate(feature_names):
        X_abl = X.copy()
        X_abl[:, i] = 0.0  # mean-centering → zero after StandardScaler

        model_abl = Ridge(alpha=alpha)
        model_abl.fit(X_abl, y)
        ablation_r2 = r2_score(y, model_abl.predict(X_abl))
        r2_drop = baseline_r2 - ablation_r2
        drop_pct = (r2_drop / baseline_r2 * 100) if baseline_r2 > 1e-6 else 0.0

        if drop_pct > 30:
            importance = "critical"
        elif drop_pct > 10:
            importance = "notable"
        elif drop_pct > 1:
            importance = "minor"
        else:
            importance = "negligible"

        cf_list.append({
            "feature": fname,
            "r2_drop": round(r2_drop, 4),
            "drop_pct": round(drop_pct, 1),
            "importance": importance,
        })

    cf_list.sort(key=lambda x: -x["r2_drop"])

    return {
        "baseline_r2": round(baseline_r2, 4),
        "counterfactuals": cf_list,
    }
