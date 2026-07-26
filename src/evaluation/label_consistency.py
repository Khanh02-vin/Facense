"""Label consistency metrics — Cohen's Kappa + self-consistency.

* **Cohen's Kappa**: inter-rater / test-retest agreement on pairwise labels.
  Uses ``sklearn.metrics.cohen_kappa_score``.

* **Self-consistency**: split pairwise comparisons into two halves,
  fit BT on each, compare top feature rankings via Kendall Tau.
  Answers: "are conclusions consistent across subsets of data?"
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau
from typing import Sequence

from src.preference_learning.preference import PairwiseSample, BradleyTerryModel
from .feature_validation import _ridge_importances  # shared helper


# ==========================================================================
# Cohen's Kappa — test-retest label agreement
# ==========================================================================


_LANDIS_KOCH = [
    (0.0, "poor"),
    (0.2, "slight"),
    (0.4, "fair"),
    (0.6, "moderate"),
    (0.8, "substantial"),
    (1.01, "almost_perfect"),
]


def _kappa_interpretation(kappa: float) -> str:
    for threshold, label in _LANDIS_KOCH:
        if kappa < threshold:
            return label
    return "almost_perfect"


def label_consistency(
    labels1: list[str],
    labels2: list[str],
) -> dict:
    """Cohen's Kappa between two rounds of labels for the same pairs.

    Both lists must be the same length, aligned by pair.  Accepted
    label values: ``"A"``, ``"B"``, ``"a"``, ``"b"``, ``"equal"``,
    ``"skip"``.

    Returns:
    ```python
    {
      "cohens_kappa": 0.91,
      "agreement_pct": 95.0,
      "n_pairs": 200,
      "interpretation": "almost_perfect",
    }
    ```
    """
    if len(labels1) != len(labels2):
        raise ValueError(
            f"Label lists must have same length: {len(labels1)} vs {len(labels2)}"
        )
    if not labels1:
        return {
            "cohens_kappa": 1.0,
            "agreement_pct": 100.0,
            "n_pairs": 0,
            "interpretation": "no_data",
        }

    # Normalise
    def _norm(seq: list[str]) -> list[str]:
        return [s.upper() if s.lower() in ("a", "b") else s.lower()
                for s in seq]

    n1 = _norm(labels1)
    n2 = _norm(labels2)

    kappa = float(cohen_kappa_score(n1, n2))
    agreement = sum(1 for a, b in zip(n1, n2) if a == b) / len(n1) * 100.0

    return {
        "cohens_kappa": round(kappa, 4),
        "agreement_pct": round(agreement, 2),
        "n_pairs": len(labels1),
        "interpretation": _kappa_interpretation(kappa),
    }


# ==========================================================================
# Self-consistency — split-half rank correlation
# ==========================================================================


def self_consistency(
    pairs: list[PairwiseSample],
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] | None = None,
) -> dict:
    """Split pairwise comparisons into two halves, fit BT on each,
    compare top-5 feature rankings via Kendall Tau.

    Returns:
    ```python
    {
      "kendall_tau": 0.85,
      "verdict": "consistent",
      "top_features_half_a": [...],
      "top_features_half_b": [...],
    }
    ```
    """
    if not pairs or len(pairs) < 10:
        return {
            "kendall_tau": 1.0,
            "verdict": "insufficient_data",
            "top_features_half_a": [],
            "top_features_half_b": [],
        }

    if feature_names is None:
        from src.explainability.evidence import FEATURE_NAMES
        feature_names = FEATURE_NAMES

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(pairs))
    mid = len(idx) // 2

    def _top_features(subset: list[PairwiseSample]) -> list[str]:
        # Fit BT on subset to get BT scores for those clips
        bt = BradleyTerryModel()
        res = bt.fit(subset)
        imp = _ridge_importances(clip_features, res.item_scores, feature_names)
        ranked = sorted(
            range(len(imp)), key=lambda i: -abs(imp[i])
        )
        return [feature_names[i] for i in ranked[:5]]

    half_a = [pairs[i] for i in idx[:mid]]
    half_b = [pairs[i] for i in idx[mid:]]

    top_a = _top_features(half_a)
    top_b = _top_features(half_b)

    # Kendall Tau on top features (matching full names)
    all_f = list(dict.fromkeys(top_a + top_b))
    if len(all_f) < 2:
        tau = 1.0
    else:
        ra = [top_a.index(f) if f in top_a else len(top_a) for f in all_f]
        rb = [top_b.index(f) if f in top_b else len(top_b) for f in all_f]
        tau, _ = kendalltau(ra, rb)
        tau = float(tau) if not np.isnan(tau) else 0.0

    verdict = "consistent" if tau >= 0.7 else "inconsistent"

    return {
        "kendall_tau": round(tau, 4),
        "verdict": verdict,
        "top_features_half_a": top_a,
        "top_features_half_b": top_b,
    }
