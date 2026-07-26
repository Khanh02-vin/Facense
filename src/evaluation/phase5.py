"""Phase 5 orchestrator — runs all validation checks, produces KPI report.

Combines:
  1. Label consistency (Cohen's Kappa + self-consistency)
  2. Feature validation (BT direct validation)
  3. Generalization (train/test split)
  4. Robustness (incremental data)
  5. Stability reference (from Phase 4)

Outputs a unified validation report with 3 KPI pass/fail verdicts.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from .label_consistency import label_consistency, self_consistency
from .feature_validation import (
    bt_feature_validation,
    generalization_check,
)
from .robustness import robustness_check


# ==========================================================================
# KPI evaluators
# ==========================================================================


def _kpi_stability(
    robustness_result: dict,
    stability_reference: dict | None,
) -> dict:
    """KPI 1: Ổn định — same data → same results.
    Uses robustness (data addition stability) + Phase 4 stability reference.
    """
    issues = []

    rob_verdict = robustness_result.get("verdict", "unknown")
    if rob_verdict == "insufficient_data":
        return {"verdict": "skip", "reason": "insufficient_data", "pass": None}

    if rob_verdict == "unstable":
        issues.append(f"robustness={rob_verdict}")

    if stability_reference:
        stab_verdict = stability_reference.get("verdict", "")
        if stab_verdict in ("unstable", "no_data"):
            issues.append(f"bootstrap_stability={stab_verdict}")

    passed = len(issues) == 0
    return {
        "verdict": "pass" if passed else "fail",
        "pass": passed,
        "issues": issues if not passed else [],
        "robustness_verdict": rob_verdict,
        "stability_reference": stability_reference,
    }


def _kpi_consistency(
    self_consistency_result: dict,
    generalization_result: dict | None,
    label_consistency_result: dict | None,
) -> dict:
    """KPI 2: Nhất quán — thêm ít dữ liệu không đảo lộn kết luận."""
    issues = []

    sc_verdict = self_consistency_result.get("verdict", "")
    if sc_verdict == "inconsistent":
        issues.append(f"self_consistency={sc_verdict}")

    if generalization_result:
        gen_verdict = generalization_result.get("verdict", "")
        if gen_verdict == "limited":
            issues.append(f"generalization={gen_verdict}")

    passed = len(issues) == 0

    result = {
        "verdict": "pass" if passed else "fail",
        "pass": passed,
        "issues": issues if not passed else [],
        "self_consistency_verdict": sc_verdict,
        "generalization_verdict": generalization_result.get("verdict") if generalization_result else None,
    }

    if label_consistency_result:
        result["label_consistency"] = label_consistency_result

    return result


def _kpi_user_confirmable(
    feature_validation_results: list[dict],
) -> dict:
    """KPI 3: Người dùng xác nhận — features validated by BT score gaps."""
    if not feature_validation_results:
        return {"verdict": "skip", "reason": "no_features_to_validate", "pass": None}

    n_validated = sum(1 for fv in feature_validation_results
                      if fv.get("importance") == "validated")
    n_total = len(feature_validation_results)

    # Pass if at least top-3 features are validated
    top3 = feature_validation_results[:3]
    top3_validated = sum(1 for fv in top3 if fv.get("importance") == "validated")
    passed = top3_validated >= 2

    return {
        "verdict": "pass" if passed else "fail",
        "pass": passed,
        "n_validated": n_validated,
        "n_total": n_total,
        "top3_validated": top3_validated,
    }


# ==========================================================================
# Orchestrator
# ==========================================================================


def validate_explanations(
    pairs: list,
    clip_features: dict[str, dict[str, float]],
    bt_scores: dict[str, float],
    feature_names: Sequence[str] | None = None,
    labels1: list[str] | None = None,
    labels2: list[str] | None = None,
    stability_reference: dict | None = None,
) -> dict:
    """Run all Phase 5 validation checks and produce KPI report.

    Args:
        pairs: List of PairwiseSample (or compatible objects).
        clip_features: {clip_name: {feature: value}}
        bt_scores: {clip_name: BT score}
        feature_names: Feature name list (default: ``FEATURE_NAMES``).
        labels1: Optional first-round labels for Cohen's Kappa.
        labels2: Optional second-round labels for Cohen's Kappa.
        stability_reference: Optional Phase 4 stability result.

    Returns:
        Validation report dict.
    """
    from src.preference_learning.preference import PairwiseSample

    if feature_names is None:
        from src.explainability.evidence import FEATURE_NAMES
        feature_names = FEATURE_NAMES

    # Convert pairs
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

    n_pairs = len(pair_objects)
    n_clips = len(clip_features)

    # ── Label consistency ────────────────────────────────────────────
    lc_result = None
    if labels1 is not None and labels2 is not None:
        lc_result = label_consistency(labels1, labels2)

    sc_result = self_consistency(pair_objects, clip_features, bt_scores, feature_names)

    # ── Feature validation ──────────────────────────────────────────
    fv_result = bt_feature_validation(clip_features, bt_scores, feature_names)

    # ── Generalization ──────────────────────────────────────────────
    gen_result = generalization_check(pair_objects, clip_features, bt_scores, feature_names)

    # ── Robustness ──────────────────────────────────────────────────
    rob_result = robustness_check(clip_features, bt_scores, feature_names)

    # ── KPI summary ─────────────────────────────────────────────────
    kpi_stab = _kpi_stability(rob_result, stability_reference)
    kpi_cons = _kpi_consistency(sc_result, gen_result, lc_result)
    kpi_user = _kpi_user_confirmable(fv_result)

    kpi_results = [kpi_stab, kpi_cons, kpi_user]
    n_pass = sum(1 for k in kpi_results if k.get("verdict") == "pass")
    n_total_kpi = sum(1 for k in kpi_results if k.get("pass") is not None)

    if n_total_kpi == 0:
        overall = "insufficient_data"
    elif n_pass == n_total_kpi:
        overall = "trustworthy"
    elif n_pass >= n_total_kpi // 2:
        overall = "partially_trustworthy"
    else:
        overall = "untrustworthy"

    return {
        "label_consistency": lc_result,
        "self_consistency": sc_result,
        "feature_validation": fv_result,
        "generalization": gen_result,
        "robustness": rob_result,
        "kpi_summary": {
            "stability": {"kpi": "Ổn định", **kpi_stab},
            "consistency": {"kpi": "Nhất quán", **kpi_cons},
            "user_confirmable": {"kpi": "Người dùng xác nhận", **kpi_user},
            "overall": overall,
            "n_pairs": n_pairs,
            "n_clips": n_clips,
        },
    }
