"""Evaluation & Validation module — Phase 5.

Kiểm chứng lời giải thích từ Phase 4 bằng 6 phép đo,
ánh xạ vào 3 KPI cốt lõi: stability, consistency, user_confirmable.
"""

from .label_consistency import label_consistency, self_consistency
from .feature_validation import (
    bt_feature_validation,
    generalization_check,
    sample_counterfactual,
)
from .robustness import robustness_check
from .phase5 import validate_explanations

__all__ = [
    "label_consistency",
    "self_consistency",
    "bt_feature_validation",
    "generalization_check",
    "sample_counterfactual",
    "robustness_check",
    "validate_explanations",
]
