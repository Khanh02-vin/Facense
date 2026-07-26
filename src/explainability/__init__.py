"""Explainability module — Phase 4.

Makes preference scores interpretable by computing feature-level
evidence from pairwise labels, counterfactual impact, stability,
and cluster-based preference typing.

All evidence weights are computed from data, never from LLM.
"""

from .evidence import compute_evidence
from .counterfactual import counterfactual_analysis
from .stability import stability_check
from .report import generate_report

__all__ = [
    "compute_evidence",
    "counterfactual_analysis",
    "stability_check",
    "generate_report",
]
