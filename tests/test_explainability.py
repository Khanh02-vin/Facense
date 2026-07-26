"""Tests for Phase 4 — Explainability.

Run: python -m unittest tests.test_explainability -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preference_learning.preference import PairwiseSample
from src.explainability.evidence import (
    compute_evidence,
    pairwise_delta,
    linear_model_importance,
    permutation_importance,
    FEATURE_NAMES,
)
from src.explainability.counterfactual import counterfactual_analysis
from src.explainability.stability import stability_check
from src.explainability.report import generate_report


# ==========================================================================
# Helpers
# ==========================================================================

N_FEATURES = len(FEATURE_NAMES)


def _synthetic_data(
    n_clips: int = 50,
    n_pairs: int = 100,
    seed: int = 42,
) -> tuple[list[PairwiseSample], dict[str, dict[str, float]], dict[str, float]]:
    """Create synthetic clip features, BT scores, and pairwise labels.

    Inserts a known signal: ``eye_contact`` is the dominant feature
    (correlated with BT score), ``smile`` is secondary.
    """
    rng = np.random.default_rng(seed)

    clip_names = [f"clip_{i:04d}" for i in range(n_clips)]

    # Generate features with known structure
    clip_features: dict[str, dict[str, float]] = {}
    bt_scores: dict[str, float] = {}

    for cn in clip_names:
        feat = {}
        for f in FEATURE_NAMES:
            feat[f] = float(rng.uniform(0.0, 1.0))
        clip_features[cn] = feat

        # BT score = eye_contact*2 + smile*0.5 + noise*0.1
        score = (
            2.0 * feat["eye_contact"]
            + 0.5 * feat["smile"]
            + 0.1 * rng.normal()
        )
        bt_scores[cn] = round(score, 4)

    # Normalise BT scores to be centred
    mean_bt = np.mean(list(bt_scores.values()))
    for cn in bt_scores:
        bt_scores[cn] -= mean_bt

    # Generate pairwise labels based on BT scores
    pairs: list[PairwiseSample] = []
    for _ in range(n_pairs):
        a, b = rng.choice(clip_names, size=2, replace=False)
        winner = "A" if bt_scores[a] > bt_scores[b] else "B"
        pairs.append(PairwiseSample("u1", a, b, winner))

    return pairs, clip_features, bt_scores


# ==========================================================================
# Test: Evidence — PairwiseDelta
# ==========================================================================


class TestPairwiseDelta(unittest.TestCase):
    def test_eye_contact_ranked_top(self):
        """eye_contact should be top feature in synthetic data."""
        pairs, cfeat, _ = _synthetic_data()
        result = pairwise_delta(pairs, cfeat)
        self.assertTrue(result, "pairwise_delta returned empty")
        self.assertEqual(result[0]["feature"], "eye_contact")

    def test_delta_weights_in_range(self):
        """All weights should be in [-1, 1]."""
        pairs, cfeat, _ = _synthetic_data()
        result = pairwise_delta(pairs, cfeat)
        for r in result:
            self.assertGreaterEqual(r["weight"], -1.0)
            self.assertLessEqual(r["weight"], 1.0)

    def test_delta_has_n_pairs_field(self):
        """Each result should include n_pairs count."""
        pairs, cfeat, _ = _synthetic_data()
        result = pairwise_delta(pairs, cfeat)
        for r in result:
            self.assertIn("n_pairs", r)
            self.assertGreater(r["n_pairs"], 0)

    def test_empty_pairs(self):
        """Empty pairs should yield empty result."""
        result = pairwise_delta([], {"a": {"smile": 1.0}})
        self.assertEqual(result, [])

    def test_single_pair(self):
        """Single pair should produce deterministic weights."""
        cfeat = {
            "a": {f: 0.0 for f in FEATURE_NAMES},
            "b": {f: 0.0 for f in FEATURE_NAMES},
        }
        cfeat["a"]["smile"] = 1.0
        cfeat["a"]["eye_contact"] = 0.8
        pairs = [PairwiseSample("u", "a", "b", "A")]
        result = pairwise_delta(pairs, cfeat)
        self.assertTrue(result)
        # smile (delta=1.0) should rank higher than eye_contact (delta=0.8)
        self.assertEqual(result[0]["feature"], "smile")


# ==========================================================================
# Test: Evidence — LinearModel
# ==========================================================================


class TestLinearModel(unittest.TestCase):
    def test_eye_contact_ranked_top(self):
        """eye_contact should be top feature."""
        _, cfeat, bts = _synthetic_data()
        result = linear_model_importance(cfeat, bts)
        self.assertTrue(result)
        self.assertEqual(result[0]["feature"], "eye_contact")

    def test_weights_in_range(self):
        """Normalised weights in [0, 1]."""
        _, cfeat, bts = _synthetic_data()
        result = linear_model_importance(cfeat, bts)
        for r in result:
            self.assertGreaterEqual(r["weight"], 0.0)
            self.assertLessEqual(r["weight"], 1.0)

    def test_has_r2(self):
        """Result should contain R²."""
        _, cfeat, bts = _synthetic_data()
        result = linear_model_importance(cfeat, bts)
        for r in result:
            self.assertIn("r2", r)

    def test_empty_features(self):
        """Empty feature dict yields empty result."""
        result = linear_model_importance({}, {})
        self.assertEqual(result, [])


# ==========================================================================
# Test: Evidence — Permutation
# ==========================================================================


class TestPermutationImportance(unittest.TestCase):
    def test_eye_contact_top_by_drop(self):
        """eye_contact should have largest drop."""
        _, cfeat, bts = _synthetic_data()
        result = permutation_importance(cfeat, bts, n_repeats=3)
        self.assertTrue(result)
        self.assertEqual(result[0]["feature"], "eye_contact")

    def test_drop_is_positive(self):
        """Drop should be non-negative."""
        _, cfeat, bts = _synthetic_data()
        result = permutation_importance(cfeat, bts, n_repeats=3)
        for r in result:
            self.assertGreaterEqual(r["drop"], -1e-6)

    def test_empty_data_returns_empty(self):
        """Empty data returns empty list."""
        result = permutation_importance({}, {}, n_repeats=3)
        self.assertEqual(result, [])


# ==========================================================================
# Test: Evidence — compute_evidence (top-level)
# ==========================================================================


class TestComputeEvidence(unittest.TestCase):
    def test_all_methods_present(self):
        """Result dict should have all 4 keys."""
        pairs, cfeat, bts = _synthetic_data()
        result = compute_evidence(pairs, cfeat, bts)
        for key in ("pairwise_delta", "regression", "permutation", "consensus"):
            self.assertIn(key, result)

    def test_consensus_has_eye_contact_top(self):
        """Consensus should rank eye_contact first."""
        pairs, cfeat, bts = _synthetic_data()
        result = compute_evidence(pairs, cfeat, bts)
        cs = result["consensus"]
        self.assertTrue(cs)
        self.assertEqual(cs[0]["feature"], "eye_contact",
                         f"Expected eye_contact, got {cs[0]}")


# ==========================================================================
# Test: Counterfactual
# ==========================================================================


class TestCounterfactual(unittest.TestCase):
    def test_baseline_r2_present(self):
        """Result should contain baseline_r2."""
        _, cfeat, bts = _synthetic_data()
        result = counterfactual_analysis(cfeat, bts)
        self.assertIn("baseline_r2", result)
        self.assertIsInstance(result["baseline_r2"], float)

    def test_all_features_present(self):
        """Counterfactual list should cover all features."""
        _, cfeat, bts = _synthetic_data()
        result = counterfactual_analysis(cfeat, bts)
        features_in_result = {r["feature"] for r in result["counterfactuals"]}
        self.assertEqual(features_in_result, set(FEATURE_NAMES))

    def test_eye_contact_critical(self):
        """eye_contact should be critical/notable."""
        _, cfeat, bts = _synthetic_data()
        result = counterfactual_analysis(cfeat, bts)
        for r in result["counterfactuals"]:
            if r["feature"] == "eye_contact":
                self.assertIn(r["importance"], ("critical", "notable"))
                return
        self.fail("eye_contact not found")

    def test_drop_pct_fields(self):
        """Each counterfactual should have r2_drop, drop_pct, importance."""
        _, cfeat, bts = _synthetic_data()
        result = counterfactual_analysis(cfeat, bts)
        for r in result["counterfactuals"]:
            self.assertIn("r2_drop", r)
            self.assertIn("drop_pct", r)
            self.assertIn("importance", r)


# ==========================================================================
# Test: Stability
# ==========================================================================


class TestStability(unittest.TestCase):
    def test_stable_with_synthetic_data(self):
        """Should return stable or moderately_stable with synthetic data."""
        _, cfeat, bts = _synthetic_data(n_clips=50)
        result = stability_check(cfeat, bts, n_bootstrap=20)
        self.assertIn(result["verdict"], ("stable", "moderately_stable"))
        self.assertIn("kendall_tau", result)

    def test_insufficient_data(self):
        """Very few clips → insufficient_data."""
        cfeat = {f"c{i}": {f: 0.0 for f in FEATURE_NAMES} for i in range(3)}
        bts = {f"c{i}": 0.0 for i in range(3)}
        result = stability_check(cfeat, bts, n_bootstrap=5)
        self.assertEqual(result["verdict"], "insufficient_data")

    def test_top_features_listed(self):
        """Should list top features."""
        _, cfeat, bts = _synthetic_data(n_clips=50)
        result = stability_check(cfeat, bts, n_bootstrap=20)
        self.assertIn("top_features", result)
        self.assertEqual(len(result["top_features"]), result["top_k"])


# ==========================================================================
# Test: Report
# ==========================================================================


class TestReport(unittest.TestCase):
    def test_all_sections_present(self):
        """Report should contain all top-level keys."""
        pairs, cfeat, bts = _synthetic_data(n_clips=50, n_pairs=100)
        report = generate_report(pairs, cfeat, bts, n_bootstrap=20)
        for key in ("summary", "evidence", "counterfactual", "stability", "clusters", "confidence"):
            self.assertIn(key, report, f"Missing key: {key}")

    def test_summary_is_string(self):
        """Summary should be a non-empty string."""
        pairs, cfeat, bts = _synthetic_data(n_clips=50, n_pairs=100)
        report = generate_report(pairs, cfeat, bts, n_bootstrap=20)
        self.assertIsInstance(report["summary"], str)
        self.assertTrue(len(report["summary"]) > 0)

    def test_confidence_counts(self):
        """Confidence should include n_pairs and n_clips."""
        pairs, cfeat, bts = _synthetic_data(n_clips=50, n_pairs=100)
        report = generate_report(pairs, cfeat, bts, n_bootstrap=20)
        self.assertEqual(report["confidence"]["n_pairs"], 100)
        self.assertEqual(report["confidence"]["n_clips"], 50)

    def test_clusters_list(self):
        """Clusters should be a list (may be empty with few clips)."""
        pairs, cfeat, bts = _synthetic_data(n_clips=20, n_pairs=30)
        report = generate_report(pairs, cfeat, bts, n_bootstrap=10)
        self.assertIsInstance(report["clusters"], list)

    def test_report_json_serialisable(self):
        """Report should be JSON-serialisable."""
        import json
        pairs, cfeat, bts = _synthetic_data(n_clips=30, n_pairs=50)
        report = generate_report(pairs, cfeat, bts, n_bootstrap=10)
        try:
            json.dumps(report)
        except (TypeError, ValueError) as e:
            self.fail(f"Report not JSON-serialisable: {e}")


# ==========================================================================
# Test: Edge cases
# ==========================================================================


class TestEdgeCases(unittest.TestCase):
    def test_empty_pairs_empty_report(self):
        """No pairs should produce a report with 0 n_pairs."""
        cfeat = {f"c{i}": {f: 0.1 for f in FEATURE_NAMES} for i in range(5)}
        bts = {f"c{i}": 0.0 for i in range(5)}
        report = generate_report([], cfeat, bts, n_bootstrap=5)
        self.assertEqual(report["confidence"]["n_pairs"], 0)

    def test_single_feature_value(self):
        """All features identical → no evidence divergence."""
        cfeat = {f"c{i}": {f: 0.5 for f in FEATURE_NAMES} for i in range(10)}
        bts = {f"c{i}": float(i) for i in range(10)}
        pairs = [PairwiseSample("u", "c0", "c1", "A")]
        report = generate_report(pairs, cfeat, bts, n_bootstrap=5)
        # Should not crash
        self.assertIn("evidence", report)

    def test_missing_feature_values(self):
        """Some clips missing feature keys should not crash."""
        cfeat = {f"c{i}": {} for i in range(10)}
        bts = {f"c{i}": 0.0 for i in range(10)}
        pairs = [PairwiseSample("u", "c0", "c1", "A")]
        report = generate_report(pairs, cfeat, bts, n_bootstrap=5)
        self.assertIn("evidence", report)


if __name__ == "__main__":
    unittest.main()
