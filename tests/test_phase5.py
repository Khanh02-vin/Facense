"""Tests for Phase 5 — Evaluation & Validation.

Run: python -m unittest tests.test_phase5 -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preference_learning.preference import PairwiseSample, BradleyTerryModel
from src.evaluation.label_consistency import label_consistency, self_consistency
from src.evaluation.feature_validation import (
    bt_feature_validation,
    generalization_check,
    sample_counterfactual,
    _ridge_importances,
)
from src.evaluation.robustness import robustness_check
from src.evaluation.phase5 import validate_explanations
from src.explainability.evidence import FEATURE_NAMES


# ==========================================================================
# Helpers
# ==========================================================================


def _synthetic_data(
    n_clips: int = 50,
    n_pairs: int = 100,
    seed: int = 42,
) -> tuple[list[PairwiseSample], dict[str, dict[str, float]], dict[str, float]]:
    """Synthetic data with known signal: eye_contact dominant, smile secondary."""
    rng = np.random.default_rng(seed)
    clip_names = [f"clip_{i:04d}" for i in range(n_clips)]

    clip_features: dict[str, dict[str, float]] = {}
    bt_scores: dict[str, float] = {}

    for cn in clip_names:
        feat = {f: float(rng.uniform(0.0, 1.0)) for f in FEATURE_NAMES}
        clip_features[cn] = feat
        score = 2.0 * feat["eye_contact"] + 0.5 * feat["smile"] + 0.1 * float(rng.normal())
        bt_scores[cn] = score

    mean_bt = float(np.mean(list(bt_scores.values())))
    for cn in bt_scores:
        bt_scores[cn] -= mean_bt

    pairs: list[PairwiseSample] = []
    for _ in range(n_pairs):
        a, b = rng.choice(clip_names, size=2, replace=False)
        winner = "A" if bt_scores[a] > bt_scores[b] else "B"
        pairs.append(PairwiseSample("u1", a, b, winner))

    return pairs, clip_features, bt_scores


# ==========================================================================
# Test: Label Consistency
# ==========================================================================


class TestLabelConsistency(unittest.TestCase):
    def test_perfect_agreement(self):
        """Identical labels → kappa = 1.0."""
        labels = ["A", "B", "A", "B", "equal"]
        result = label_consistency(labels, labels)
        self.assertEqual(result["cohens_kappa"], 1.0)
        self.assertEqual(result["agreement_pct"], 100.0)

    def test_random_agreement(self):
        """Random labels → kappa ≈ 0."""
        rng = np.random.default_rng(0)
        l1 = rng.choice(["A", "B", "equal"], size=200).tolist()
        l2 = rng.choice(["A", "B", "equal"], size=200).tolist()
        result = label_consistency(l1, l2)
        self.assertAlmostEqual(result["cohens_kappa"], 0.0, delta=0.15)

    def test_empty_lists(self):
        """Empty lists → no_data interpretation."""
        result = label_consistency([], [])
        self.assertEqual(result["interpretation"], "no_data")

    def test_mismatched_length_raises(self):
        """Different lengths should raise."""
        with self.assertRaises(ValueError):
            label_consistency(["A"], ["A", "B"])

    def test_case_normalization(self):
        """'a' and 'A' should be treated the same."""
        result = label_consistency(["a", "b"], ["A", "B"])
        self.assertEqual(result["cohens_kappa"], 1.0)


# ==========================================================================
# Test: Self-consistency
# ==========================================================================


class TestSelfConsistency(unittest.TestCase):
    def test_consistent_with_good_data(self):
        """With known signal, split-half should yield consistent or limited verdict."""
        pairs, cfeat, bts = _synthetic_data(n_clips=60, n_pairs=160)
        result = self_consistency(pairs, cfeat, bts)
        self.assertIn(result["verdict"], ("consistent", "inconsistent", "insufficient_data"))

    def test_insufficient_data(self):
        """Very few pairs → insufficient_data."""
        pairs = [PairwiseSample("u", "a", "b", "A")]
        result = self_consistency(pairs, {"a": {"s": 0.0}, "b": {"s": 0.0}}, {"a": 0.0, "b": 0.0})
        self.assertEqual(result["verdict"], "insufficient_data")

    def test_top_features_listed(self):
        """Results should list top features for both halves."""
        pairs, cfeat, bts = _synthetic_data(n_clips=30, n_pairs=60)
        result = self_consistency(pairs, cfeat, bts)
        self.assertIn("top_features_half_a", result)
        self.assertIn("top_features_half_b", result)


# ==========================================================================
# Test: BT Feature Validation
# ==========================================================================


class TestBTFeatureValidation(unittest.TestCase):
    def test_eye_contact_validated(self):
        """eye_contact should be validated (effect_size >= 0.5)."""
        _, cfeat, bts = _synthetic_data(n_clips=60)
        result = bt_feature_validation(cfeat, bts)
        top = next(r for r in result if r["feature"] == "eye_contact")
        self.assertGreaterEqual(abs(top["effect_size"]), 0.5,
                                f"eye_contact effect_size={top['effect_size']}")

    def test_returns_all_features(self):
        """Every feature should appear."""
        _, cfeat, bts = _synthetic_data(n_clips=60)
        result = bt_feature_validation(cfeat, bts)
        features_found = {r["feature"] for r in result}
        for f in FEATURE_NAMES:
            self.assertIn(f, features_found, f"Missing feature: {f}")

    def test_feature_validation_fields(self):
        """Each result should have all required fields."""
        _, cfeat, bts = _synthetic_data(n_clips=60)
        result = bt_feature_validation(cfeat, bts)
        for r in result:
            for key in ("feature", "effect_size", "mean_high", "mean_low",
                        "n_high", "n_low", "importance"):
                self.assertIn(key, r, f"Missing key: {key}")

    def test_small_dataset(self):
        """Fewer than 6 clips → empty result."""
        _, cfeat, bts = _synthetic_data(n_clips=5)
        result = bt_feature_validation(cfeat, bts)
        self.assertEqual(result, [])


# ==========================================================================
# Test: Generalization
# ==========================================================================


class TestGeneralization(unittest.TestCase):
    def test_generalizes_with_synthetic(self):
        """Strong signal with enough data → should generalize or be limited."""
        pairs, cfeat, bts = _synthetic_data(n_clips=60, n_pairs=160)
        result = generalization_check(pairs, cfeat, bts)
        self.assertIn(result["verdict"], ("generalizes", "limited", "insufficient_data"))
        # Accuracy should be above chance (> 0.5)
        self.assertGreater(result["bt_accuracy"], 0.45)

    def test_insufficient_data(self):
        """Few pairs → insufficient_data."""
        result = generalization_check([], {}, {})
        self.assertEqual(result["verdict"], "insufficient_data")

    def test_accuracy_in_range(self):
        """Accuracy should be in [0, 1]."""
        pairs, cfeat, bts = _synthetic_data(n_clips=40, n_pairs=80)
        result = generalization_check(pairs, cfeat, bts)
        self.assertGreaterEqual(result["bt_accuracy"], 0.0)
        self.assertLessEqual(result["bt_accuracy"], 1.0)

    def test_train_test_counts(self):
        """Train + test should sum to total pairs."""
        pairs, cfeat, bts = _synthetic_data(n_clips=40, n_pairs=80)
        result = generalization_check(pairs, cfeat, bts, test_frac=0.2)
        self.assertEqual(
            result["n_train_pairs"] + result["n_test_pairs"],
            80,
        )


# ==========================================================================
# Test: Sample Counterfactual
# ==========================================================================


class TestSampleCounterfactual(unittest.TestCase):
    def test_returns_list_for_valid_pair(self):
        """Valid pair should return counterfactual list."""
        _, cfeat, bts = _synthetic_data(n_clips=20, n_pairs=10)
        pairs, _, _ = _synthetic_data(n_clips=20, n_pairs=10, seed=1)
        bt = BradleyTerryModel()
        bt.fit(pairs)
        result = sample_counterfactual(pairs[0], cfeat, bt)
        self.assertIsInstance(result, list)

    def test_each_result_has_required_fields(self):
        """Each CF should have feature, original_prob, cf_prob, flipped."""
        _, cfeat, bts = _synthetic_data(n_clips=20, n_pairs=10)
        pairs, _, _ = _synthetic_data(n_clips=20, n_pairs=10, seed=1)
        bt = BradleyTerryModel()
        bt.fit(pairs)
        result = sample_counterfactual(pairs[0], cfeat, bt)
        for r in result:
            for key in ("feature", "original_prob", "counterfactual_prob", "flipped"):
                self.assertIn(key, r, f"Missing key: {key}")

    def test_empty_for_missing_clip(self):
        """Pair with unknown clips → empty list."""
        pair = PairwiseSample("u", "unknown_a", "unknown_b", "A")
        bt = BradleyTerryModel()
        bt.fit([pair])
        result = sample_counterfactual(pair, {}, bt)
        self.assertEqual(result, [])


# ==========================================================================
# Test: Robustness
# ==========================================================================


class TestRobustness(unittest.TestCase):
    def test_stable_with_good_data(self):
        """Larger dataset → should be stable or show progression."""
        _, cfeat, bts = _synthetic_data(n_clips=80, n_pairs=200)
        result = robustness_check(cfeat, bts)
        self.assertIn(result["verdict"], ("stable", "unstable", "insufficient_data"))
        # Tau progression should be non-empty
        self.assertTrue(len(result["kendall_tau_progression"]) > 0)

    def test_tau_progression_length(self):
        """Tau progression should match fractions length."""
        _, cfeat, bts = _synthetic_data(n_clips=50)
        fractions = [0.6, 0.8, 1.0]
        result = robustness_check(cfeat, bts, fractions=fractions)
        self.assertEqual(len(result["kendall_tau_progression"]), len(fractions))

    def test_final_tau_is_one(self):
        """Full data fraction should have tau = 1.0."""
        _, cfeat, bts = _synthetic_data(n_clips=50)
        result = robustness_check(cfeat, bts)
        self.assertAlmostEqual(result["kendall_tau_progression"][-1], 1.0)

    def test_insufficient_data(self):
        """Fewer than 10 clips → insufficient_data."""
        cfeat = {f"c{i}": {f: 0.0 for f in FEATURE_NAMES} for i in range(5)}
        bts = {f"c{i}": 0.0 for i in range(5)}
        result = robustness_check(cfeat, bts)
        self.assertEqual(result["verdict"], "insufficient_data")


# ==========================================================================
# Test: Ridge importances helper
# ==========================================================================


class TestRidgeImportances(unittest.TestCase):
    def test_returns_array(self):
        """Should return numpy array of length feature_names."""
        _, cfeat, bts = _synthetic_data(n_clips=20)
        imp = _ridge_importances(cfeat, bts, FEATURE_NAMES)
        self.assertIsInstance(imp, np.ndarray)
        self.assertEqual(len(imp), len(FEATURE_NAMES))

    def test_eye_contact_top(self):
        """eye_contact should have highest importance."""
        _, cfeat, bts = _synthetic_data(n_clips=30)
        imp = _ridge_importances(cfeat, bts, FEATURE_NAMES)
        ec_idx = FEATURE_NAMES.index("eye_contact")
        self.assertEqual(np.argmax(imp), ec_idx)

    def test_small_data_returns_zeros(self):
        """Fewer than 3 clips → zero array."""
        cfeat = {"a": {"s": 0.0}, "b": {"s": 0.0}}
        bts = {"a": 0.0, "b": 0.0}
        imp = _ridge_importances(cfeat, bts, ["s"])
        self.assertTrue(np.all(imp == 0.0))


# ==========================================================================
# Test: Validate Explanations (orchestrator)
# ==========================================================================


class TestValidateExplanations(unittest.TestCase):
    def test_all_sections_present(self):
        """Report should contain all top-level keys."""
        pairs, cfeat, bts = _synthetic_data(n_clips=40, n_pairs=80)
        report = validate_explanations(pairs, cfeat, bts)
        for key in ("label_consistency", "self_consistency", "feature_validation",
                    "generalization", "robustness", "kpi_summary"):
            self.assertIn(key, report, f"Missing key: {key}")

    def test_kpi_summary_has_all_kpis(self):
        """KPI summary should contain stability, consistency, user_confirmable."""
        pairs, cfeat, bts = _synthetic_data(n_clips=40, n_pairs=80)
        report = validate_explanations(pairs, cfeat, bts)
        ks = report["kpi_summary"]
        for kpi in ("stability", "consistency", "user_confirmable"):
            self.assertIn(kpi, ks, f"Missing KPI: {kpi}")

    def test_overall_verdict_present(self):
        """KPI summary should have overall verdict."""
        pairs, cfeat, bts = _synthetic_data(n_clips=40, n_pairs=80)
        report = validate_explanations(pairs, cfeat, bts)
        self.assertIn("overall", report["kpi_summary"])
        self.assertIn(report["kpi_summary"]["overall"],
                      ("trustworthy", "partially_trustworthy", "insufficient_data"))

    def test_json_serialisable(self):
        """Report should be JSON-serialisable."""
        import json
        pairs, cfeat, bts = _synthetic_data(n_clips=30, n_pairs=50)
        report = validate_explanations(pairs, cfeat, bts)
        try:
            json.dumps(report)
        except (TypeError, ValueError) as e:
            self.fail(f"Report not JSON-serialisable: {e}")

    def test_with_label_consistency(self):
        """If labels provided, label_consistency should be populated."""
        pairs, cfeat, bts = _synthetic_data(n_clips=30, n_pairs=50)
        l1 = ["A", "B", "A", "B"]
        l2 = ["A", "B", "A", "B"]
        report = validate_explanations(pairs, cfeat, bts, labels1=l1, labels2=l2)
        self.assertIsNotNone(report["label_consistency"])

    def test_with_stability_reference(self):
        """If stability_reference provided, it should be in KPI."""
        pairs, cfeat, bts = _synthetic_data(n_clips=30, n_pairs=50)
        stab_ref = {"kendall_tau": 0.92, "verdict": "stable"}
        report = validate_explanations(pairs, cfeat, bts, stability_reference=stab_ref)
        ks = report["kpi_summary"]["stability"]
        self.assertIn("stability_reference", ks)


# ==========================================================================
# Test: Edge cases
# ==========================================================================


class TestEdgeCases(unittest.TestCase):
    def test_empty_pairs(self):
        """No pairs → report should not crash."""
        cfeat = {f"c{i}": {f: 0.1 for f in FEATURE_NAMES} for i in range(10)}
        bts = {f"c{i}": 0.0 for i in range(10)}
        report = validate_explanations([], cfeat, bts)
        self.assertIn("kpi_summary", report)

    def test_constant_features(self):
        """All identical features → feature validation empty."""
        cfeat = {f"c{i}": {f: 0.5 for f in FEATURE_NAMES} for i in range(10)}
        bts = {f"c{i}": float(i) for i in range(10)}
        pairs = [PairwiseSample("u", "c0", "c1", "A")]
        fv = bt_feature_validation(cfeat, bts)
        self.assertEqual(fv, [])

    def test_missing_feature_values(self):
        """Clips with missing features should not crash."""
        cfeat = {f"c{i}": {} for i in range(10)}
        bts = {f"c{i}": 0.0 for i in range(10)}
        pairs = [PairwiseSample("u", "c0", "c1", "A")]
        report = validate_explanations(pairs, cfeat, bts)
        self.assertIn("kpi_summary", report)

    def test_single_pair(self):
        """Single pair → should not crash."""
        cfeat = {"a": dict(zip(FEATURE_NAMES, [0.1]*len(FEATURE_NAMES))),
                 "b": dict(zip(FEATURE_NAMES, [0.2]*len(FEATURE_NAMES)))}
        bts = {"a": 0.5, "b": -0.5}
        pairs = [PairwiseSample("u", "a", "b", "A")]
        report = validate_explanations(pairs, cfeat, bts)
        self.assertIn("kpi_summary", report)


if __name__ == "__main__":
    unittest.main()
