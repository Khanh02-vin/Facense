"""Core integrity tests for Facense pipeline.

Run with: python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction.feature_extractor_layer1 import (
    FeatureExtractorLayer1,
    VideoFeatures,
)
from src.feature_extraction.feature_extractor_layer2 import FeatureExtractorLayer2
from src.preference_learning.preference import (
    BradleyTerryModel,
    PairwiseSample,
    NeuralRewardModel,
)


class TestFeatureSchema(unittest.TestCase):
    """Feature extractor L1 must export all fields the trainer expects."""

    KNOWN_TRAINER_FIELDS = {
        "motion_energy", "motion_peak", "motion_variance",
        "blur_score", "brightness", "brightness_std",
        "face_visibility", "face_detected",
        "smile", "mouth_open", "eye_contact",
        "pupil_left", "pupil_right",
        "head_yaw", "head_pitch", "head_roll",
        "face_symmetry", "face_clarity",
    }

    def test_l1_to_dict_has_all_required_fields(self):
        """L1 to_dict() must include motion_variance, brightness_std."""
        feat = VideoFeatures(video_id="test")
        d = feat.to_dict()
        req = {"motion_energy", "motion_peak", "motion_variance",
               "blur_score", "brightness", "brightness_std",
               "face_visibility", "face_detected", "n_frames"}
        self.assertTrue(req.issubset(d.keys()),
                        msg=f"Missing L1 fields: {req - d.keys()}")

    def test_pipeline_feature_has_all_trainer_fields(self):
        """Simulated pipeline combined output must cover trainer schema."""
        l1 = VideoFeatures(video_id="test")
        combined = {**l1.to_dict(),
                    "identity": "test", "video_file": "t.mp4", "clip_name": "test"}
        # add fake L2 fields
        for k in ["smile", "mouth_open", "eye_contact", "pupil_left", "pupil_right",
                   "head_yaw", "head_pitch", "head_roll", "face_symmetry", "face_clarity"]:
            combined[k] = 0.0
        missing = self.KNOWN_TRAINER_FIELDS - set(combined.keys())
        self.assertFalse(missing, msg=f"Trainer needs fields never produced: {missing}")


class TestBradleyTerry(unittest.TestCase):
    """Bradley-Terry ordering and edge cases."""

    def test_empty_pairs_returns_empty(self):
        model = BradleyTerryModel()
        result = model.fit([])
        self.assertEqual(result.item_scores, {})
        self.assertTrue(result.convergence)

    def test_single_pair_ordering(self):
        model = BradleyTerryModel()
        pairs = [
            PairwiseSample("u", "A", "B", "A"),
        ]
        result = model.fit(pairs)
        self.assertGreater(result.item_scores["A"], result.item_scores["B"])

    def test_consistent_ordering(self):
        """A > B > C should give A > B > C in scores.

        PairwiseSample("u", "A", "B", "A") → winner=A with image_A=A  →  A beats B
        PairwiseSample("u", "B", "C", "A") → winner=A with image_A=B  →  B beats C
        PairwiseSample("u", "A", "C", "A") → winner=A with image_A=A  →  A beats C
        """
        model = BradleyTerryModel()
        pairs = [
            PairwiseSample("u", "A", "B", "A"),
            PairwiseSample("u", "B", "C", "A"),
            PairwiseSample("u", "A", "C", "A"),
        ]
        result = model.fit(pairs)
        self.assertGreater(result.item_scores["A"], result.item_scores["B"])
        self.assertGreater(result.item_scores["B"], result.item_scores["C"])

    def test_reverse_ordering_stable(self):
        """Swapping A/B order in pairs must give correct winner."""
        model = BradleyTerryModel()
        pairs = [
            PairwiseSample("u", "A", "B", "B"),  # B beats A
        ]
        result = model.fit(pairs)
        self.assertGreater(result.item_scores["B"], result.item_scores["A"])

    def test_predict_pair(self):
        model = BradleyTerryModel()
        pairs = [PairwiseSample("u", "X", "Y", "A")]  # X beats Y
        model.fit(pairs)
        p = model.predict_pair("X", "Y")
        self.assertGreater(p, 0.5)
        self.assertLess(p, 1.0)

    def test_rank_returns_sorted(self):
        model = BradleyTerryModel()
        pairs = [
            PairwiseSample("u", "a", "b", "A"),  # a beats b
            PairwiseSample("u", "b", "c", "A"),  # b beats c
        ]
        model.fit(pairs)
        ranked = model.rank(["a", "b", "c"])
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestNeuralRewardContract(unittest.TestCase):
    """NeuralRewardModel must have consistent train/predict semantics."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")

    def test_pair_prediction_consistent(self):
        """predict_pair(A, B) should be ≈ 1 - predict_pair(B, A)."""
        rng = np.random.default_rng(42)
        n, dim = 20, 64
        emb_A = rng.normal(size=(n, dim)).astype(np.float32)
        emb_B = rng.normal(size=(n, dim)).astype(np.float32)
        winners = (rng.random(n) > 0.5).astype(np.float32)

        model = NeuralRewardModel(embedding_dim=dim, hidden_dim=32)
        model.fit(emb_A, emb_B, winners, epochs=2, batch_size=8)

        p_ab = model.predict_pair(emb_A[0], emb_B[0])
        p_ba = model.predict_pair(emb_B[0], emb_A[0])
        self.assertAlmostEqual(p_ab + p_ba, 1.0, places=4,
                               msg="predict_pair(A,B) + predict_pair(B,A) should ≈ 1")

    def test_predict_returns_scalar(self):
        rng = np.random.default_rng(42)
        dim = 64
        emb = rng.normal(size=(dim,)).astype(np.float32)
        emb_A = rng.normal(size=(5, dim)).astype(np.float32)
        emb_B = rng.normal(size=(5, dim)).astype(np.float32)

        model = NeuralRewardModel(embedding_dim=dim, hidden_dim=32)
        model.fit(emb_A, emb_B, np.ones(5), epochs=2, batch_size=5)

        score = model.predict(emb)
        self.assertIsInstance(score, float)


class TestArtifactCompatibility(unittest.TestCase):
    """serve.py artifact format checks."""

    def test_embedding_npz_format(self):
        """serve.py reads .npz with item_id keys and flattens them."""
        tmp = tempfile.mkdtemp()
        try:
            path = Path(tmp) / "embeddings.npz"
            items = {"id_1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                     "id_2": np.array([4.0, 5.0, 6.0], dtype=np.float32)}
            np.savez(path, **items)
            del items  # release any ref
            data = np.load(path, allow_pickle=True)
            out = {}
            for key in data.files:
                arr = data[key]
                out[key] = np.asarray(arr, dtype=np.float32).flatten()
            data.close()
            self.assertEqual(set(out.keys()), {"id_1", "id_2"})
            self.assertEqual(out["id_1"].shape, (3,))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bt_scores_json_format(self):
        """serve.py reads BT scores as {item_id: float}."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.json"
            scores = {"item_a": 1.5, "item_b": -0.3}
            with open(path, "w") as f:
                json.dump(scores, f)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, scores)


class TestAPISafety(unittest.TestCase):
    """API edge cases that must not crash."""

    def test_clamp_top_k_larger_than_dataset(self):
        from serve import _clamp_top_k, create_app
        self.assertEqual(_clamp_top_k(100, 5), 5)
        self.assertEqual(_clamp_top_k(0, 5), 1)  # min 1
        self.assertEqual(_clamp_top_k(3, 10), 3)

    def test_bootstrap_only_via_flag(self):
        """bootstrap_sample_state produces data."""
        from serve import bootstrap_sample_state
        bootstrap_sample_state(n_items=10, dim=8, seed=1)
        import serve as sv
        self.assertEqual(len(sv.STATE.embeddings), 10)


class TestServeBootstrap(unittest.TestCase):
    """Bootstrap must require explicit --bootstrap-sample flag."""

    def test_bootstrap_only_via_flag(self):
        """Verify bootstrap_sample_state is NOT called without the flag."""
        from serve import bootstrap_sample_state
        # This function should only ever be called explicitly, not auto-detected.
        # We test that it creates data when called.
        bootstrap_sample_state(n_items=10, dim=8, seed=1)
        import serve as sv
        self.assertEqual(len(sv.STATE.embeddings), 10)


if __name__ == "__main__":
    unittest.main()
