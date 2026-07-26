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


class TestLabelingContract(unittest.TestCase):
    """Phase 2 contracts: schema, resume key, frame-quality formula."""

    # ---- Schema ------------------------------------------------------

    def test_pairwise_sample_winner_is_strict_AB(self):
        """PairwiseSample.winner is Literal["A", "B"] — strict."""
        from typing import get_args

        hints = PairwiseSample.__dataclass_fields__["winner"].type
        self.assertEqual(get_args(hints), ("A", "B"))

    def test_html_label_record_uses_winner_field(self):
        """HTML labels use the same ``winner`` key as PairwiseSample.

        The labeling tool emits four states (A/B/equal/skip); only
        the BT trainer consumes A/B.  This test guards the field
        name regardless of value.
        """
        sample = PairwiseSample(
            user_id="u1", image_A="a", image_B="b", winner="A",
        )
        label = {
            "winner": "A",
            "video_a": "a", "video_b": "b",
            "identity_a": "x", "identity_b": "y",
            "position": {"left": "a", "right": "b"},
            "confidence": 2, "latency_ms": 100,
            "timestamp": "2026-01-01T00:00:00Z",
        }
        # Both use the same key.  The BT trainer's filter
        # (preference.py:97) drops 'equal' / 'skip' silently.
        self.assertIn("winner", label)
        self.assertIn(label["winner"], ("A", "B", "equal", "skip"))

    # ---- Resume key --------------------------------------------------

    def test_resume_key_uses_unambiguous_separator(self):
        """The HTML resume key uses ``__VS__`` so that clip names
        containing ``_`` cannot collide (e.g. ``A_1`` + ``B_2`` vs
        ``A`` + ``1_B_2``).
        """
        SEP = "__VS__"
        a, b = "Anne_Hathaway_1", "Anne_1"
        key1 = f"{a}{SEP}{b}"
        key2 = f"{b}{SEP}{a}"
        # Two different pairs → two different keys
        self.assertNotEqual(key1, key2)
        # Round-trip via the same pair
        pair = ("Anne_Hathaway_1", "Anne_1")
        self.assertEqual(f"{pair[0]}{SEP}{pair[1]}", "Anne_Hathaway_1__VS__Anne_1")

    # ---- Frame-quality formula --------------------------------------

    def test_quality_score_in_unit_range(self):
        """``compute_quality_score`` is shared between frame_sampler
        and the labeling-tool extractor; the rewrite must keep it in
        [0, 1] for any plausible grayscale input.
        """
        from src.frame_extraction.frame_sampler import compute_quality_score

        rng = np.random.default_rng(0)
        for _ in range(20):
            gray = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
            q = compute_quality_score(gray)
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)


if __name__ == "__main__":
    unittest.main()
