"""
Test Module - Unit tests for Face Project modules
"""

import numpy as np
import unittest
from unittest.mock import MagicMock, patch


class TestEmbedding(unittest.TestCase):
    """Tests for embedding module."""

    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass."""
        from src.embedding import EmbeddingResult

        result = EmbeddingResult(
            embedding=np.array([0.1, 0.2, 0.3]),
            model="test",
            image_id="img_001",
            timestamp=1234567890.0
        )

        self.assertEqual(result.image_id, "img_001")
        self.assertEqual(result.model, "test")
        self.assertEqual(len(result.embedding), 3)

    def test_appearance_embedding_init(self):
        """Test AppearanceEmbedding initialization."""
        from src.embedding import AppearanceEmbedding

        emb = AppearanceEmbedding(model_name="siglip", normalize=True)

        self.assertEqual(emb.model_name, "siglip")
        self.assertTrue(emb.normalize)
        self.assertIsNone(emb.model)

    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_detection(self, mock_cuda):
        """Test CUDA detection fallback."""
        from src.embedding import AppearanceEmbedding

        emb = AppearanceEmbedding(model_name="siglip")
        self.assertEqual(emb.device, "cpu")


class TestValidation(unittest.TestCase):
    """Tests for validation module."""

    def test_stability_validator(self):
        """Test StabilityValidator."""
        from src.validation import StabilityValidator

        validator = StabilityValidator(threshold=0.85)

        # Create test embeddings
        original = np.random.randn(128)
        original = original / np.linalg.norm(original)

        augmented = [
            original + np.random.randn(128) * 0.01
            for _ in range(5)
        ]
        augmented = [a / np.linalg.norm(a) for a in augmented]

        result = validator.test(original, augmented)

        self.assertEqual(result.test_name, "stability")
        # Result can be bool or np.bool_
        self.assertTrue(isinstance(result.passed, (bool, np.bool_)))

    def test_feature_importance_validator(self):
        """Test FeatureImportanceValidator."""
        from src.validation import FeatureImportanceValidator

        validator = FeatureImportanceValidator()

        # Create synthetic data
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        labels = (embeddings[:, 0] > 0).astype(int)

        result = validator.compute_ablation_scores(embeddings, labels, n_ablations=10)

        self.assertIn("original_auc", result)
        self.assertIn("ablation_scores", result)
        self.assertIsInstance(result["ablation_scores"], list)

    def test_cross_model_validator(self):
        """Test CrossModelValidator."""
        from src.validation import CrossModelValidator

        validator = CrossModelValidator()

        predictions = {
            "siglip": np.random.rand(100),
            "dinov2": np.random.rand(100),
            "clip": np.random.rand(100)
        }

        result = validator.compute_agreement(predictions)

        self.assertIn("mean_correlation", result)
        self.assertEqual(result["n_models"], 3)


class TestNullModels(unittest.TestCase):
    """Tests for null models module."""

    def test_label_permutation_null(self):
        """Test LabelPermutationNull."""
        from src.null_models import LabelPermutationNull

        null_test = LabelPermutationNull(n_permutations=100, alpha=0.05)

        # Create synthetic data
        np.random.seed(42)
        embeddings = np.random.randn(200, 32)
        labels = (embeddings[:, 0] > 0).astype(int)

        result = null_test.test(embeddings, labels)

        self.assertEqual(result.test_name, "label_permutation")
        # null_rejected can be bool or np.bool_
        self.assertTrue(isinstance(result.null_rejected, (bool, np.bool_)))
        self.assertGreaterEqual(result.p_value, 0)
        self.assertLessEqual(result.p_value, 1)

    def test_feature_shuffle_null(self):
        """Test FeatureShuffleNull."""
        from src.null_models import FeatureShuffleNull

        null_test = FeatureShuffleNull(n_permutations=50, alpha=0.05)

        np.random.seed(42)
        embeddings = np.random.randn(100, 32)
        labels = (embeddings[:, 0] > 0).astype(int)

        result = null_test.test(embeddings, labels)

        self.assertEqual(result.test_name, "feature_shuffle")

    def test_null_suite(self):
        """Test NullModelSuite."""
        from src.null_models import NullModelSuite

        np.random.seed(42)
        embeddings = np.random.randn(100, 32)
        labels = (embeddings[:, 0] > 0).astype(int)
        user_ids = np.array([f"user_{i % 10}" for i in range(100)])

        suite = NullModelSuite(n_permutations=50, alpha=0.05)
        results = suite.run_all(embeddings, labels, user_ids)

        self.assertIn("label_permutation", results)
        self.assertIn("feature_shuffle", results)
        self.assertIn("cross_user", results)
        self.assertIn("summary", results)


class TestPreference(unittest.TestCase):
    """Tests for preference module."""

    def test_pairwise_sample_dataclass(self):
        """Test PairwiseSample dataclass."""
        from src.preference import PairwiseSample

        sample = PairwiseSample(
            user_id="user_001",
            image_A="img_001",
            image_B="img_002",
            winner="A",
            timestamp=1234567890.0
        )

        self.assertEqual(sample.winner, "A")

    def test_bradley_terry_empty(self):
        """Test BradleyTerryModel with empty data."""
        from src.preference import BradleyTerryModel

        model = BradleyTerryModel()
        result = model.fit([])

        self.assertEqual(result.item_scores, {})
        self.assertTrue(result.convergence)

    def test_bradley_terry_fit(self):
        """Test BradleyTerryModel fitting."""
        from src.preference import BradleyTerryModel, PairwiseSample

        model = BradleyTerryModel(max_iterations=50)

        pairs = [
            PairwiseSample("u1", "a", "b", "A"),
            PairwiseSample("u1", "a", "c", "A"),
            PairwiseSample("u1", "b", "c", "B"),
        ]

        result = model.fit(pairs)

        self.assertIsInstance(result.item_scores, dict)
        self.assertTrue(len(result.item_scores) > 0)

    def test_mixture_of_prototypes(self):
        """Test MixtureOfPrototypes."""
        from src.preference import MixtureOfPrototypes

        np.random.seed(42)
        positive = np.random.randn(50, 64)

        model = MixtureOfPrototypes(n_prototypes=3)
        result = model.fit(positive)

        self.assertIn("n_iterations", result)
        self.assertEqual(model.prototypes.shape[0], 3)

        # Test scoring
        test_emb = np.random.randn(64)
        score = model.score(test_emb)
        self.assertIsInstance(score, float)


class TestRetrieval(unittest.TestCase):
    """Tests for retrieval module."""

    def test_retrieval_engine_init(self):
        """Test RetrievalEngine initialization."""
        try:
            from src.retrieval import RetrievalEngine
        except ModuleNotFoundError:
            self.skipTest("sklearn not available")

        np.random.seed(42)
        embeddings = {
            f"img_{i}": np.random.randn(64) for i in range(20)
        }

        engine = RetrievalEngine(embeddings)

        self.assertEqual(len(engine.ids), 20)
        self.assertEqual(engine.matrix.shape[0], 20)

    def test_retrieval_engine_retrieve(self):
        """Test RetrievalEngine retrieval."""
        try:
            from src.retrieval import RetrievalEngine
        except ModuleNotFoundError:
            self.skipTest("sklearn not available")

        np.random.seed(42)
        embeddings = {
            f"img_{i}": np.random.randn(64) for i in range(20)
        }

        engine = RetrievalEngine(embeddings)
        results = engine.retrieve("img_0", k=5)

        self.assertLessEqual(len(results), 5)
        self.assertNotIn("img_0", results)

    def test_retrieval_evaluator(self):
        """Test RetrievalEvaluator."""
        from src.retrieval import RetrievalEvaluator, RetrievalResult, RetrievalMetrics

        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

        result = RetrievalResult(
            query_id="q1",
            candidate_pool=["c1", "c2", "c3", "c4", "c5"],
            retrieved_ids=["c1", "c2", "c3", "c4", "c5"],
            user_ratings=[5, 4, 3, 2, 1]
        )

        metrics = evaluator.evaluate(
            result.query_id,
            result.retrieved_ids,
            result.user_ratings
        )

        self.assertIsInstance(metrics, RetrievalMetrics)
        self.assertGreaterEqual(metrics.mrr, 0)
        self.assertLessEqual(metrics.mrr, 1)


class TestAnnotation(unittest.TestCase):
    """Tests for annotation module."""

    def test_preference_annotation_dataclass(self):
        """Test PreferenceAnnotation dataclass."""
        from src.annotation import PreferenceAnnotation

        ann = PreferenceAnnotation(
            user_id="user_001",
            image_id="img_001",
            timestamp="2026-06-01T12:00:00Z",
            preference_level=5,
            face_score=5,
            hair_score=4
        )

        self.assertEqual(ann.preference_level, 5)
        self.assertEqual(ann.face_score, 5)

    def test_annotation_collection(self):
        """Test AnnotationCollection."""
        from src.annotation import PreferenceAnnotation, AnnotationCollection

        collection = AnnotationCollection()

        collection.add(PreferenceAnnotation(
            user_id="u1", image_id="i1", timestamp="t1"
        ))
        collection.add(PreferenceAnnotation(
            user_id="u1", image_id="i2", timestamp="t2"
        ))
        collection.add(PreferenceAnnotation(
            user_id="u2", image_id="i3", timestamp="t3"
        ))

        self.assertEqual(len(collection.annotations), 3)

        u1_filtered = collection.filter_by_user("u1")
        self.assertEqual(len(u1_filtered.annotations), 2)

    def test_pu_adapter(self):
        """Test PositiveUnlabeledAdapter."""
        from src.annotation import PreferenceAnnotation, AnnotationCollection, PositiveUnlabeledAdapter

        collection = AnnotationCollection()

        for i in range(10):
            collection.add(PreferenceAnnotation(
                user_id="u1",
                image_id=f"img_{i}",
                timestamp="t1",
                preference_level=(i % 5) + 1
            ))

        adapter = PositiveUnlabeledAdapter()
        pairs = adapter.to_pairwise(collection, "u1")

        self.assertIsInstance(pairs, list)

        metrics = adapter.compute_pu_metrics(collection)
        self.assertIn("positive_rate", metrics)


class TestCycleDiagnostics(unittest.TestCase):
    """Tests for cycle diagnostics module."""

    def test_cycle_detector_init(self):
        """Test CycleDetector initialization."""
        from src.cycle_diagnostics import CycleDetector

        detector = CycleDetector(confidence_level=0.95)
        self.assertEqual(detector.confidence_level, 0.95)

    def test_build_pairwise_matrix(self):
        """Test pairwise matrix building."""
        from src.cycle_diagnostics import build_pairwise_matrix

        pairs = [
            ("a", "b", "A"),
            ("a", "c", "A"),
            ("b", "c", "B"),
        ]

        matrix = build_pairwise_matrix(pairs, ["a", "b", "c"])

        self.assertEqual(matrix.shape, (3, 3))
        self.assertGreaterEqual(matrix[0, 1], 0.5)  # a > b
        self.assertGreaterEqual(matrix[0, 2], 0.5)  # a > c

    def test_transitivity_analyzer(self):
        """Test TransitivityAnalyzer."""
        from src.cycle_diagnostics import TransitivityAnalyzer

        analyzer = TransitivityAnalyzer()

        rankings = [
            [0, 1, 2],  # User 1: a > b > c
            [0, 1, 2],  # User 2: a > b > c
            [0, 1, 2],  # User 3: a > b > c
        ]

        W = analyzer.kendall_w(rankings)
        self.assertGreaterEqual(W, 0)
        self.assertLessEqual(W, 1)

    def test_copeland_score(self):
        """Test Copeland score computation."""
        from src.cycle_diagnostics import TransitivityAnalyzer

        analyzer = TransitivityAnalyzer()

        # Perfectly transitive: a > b > c
        matrix = np.array([
            [0.5, 0.8, 0.9],  # a beats b (0.8), a beats c (0.9)
            [0.2, 0.5, 0.8],   # b beats c (0.8)
            [0.1, 0.2, 0.5],   # ties on diagonal
        ])

        scores = analyzer.copeland_score(matrix, ["a", "b", "c"])

        self.assertEqual(len(scores), 3)
        # a should have highest score (beats both)
        self.assertEqual(scores[0][0], "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
