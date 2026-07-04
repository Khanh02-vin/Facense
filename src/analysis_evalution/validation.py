"""
Validation Module - Representation Validation Tests

Tests:
- Stability: augmentation consistency
- Invariance: feature importance analysis
- Cross-model: agreement across different encoders
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validation test."""
    test_name: str
    passed: bool
    metric: float
    p_value: Optional[float] = None
    details: Optional[dict] = None


class StabilityValidator:
    """Test embedding stability under perturbations."""

    def __init__(self, threshold: float = 0.85):
        """
        Args:
            threshold: Minimum cosine similarity for stability
        """
        self.threshold = threshold

    def test(
        self,
        original_embedding: np.ndarray,
        augmented_embeddings: list[np.ndarray]
    ) -> ValidationResult:
        """Test if augmented embeddings are close to original.

        Args:
            original_embedding: Embedding of original image
            augmented_embeddings: List of embeddings from augmented versions

        Returns:
            ValidationResult with stability metrics
        """
        similarities = [
            np.dot(original_embedding, aug) / (
                np.linalg.norm(original_embedding) * np.linalg.norm(aug)
            )
            for aug in augmented_embeddings
        ]

        mean_sim = np.mean(similarities)
        passed = mean_sim >= self.threshold

        return ValidationResult(
            test_name="stability",
            passed=passed,
            metric=mean_sim,
            details={
                "threshold": self.threshold,
                "std": np.std(similarities),
                "min": np.min(similarities),
                "max": np.max(similarities),
                "n_augmentations": len(augmented_embeddings)
            }
        )


class FeatureImportanceValidator:
    """Analyze which features contribute to preference signal."""

    def __init__(self):
        pass

    def compute_ablation_scores(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_ablations: int = 50
    ) -> dict:
        """Compute feature ablation importance scores.

        Args:
            embeddings: (n_samples, n_features) embedding matrix
            labels: (n_samples,) binary labels
            n_ablations: Number of ablation iterations

        Returns:
            dict with ablation importance scores
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        n_features = embeddings.shape[1]
        original_scores = []

        # Original performance
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        original_auc = np.mean(cross_val_score(lr, embeddings, labels, cv=5, scoring='roc_auc'))
        original_scores.append(original_auc)

        # Ablation: shuffle each dimension
        ablation_scores = []
        for dim in range(n_features):
            shuffled_embeddings = embeddings.copy()
            dim_scores = []

            for _ in range(n_ablations // n_features + 1):
                perm = np.random.permutation(len(shuffled_embeddings))
                shuffled_embeddings[:, dim] = shuffled_embeddings[perm, dim]

                lr = LogisticRegression(max_iter=1000, solver='lbfgs')
                try:
                    score = np.mean(cross_val_score(
                        lr, shuffled_embeddings, labels, cv=3, scoring='roc_auc'
                    ))
                    dim_scores.append(score)
                except:
                    dim_scores.append(0.5)

            ablation_scores.append({
                "dimension": dim,
                "mean_score": np.mean(dim_scores),
                "drop": original_auc - np.mean(dim_scores)
            })

        # Sort by importance (largest drop = most important)
        ablation_scores.sort(key=lambda x: x["drop"], reverse=True)

        return {
            "original_auc": original_auc,
            "ablation_scores": ablation_scores,
            "top_features": ablation_scores[:10] if len(ablation_scores) >= 10 else ablation_scores
        }


class CrossModelValidator:
    """Validate signal across different embedding models."""

    def __init__(self):
        pass

    def compute_agreement(
        self,
        predictions_dict: dict[str, np.ndarray]
    ) -> dict:
        """Compute agreement across model predictions.

        Args:
            predictions_dict: Dict mapping model_name -> predicted probabilities

        Returns:
            dict with agreement metrics
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)

        if n_models < 2:
            return {"error": "Need at least 2 models"}

        # Pairwise correlations
        correlations = {}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_i, name_j = model_names[i], model_names[j]
                preds_i = predictions_dict[name_i]
                preds_j = predictions_dict[name_j]

                corr = np.corrcoef(preds_i, preds_j)[0, 1]
                correlations[f"{name_i}_vs_{name_j}"] = corr

        mean_correlation = np.mean(list(correlations.values()))

        return {
            "pairwise_correlations": correlations,
            "mean_correlation": mean_correlation,
            "min_correlation": np.min(list(correlations.values())),
            "n_models": n_models,
            "agreement": "high" if mean_correlation > 0.7 else "moderate" if mean_correlation > 0.4 else "low"
        }


class RepresentationValidator:
    """Combined representation validation."""

    def __init__(
        self,
        stability_threshold: float = 0.85,
        invariance_threshold: float = 0.05,
        cross_model_threshold: float = 0.4
    ):
        self.stability = StabilityValidator(threshold=stability_threshold)
        self.feature_importance = FeatureImportanceValidator()
        self.cross_model = CrossModelValidator()
        self.thresholds = {
            "stability": stability_threshold,
            "invariance": invariance_threshold,
            "cross_model": cross_model_threshold
        }

    def validate(
        self,
        embeddings: dict,
        labels: np.ndarray,
        augmented_embeddings: Optional[dict] = None
    ) -> dict:
        """Run full representation validation.

        Args:
            embeddings: Dict of model_name -> embedding matrix
            labels: Ground truth labels
            augmented_embeddings: Optional dict for stability testing

        Returns:
            dict with all validation results
        """
        results = {}

        # Stability validation
        if augmented_embeddings:
            for model_name, aug_embs in augmented_embeddings.items():
                if model_name in embeddings:
                    orig = embeddings[model_name]
                    # Use mean of augmented embeddings for comparison
                    aug_mean = np.mean(aug_embs, axis=0)
                    result = self.stability.test(orig, [aug_mean])
                    results[f"stability_{model_name}"] = result

        # Feature importance
        for model_name, emb in embeddings.items():
            importance = self.feature_importance.compute_ablation_scores(emb, labels)
            results[f"importance_{model_name}"] = importance

        # Cross-model agreement
        if len(embeddings) >= 2:
            predictions = {}
            for model_name, emb in embeddings.items():
                # Simple logistic regression predictions
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_predict
                lr = LogisticRegression(max_iter=1000)
                probs = cross_val_predict(
                    lr, emb, labels, cv=5, method='predict_proba'
                )[:, 1]
                predictions[model_name] = probs

            results["cross_model_agreement"] = self.cross_model.compute_agreement(predictions)

        return results
