"""
Null Models Module - Statistical Null Hypothesis Testing

Null models:
- Label Permutation Null: shuffled labels test
- Feature Shuffle Null: random projection test
- Cross-User Null: train on other users test
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


@dataclass
class NullTestResult:
    """Result of null hypothesis test."""
    test_name: str
    null_rejected: bool
    p_value: float
    effect_size: float
    original_metric: float
    null_metric_mean: float
    null_metric_std: float
    ci_lower: float
    ci_upper: float


class BaseNullTest:
    """Base class for null hypothesis tests."""

    def __init__(
        self,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        metric: str = "roc_auc"
    ):
        """
        Args:
            n_permutations: Number of permutations for p-value
            alpha: Significance level
            metric: Evaluation metric
        """
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.metric = metric

    def compute_p_value(
        self,
        original_scores: np.ndarray,
        null_scores: np.ndarray
    ) -> float:
        """Compute empirical p-value."""
        n_greater = np.sum(null_scores >= original_scores)
        p_value = (n_greater + 1) / (len(null_scores) + 1)
        return min(p_value, 1.0)

    def compute_ci(
        self,
        scores: np.ndarray,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute confidence interval."""
        alpha = 1 - confidence
        lower = np.percentile(scores, alpha / 2 * 100)
        upper = np.percentile(scores, (1 - alpha / 2) * 100)
        return lower, upper


class LabelPermutationNull(BaseNullTest):
    """Test if labels carry predictive signal (vs random)."""

    def test(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        model: Optional[object] = None
    ) -> NullTestResult:
        """Test against label-permuted null.

        Args:
            embeddings: (n_samples, n_features)
            labels: (n_samples,)
            model: Classifier to use (default: LogisticRegression)

        Returns:
            NullTestResult with test statistics
        """
        if model is None:
            model = LogisticRegression(max_iter=1000, solver='lbfgs')

        # Original performance
        original_scores = cross_val_score(
            model, embeddings, labels, cv=5, scoring=self.metric
        )
        original_metric = np.mean(original_scores)

        # Null distribution
        null_metrics = []
        n_samples = len(labels)

        rng = np.random.default_rng()

        for _ in range(self.n_permutations):
            shuffled_labels = rng.permutation(labels)
            try:
                null_scores = cross_val_score(
                    model, embeddings, shuffled_labels, cv=5, scoring=self.metric
                )
                null_metrics.append(np.mean(null_scores))
            except:
                null_metrics.append(0.5)

        null_metrics = np.array(null_metrics)
        null_mean = np.mean(null_metrics)
        null_std = np.std(null_metrics)

        # Compute p-value
        p_value = self.compute_p_value(original_metric, null_metrics)
        ci_lower, ci_upper = self.compute_ci(null_metrics)

        return NullTestResult(
            test_name="label_permutation",
            null_rejected=p_value < self.alpha,
            p_value=p_value,
            effect_size=original_metric - null_mean,
            original_metric=original_metric,
            null_metric_mean=null_mean,
            null_metric_std=null_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )


class FeatureShuffleNull(BaseNullTest):
    """Test if embedding structure carries signal (vs random projections)."""

    def __init__(self, n_permutations: int = 1000, alpha: float = 0.05):
        super().__init__(n_permutations, alpha)

    def test(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_random_projections: int = 5
    ) -> NullTestResult:
        """Test against feature-shuffled null.

        Args:
            embeddings: (n_samples, n_features)
            labels: (n_samples,)
            n_random_projections: Number of random projection dimensions to add

        Returns:
            NullTestResult
        """
        from sklearn.random_projection import GaussianRandomProjection

        model = LogisticRegression(max_iter=1000, solver='lbfgs')

        # Original performance
        original_scores = cross_val_score(
            model, embeddings, labels, cv=5, scoring=self.metric
        )
        original_metric = np.mean(original_scores)

        # Null: random projections of embeddings
        null_metrics = []

        rng = np.random.default_rng()

        for _ in range(self.n_permutations):
            try:
                # Random projection
                rp = GaussianRandomProjection(
                    n_components=min(embeddings.shape[1], 64),
                    random_state=rng.integers(0, 2**31)
                )
                random_emb = rp.fit_transform(embeddings)

                null_scores = cross_val_score(
                    model, random_emb, labels, cv=5, scoring=self.metric
                )
                null_metrics.append(np.mean(null_scores))
            except:
                null_metrics.append(0.5)

        null_metrics = np.array(null_metrics)
        null_mean = np.mean(null_metrics)
        null_std = np.std(null_metrics)

        p_value = self.compute_p_value(original_metric, null_metrics)
        ci_lower, ci_upper = self.compute_ci(null_metrics)

        return NullTestResult(
            test_name="feature_shuffle",
            null_rejected=p_value < self.alpha,
            p_value=p_value,
            effect_size=original_metric - null_mean,
            original_metric=original_metric,
            null_metric_mean=null_mean,
            null_metric_std=null_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )


class CrossUserNull(BaseNullTest):
    """Test if signal is user-specific vs general."""

    def __init__(self, n_permutations: int = 1000, alpha: float = 0.05):
        super().__init__(n_permutations, alpha, metric="roc_auc")

    def test(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        user_ids: np.ndarray,
        held_out_user_id: Optional[str] = None
    ) -> NullTestResult:
        """Test cross-user generalization.

        Args:
            embeddings: (n_samples, n_features)
            labels: (n_samples,)
            user_ids: (n_samples,) user identifiers
            held_out_user_id: Specific user to hold out

        Returns:
            NullTestResult
        """
        model = LogisticRegression(max_iter=1000, solver='lbfgs')

        unique_users = np.unique(user_ids)

        # Original: within-user prediction
        original_scores = []

        if held_out_user_id is not None:
            # Hold out specific user
            train_mask = user_ids != held_out_user_id
            test_mask = user_ids == held_out_user_id

            if np.sum(train_mask) < 10 or np.sum(test_mask) < 5:
                return NullTestResult(
                    test_name="cross_user",
                    null_rejected=False,
                    p_value=1.0,
                    effect_size=0.0,
                    original_metric=0.5,
                    null_metric_mean=0.5,
                    null_metric_std=0.0,
                    ci_lower=0.5,
                    ci_upper=0.5
                )

            X_train, y_train = embeddings[train_mask], labels[train_mask]
            X_test, y_test = embeddings[test_mask], labels[test_mask]

            model.fit(X_train, y_train)
            try:
                preds = model.predict_proba(X_test)[:, 1]
                original_metric = roc_auc_score(y_test, preds)
            except:
                original_metric = 0.5

            original_scores.append(original_metric)
        else:
            # Leave-one-user-out
            for user in unique_users:
                train_mask = user_ids != user
                test_mask = user_ids == user

                if np.sum(train_mask) < 10 or np.sum(test_mask) < 5:
                    continue

                X_train, y_train = embeddings[train_mask], labels[train_mask]
                X_test, y_test = embeddings[test_mask], labels[test_mask]

                model.fit(X_train, y_train)
                try:
                    preds = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, preds)
                    original_scores.append(score)
                except:
                    pass

        original_metric = np.mean(original_scores) if original_scores else 0.5

        # Null: shuffled user assignments
        null_metrics = []
        rng = np.random.default_rng()

        for _ in range(min(self.n_permutations, len(unique_users))):
            shuffled_user_ids = rng.permutation(user_ids)

            # Use first user as held-out
            held_out = unique_users[0]
            train_mask = shuffled_user_ids != held_out
            test_mask = shuffled_user_ids == held_out

            if np.sum(train_mask) < 10 or np.sum(test_mask) < 5:
                continue

            X_train, y_train = embeddings[train_mask], labels[train_mask]
            X_test, y_test = embeddings[test_mask], labels[test_mask]

            model.fit(X_train, y_train)
            try:
                preds = model.predict_proba(X_test)[:, 1]
                null_metrics.append(roc_auc_score(y_test, preds))
            except:
                null_metrics.append(0.5)

        if not null_metrics:
            null_metrics = np.array([0.5])

        null_mean = np.mean(null_metrics)
        null_std = np.std(null_metrics)

        p_value = self.compute_p_value(original_metric, null_metrics)
        ci_lower, ci_upper = self.compute_ci(null_metrics)

        return NullTestResult(
            test_name="cross_user",
            null_rejected=p_value < self.alpha,
            p_value=p_value,
            effect_size=original_metric - null_mean,
            original_metric=original_metric,
            null_metric_mean=null_mean,
            null_metric_std=null_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )


class NullModelSuite:
    """Run all null hypothesis tests."""

    def __init__(
        self,
        n_permutations: int = 1000,
        alpha: float = 0.05
    ):
        self.label_permutation = LabelPermutationNull(n_permutations, alpha)
        self.feature_shuffle = FeatureShuffleNull(n_permutations, alpha)
        self.cross_user = CrossUserNull(n_permutations, alpha)
        self.alpha = alpha

    def run_all(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        user_ids: Optional[np.ndarray] = None
    ) -> dict[str, NullTestResult]:
        """Run all null tests.

        Args:
            embeddings: Feature matrix
            labels: Binary labels
            user_ids: Optional user identifiers for cross-user test

        Returns:
            Dict of test_name -> NullTestResult
        """
        results = {}

        # Label permutation
        results["label_permutation"] = self.label_permutation.test(embeddings, labels)

        # Feature shuffle
        results["feature_shuffle"] = self.feature_shuffle.test(embeddings, labels)

        # Cross-user (if user_ids provided)
        if user_ids is not None:
            results["cross_user"] = self.cross_user.test(embeddings, labels, user_ids)

        # Summary
        results["summary"] = {
            "n_tests": len([r for k, r in results.items() if isinstance(r, NullTestResult)]),
            "n_rejected": sum(1 for r in results.values() if isinstance(r, NullTestResult) and r.null_rejected),
            "all_rejected": all(
                r.null_rejected for k, r in results.items()
                if isinstance(r, NullTestResult)
            ),
            "alpha": self.alpha
        }

        return results


def interpret_null_results(results: dict) -> str:
    """Generate interpretation of null test results."""
    summary = results.get("summary", {})

    interpretation = []
    interpretation.append(f"Null Hypothesis Tests ({summary['n_tests']} tests, alpha={summary['alpha']})")
    interpretation.append("=" * 60)

    for name, result in results.items():
        if name == "summary" or not isinstance(result, NullTestResult):
            continue

        status = "REJECTED" if result.null_rejected else "NOT REJECTED"
        interpretation.append(f"\n{name.replace('_', ' ').title()}:")
        interpretation.append(f"  Original metric: {result.original_metric:.4f}")
        interpretation.append(f"  Null mean +/- std: {result.null_metric_mean:.4f} +/- {result.null_metric_std:.4f}")
        interpretation.append(f"  Effect size: {result.effect_size:.4f}")
        interpretation.append(f"  p-value: {result.p_value:.4f}")
        interpretation.append(f"  {summary['alpha']*100}% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        interpretation.append(f"  Result: {status}")

    interpretation.append("\n" + "=" * 60)
    if summary["all_rejected"]:
        interpretation.append("CONCLUSION: All nulls rejected -> Strong signal exists")
    else:
        interpretation.append("CONCLUSION: Some nulls not rejected -> Signal may be weak")

    return "\n".join(interpretation)
