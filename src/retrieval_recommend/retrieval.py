"""
Retrieval Module - Counterfactual Retrieval Evaluation

Metrics:
- Hit@K, MRR, NDCG
- Identity-controlled retrieval
- Counterfactual evaluation
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    hit_rate: float
    mrr: float
    ndcg: float
    precision_at_k: float
    recall_at_k: float
    coverage: float


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    query_id: str
    candidate_pool: list[str]
    retrieved_ids: list[str]
    user_ratings: Optional[list[int]] = None
    metrics: Optional[RetrievalMetrics] = None


class RetrievalEngine:
    """Similarity-based retrieval engine."""

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        metric: str = "cosine"
    ):
        """
        Args:
            embeddings: Dict mapping image_id -> embedding
            metric: Distance metric ('cosine' or 'euclidean')
        """
        self.embeddings = embeddings
        self.metric = metric
        self._build_index()

    def _build_index(self):
        """Build nearest neighbor index."""
        ids = list(self.embeddings.keys())
        matrix = np.array([self.embeddings[i] for i in ids])

        if self.metric == "cosine":
            # Normalize for cosine similarity = dot product
            matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)

        self.ids = ids
        self.matrix = matrix

        self.nn = NearestNeighbors(
            n_neighbors=min(20, len(ids)),
            metric="euclidean" if self.metric == "euclidean" else "cosine"
        )
        self.nn.fit(matrix)

    def retrieve(
        self,
        query_id: str,
        k: int = 10,
        exclude_ids: Optional[list[str]] = None
    ) -> list[str]:
        """Retrieve top-k similar images.

        Args:
            query_id: Query image ID
            k: Number of results
            exclude_ids: IDs to exclude from results

        Returns:
            List of retrieved image IDs
        """
        if query_id not in self.embeddings:
            return []

        query_emb = self.embeddings[query_id].reshape(1, -1)
        if self.metric == "cosine":
            query_emb = query_emb / np.linalg.norm(query_emb)

        n_neighbors = min(k + 10, len(self.ids))
        distances, indices = self.nn.kneighbors(query_emb, n_neighbors=n_neighbors)

        retrieved = []
        for idx in indices[0]:
            img_id = self.ids[idx]
            if img_id == query_id:
                continue
            if exclude_ids and img_id in exclude_ids:
                continue
            retrieved.append(img_id)
            if len(retrieved) >= k:
                break

        return retrieved

    def retrieve_with_scores(
        self,
        query_id: str,
        k: int = 10,
        exclude_ids: Optional[list[str]] = None
    ) -> list[tuple[str, float]]:
        """Retrieve with similarity scores."""
        if query_id not in self.embeddings:
            return []

        query_emb = self.embeddings[query_id].reshape(1, -1)
        if self.metric == "cosine":
            query_emb = query_emb / np.linalg.norm(query_emb)

        n_neighbors = min(k + 10, len(self.ids))
        distances, indices = self.nn.kneighbors(query_emb, n_neighbors=n_neighbors)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            img_id = self.ids[idx]
            if img_id == query_id:
                continue
            if exclude_ids and img_id in exclude_ids:
                continue

            # Convert distance to similarity
            if self.metric == "cosine":
                sim = 1 - dist
            else:
                sim = 1 / (1 + dist)

            results.append((img_id, float(sim)))
            if len(results) >= k:
                break

        return results


class IdentityControlledRetrieval:
    """Retrieval with identity control."""

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        identity_groups: dict[str, list[str]]
    ):
        """
        Args:
            embeddings: Dict mapping image_id -> embedding
            identity_groups: Dict mapping identity_id -> list of image_ids
        """
        self.base_engine = RetrievalEngine(embeddings)
        self.identity_groups = identity_groups

        # Build reverse mapping
        self.image_to_identity = {}
        for identity_id, image_ids in identity_groups.items():
            for img_id in image_ids:
                self.image_to_identity[img_id] = identity_id

    def retrieve(
        self,
        query_id: str,
        k: int = 10,
        control_level: Literal["none", "partial", "strict"] = "partial"
    ) -> list[str]:
        """Retrieve with identity control.

        Args:
            query_id: Query image ID
            k: Number of results
            control_level:
                - 'none': Standard retrieval
                - 'partial': Max 1-2 images per identity
                - 'strict': No same-identity images
        """
        query_identity = self.image_to_identity.get(query_id, None)

        if control_level == "none" or query_identity is None:
            return self.base_engine.retrieve(query_id, k)

        # Get all same-identity image IDs
        same_identity = set(self.identity_groups.get(query_identity, []))
        same_identity.discard(query_id)

        # Retrieve more candidates for filtering
        candidates = self.base_engine.retrieve(query_id, k * 3, exclude_ids=[query_id])

        # Apply identity control
        if control_level == "strict":
            candidates = [c for c in candidates if c not in same_identity]
        elif control_level == "partial":
            # Allow max 2 per identity
            identity_counts = {}
            filtered = []
            for c in candidates:
                c_identity = self.image_to_identity.get(c, None)
                if c_identity == query_identity:
                    count = identity_counts.get(c_identity, 0)
                    if count < 2:
                        filtered.append(c)
                        identity_counts[c_identity] = count + 1
                else:
                    filtered.append(c)
            candidates = filtered

        return candidates[:k]


class RetrievalEvaluator:
    """Evaluate retrieval against user ratings."""

    def __init__(
        self,
        k_values: list[int] = None
    ):
        """
        Args:
            k_values: K values for Hit@K, Precision@K, etc.
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        query_id: str,
        retrieved_ids: list[str],
        user_ratings: list[int],
        relevance_threshold: int = 4
    ) -> RetrievalMetrics:
        """Evaluate single retrieval result.

        Args:
            query_id: Query image ID
            retrieved_ids: Retrieved image IDs
            user_ratings: User ratings for each retrieved image (1-5)
            relevance_threshold: Rating >= threshold is considered relevant

        Returns:
            RetrievalMetrics
        """
        n_retrieved = len(retrieved_ids)

        # Relevance binary
        relevance = [1 if r >= relevance_threshold else 0 for r in user_ratings]

        # Hit@K
        hits = {k: 0 for k in self.k_values}
        for k in self.k_values:
            if k <= n_retrieved:
                hits[k] = sum(relevance[:k]) / k

        # MRR
        mrr = 0.0
        for i, rel in enumerate(relevance):
            if rel == 1:
                mrr = 1.0 / (i + 1)
                break

        # NDCG
        dcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevance)
        )

        # Ideal DCG
        ideal_ratings = sorted(user_ratings, reverse=True)
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(ideal_ratings[:n_retrieved])
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Precision@K
        precision = {}
        for k in self.k_values:
            if k <= n_retrieved:
                precision[k] = sum(relevance[:k]) / k
            else:
                precision[k] = sum(relevance) / k if k > 0 else 0.0

        # Recall@K
        total_relevant = sum(relevance)
        recall = {}
        for k in self.k_values:
            if k <= n_retrieved:
                recall[k] = sum(relevance[:k]) / total_relevant if total_relevant > 0 else 0.0
            else:
                recall[k] = sum(relevance) / total_relevant if total_relevant > 0 else 0.0

        return RetrievalMetrics(
            hit_rate=hits[max(self.k_values)],
            mrr=mrr,
            ndcg=ndcg,
            precision_at_k=precision[max(self.k_values)],
            recall_at_k=recall[max(self.k_values)],
            coverage=n_retrieved / len(user_ratings) if user_ratings else 0.0
        )

    def evaluate_batch(
        self,
        results: list[RetrievalResult]
    ) -> dict:
        """Evaluate batch of retrieval results.

        Returns:
            Aggregated metrics dict
        """
        all_metrics = []
        for result in results:
            if result.metrics is not None:
                all_metrics.append(result.metrics)

        if not all_metrics:
            return {}

        return {
            "mean_hit_rate": np.mean([m.hit_rate for m in all_metrics]),
            "mean_mrr": np.mean([m.mrr for m in all_metrics]),
            "mean_ndcg": np.mean([m.ndcg for m in all_metrics]),
            "std_hit_rate": np.std([m.hit_rate for m in all_metrics]),
            "std_mrr": np.std([m.mrr for m in all_metrics]),
            "std_ndcg": np.std([m.ndcg for m in all_metrics]),
            "n_queries": len(all_metrics)
        }


class CounterfactualRetrieval:
    """Counterfactual retrieval evaluation protocol."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        candidate_pool_size: int = 20,
        selection_size: int = 5
    ):
        """
        Args:
            retrieval_engine: Base retrieval engine
            candidate_pool_size: Size of candidate pool
            selection_size: Number of items model selects from pool
        """
        self.engine = retrieval_engine
        self.candidate_pool_size = candidate_pool_size
        self.selection_size = selection_size

    def evaluate(
        self,
        query_id: str,
        user_preference_order: list[str],
        user_ratings: dict[str, int]
    ) -> dict:
        """Evaluate counterfactual retrieval.

        Args:
            query_id: Query image ID
            user_preference_order: User's true preference order (best first)
            user_ratings: Dict mapping image_id -> user rating

        Returns:
            Evaluation dict
        """
        # Get candidate pool
        candidates = self.engine.retrieve(query_id, self.candidate_pool_size)

        if len(candidates) < self.selection_size:
            return {"error": "Not enough candidates"}

        # Model selection (top by retrieval score)
        model_selected = candidates[:self.selection_size]

        # Ideal selection (top by user preference)
        ideal_selected = [
            img for img in user_preference_order
            if img in candidates
        ][:self.selection_size]

        # Compute metrics
        model_ratings = [user_ratings.get(img, 0) for img in model_selected]
        ideal_ratings = [user_ratings.get(img, 0) for img in ideal_selected]

        # Model MRR
        model_mrr = 0.0
        for i, img in enumerate(model_selected):
            rating = user_ratings.get(img, 0)
            if rating >= 4:
                model_mrr = 1.0 / (i + 1)
                break

        # Ideal MRR
        ideal_mrr = 0.0
        for i, img in enumerate(ideal_selected):
            rating = user_ratings.get(img, 0)
            if rating >= 4:
                ideal_mrr = 1.0 / (i + 1)
                break

        # Coverage overlap
        overlap = len(set(model_selected) & set(ideal_selected))

        return {
            "query_id": query_id,
            "model_selected": model_selected,
            "ideal_selected": ideal_selected,
            "model_mrr": model_mrr,
            "ideal_mrr": ideal_mrr,
            "overlap": overlap,
            "overlap_ratio": overlap / self.selection_size,
            "model_mean_rating": np.mean(model_ratings),
            "ideal_mean_rating": np.mean(ideal_ratings),
            "rating_gap": np.mean(model_ratings) - np.mean(ideal_ratings)
        }


def compute_identity_leakage(
    uncontrolled_metrics: RetrievalMetrics,
    controlled_metrics: RetrievalMetrics
) -> dict:
    """Estimate identity leakage from delta between controlled/uncontrolled retrieval.

    Args:
        uncontrolled_metrics: Metrics without identity control
        controlled_metrics: Metrics with identity control

    Returns:
        Dict with leakage estimates
    """
    return {
        "hit_rate_leakage": (
            uncontrolled_metrics.hit_rate - controlled_metrics.hit_rate
        ),
        "mrr_leakage": (
            uncontrolled_metrics.mrr - controlled_metrics.mrr
        ),
        "ndcg_leakage": (
            uncontrolled_metrics.ndcg - controlled_metrics.ndcg
        ),
        "leakage_interpretation": (
            "High leakage (>0.1) suggests retrieval relies on identity memorization"
            if (uncontrolled_metrics.hit_rate - controlled_metrics.hit_rate) > 0.1
            else "Low leakage suggests retrieval captures genuine preference"
        )
    }
