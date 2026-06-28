"""
RSA Module - Representational Similarity Analysis

Purpose:
    Validate that embedding space reflects human perceptual similarity.
    This is the FOUNDATIONAL test before any preference claims.

RSA Core Idea:
    Compare two similarity matrices:
    1. Human similarity judgments (psychophysics)
    2. Embedding-based similarity

If they correlate (rho > 0.5), embedding space is perceptually meaningful.
If they don't correlate, downstream preference pipeline is built on wrong foundation.

Reference: Kriegeskorte et al. (2008) - Representational Similarity Analysis
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr, kendalltau
from itertools import combinations


@dataclass
class RSAResult:
    """Result of RSA comparison."""
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    kendall_tau: float
    kendall_p: float
    n_comparisons: int
    interpretation: str


@dataclass
class SimilarityJudgment:
    """Single human similarity judgment."""
    image_A: str
    image_B: str
    similarity: float  # 0 = very different, 1 = very similar
    annotator_id: str
    timestamp: float = 0.0


class SimilarityMatrixBuilder:
    """Build similarity matrices from judgments."""

    def __init__(self):
        self.judgments = []

    def add_judgment(
        self,
        image_A: str,
        image_B: str,
        similarity: float,
        annotator_id: str = "anonymous"
    ):
        """Add a similarity judgment."""
        self.judgments.append(SimilarityJudgment(
            image_A=image_A,
            image_B=image_B,
            similarity=similarity,
            annotator_id=annotator_id
        ))

    def build_matrix(
        self,
        image_ids: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Build similarity matrix from judgments.

        Args:
            image_ids: Ordered list of image IDs

        Returns:
            (similarity_matrix, image_ids)
        """
        n = len(image_ids)
        matrix = np.ones((n, n))  # Self-similarity = 1

        # Initialize from judgments
        img_to_idx = {img: i for i, img in enumerate(image_ids)}

        for judgment in self.judgments:
            if judgment.image_A in img_to_idx and judgment.image_B in img_to_idx:
                i = img_to_idx[judgment.image_A]
                j = img_to_idx[judgment.image_B]
                matrix[i, j] = judgment.similarity
                matrix[j, i] = judgment.similarity  # Symmetric

        return matrix, image_ids

    def to_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Convert symmetric matrix to upper-triangle vector.

        Used for correlation computation (excluding diagonal).
        """
        n = matrix.shape[0]
        indices = np.triu_indices(n, k=1)
        return matrix[indices]


class EmbeddingSimilarityMatrix:
    """Build similarity matrix from embeddings."""

    def __init__(self, embeddings: dict[str, np.ndarray]):
        """
        Args:
            embeddings: Dict mapping image_id -> embedding vector
        """
        self.embeddings = embeddings

    def build_matrix(
        self,
        image_ids: list[str],
        metric: Literal["cosine", "euclidean"] = "cosine"
    ) -> np.ndarray:
        """Build similarity matrix from embeddings.

        Args:
            image_ids: Ordered list of image IDs
            metric: 'cosine' (higher = more similar) or 'euclidean' (lower = more similar)

        Returns:
            Similarity matrix
        """
        n = len(image_ids)
        matrix = np.zeros((n, n))

        # Get embedding matrix
        emb_list = []
        for img_id in image_ids:
            if img_id in self.embeddings:
                emb_list.append(self.embeddings[img_id])
            else:
                emb_list.append(np.zeros_like(list(self.embeddings.values())[0]))

        emb_matrix = np.array(emb_list)

        # Normalize for cosine
        if metric == "cosine":
            emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)

        # Compute pairwise similarities
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0  # Self-similarity
                elif metric == "cosine":
                    matrix[i, j] = np.dot(emb_matrix[i], emb_matrix[j])
                else:  # euclidean
                    dist = np.linalg.norm(emb_matrix[i] - emb_matrix[j])
                    matrix[i, j] = 1.0 / (1.0 + dist)  # Convert distance to similarity

        return matrix

    def to_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Convert symmetric matrix to upper-triangle vector."""
        n = matrix.shape[0]
        indices = np.triu_indices(n, k=1)
        return matrix[indices]


class RSAComparator:
    """Compare human and embedding similarity matrices."""

    def __init__(self):
        pass

    def compare(
        self,
        human_matrix: np.ndarray,
        embedding_matrix: np.ndarray,
        method: Literal["spearman", "pearson", "kendall", "all"] = "all"
    ) -> RSAResult:
        """Compare two similarity matrices using RSA.

        Args:
            human_matrix: Human similarity judgments
            embedding_matrix: Embedding-based similarities

        Returns:
            RSAResult with correlation statistics
        """
        # Convert to vectors (upper triangle)
        human_vec = self._matrix_to_vector(human_matrix)
        embed_vec = self._matrix_to_vector(embedding_matrix)

        n_comparisons = len(human_vec)

        results = {}

        if method in ["spearman", "all"]:
            rho, p = spearmanr(human_vec, embed_vec)
            results["spearman_rho"] = rho
            results["spearman_p"] = p

        if method in ["pearson", "all"]:
            r, p = pearsonr(human_vec, embed_vec)
            results["pearson_r"] = r
            results["pearson_p"] = p

        if method in ["kendall", "all"]:
            tau, p = kendalltau(human_vec, embed_vec)
            results["kendall_tau"] = tau
            results["kendall_p"] = p

        # Interpretation
        rho = results.get("spearman_rho", 0)

        if rho < 0:
            interpretation = "NEGATIVE correlation - embedding space INVERTS human perception. STOP pipeline."
        elif rho < 0.3:
            interpretation = "WEAK correlation (< 0.3) - embedding space does NOT reflect human perception. Consider alternative embeddings or fine-tuning."
        elif rho < 0.5:
            interpretation = "MODERATE correlation (0.3-0.5) - partial perceptual alignment. Proceed with caution."
        elif rho < 0.7:
            interpretation = "GOOD correlation (0.5-0.7) - embedding space captures human perception. PROCEED."
        else:
            interpretation = "STRONG correlation (> 0.7) - embedding space well-aligned with human perception. VALIDATED."

        return RSAResult(
            spearman_rho=results.get("spearman_rho", 0),
            spearman_p=results.get("spearman_p", 1),
            pearson_r=results.get("pearson_r", 0),
            pearson_p=results.get("pearson_p", 1),
            kendall_tau=results.get("kendall_tau", 0),
            kendall_p=results.get("kendall_p", 1),
            n_comparisons=n_comparisons,
            interpretation=interpretation
        )

    def _matrix_to_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Convert symmetric matrix to upper-triangle vector."""
        n = matrix.shape[0]
        indices = np.triu_indices(n, k=1)
        return matrix[indices]


class RSACrossAttribute:
    """RSA broken down by attribute subsets."""

    def __init__(self):
        pass

    def compare_by_attribute(
        self,
        human_matrix: np.ndarray,
        embedding_matrix: np.ndarray,
        image_to_attributes: dict[str, dict[str, any]],
        attribute_name: str
    ) -> dict:
        """Compare similarity matrices for images with specific attribute value.

        Args:
            human_matrix: Full human similarity matrix
            embedding_matrix: Full embedding similarity matrix
            image_to_attributes: Dict mapping image_id -> {attribute: value}
            attribute_name: Which attribute to filter on

        Returns:
            Dict with correlation per attribute value
        """
        # Get images with each attribute value
        attr_values = set()
        for attrs in image_to_attributes.values():
            if attribute_name in attrs:
                attr_values.add(attrs[attribute_name])

        results = {}

        for value in attr_values:
            # Get images with this attribute value
            images_with_attr = [
                img_id for img_id, attrs in image_to_attributes.items()
                if attrs.get(attribute_name) == value
            ]

            if len(images_with_attr) < 3:
                continue

            # Build sub-matrices
            # This is simplified - in practice would need index mapping
            results[value] = {
                "n_images": len(images_with_attr),
                "note": "Subset RSA not fully implemented - requires index mapping"
            }

        return results


def generate_synthetic_human_similarity(
    n_images: int = 20,
    noise_level: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """Generate synthetic human similarity judgments.

    This simulates having actual human data for testing RSA.

    Args:
        n_images: Number of images
        noise_level: Standard deviation of noise (0 = perfect correlation)
        seed: Random seed

    Returns:
        Synthetic human similarity matrix
    """
    np.random.seed(seed)

    # True underlying "perceptual" structure
    # Images cluster into groups (simulating similar types)
    n_groups = 4
    group_assignments = np.random.randint(0, n_groups, n_images)

    true_similarity = np.ones((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            if i == j:
                true_similarity[i, j] = 1.0
            elif group_assignments[i] == group_assignments[j]:
                true_similarity[i, j] = 0.9 + np.random.rand() * 0.1
            else:
                true_similarity[i, j] = 0.1 + np.random.rand() * 0.3

    # Add noise
    noisy_similarity = true_similarity + np.random.randn(n_images, n_images) * noise_level
    noisy_similarity = np.clip(noisy_similarity, 0, 1)

    # Symmetrize
    noisy_similarity = (noisy_similarity + noisy_similarity.T) / 2
    np.fill_diagonal(noisy_similarity, 1.0)

    return noisy_similarity


def interpret_rsa_result(result: RSAResult) -> str:
    """Generate interpretation of RSA result."""
    lines = []
    lines.append("=" * 60)
    lines.append("RSA Results - Embedding vs Human Perception")
    lines.append("=" * 60)
    lines.append(f"Spearman rho: {result.spearman_rho:.4f} (p = {result.spearman_p:.4e})")
    lines.append(f"Pearson r: {result.pearson_r:.4f} (p = {result.pearson_p:.4e})")
    lines.append(f"Kendall tau: {result.kendall_tau:.4f} (p = {result.kendall_p:.4e})")
    lines.append(f"Comparisons: {result.n_comparisons}")
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append(result.interpretation)
    lines.append("")
    lines.append("-" * 60)

    # Decision
    if result.spearman_rho < 0.3:
        lines.append("DECISION: STOP pipeline - embeddings don't reflect human perception")
    elif result.spearman_rho < 0.5:
        lines.append("DECISION: CAUTION - partial validation, proceed carefully")
    else:
        lines.append("DECISION: VALIDATED - embedding space reflects human perception")

    lines.append("=" * 60)

    return "\n".join(lines)


def run_rsa_with_synthetic_data(
    n_images: int = 50,
    embedding_dim: int = 256,
    noise_level: float = 0.2
) -> RSAResult:
    """Run complete RSA with synthetic data.

    Demonstrates RSA pipeline with synthetic human + synthetic embedding data.
    """
    print("[RSA] Running with synthetic data...")
    print()

    # Generate synthetic human judgments
    print(f"[RSA] Generating {n_images} synthetic human similarity judgments...")
    human_matrix = generate_synthetic_human_similarity(n_images, noise_level)

    # Generate synthetic embeddings (should correlate with human if structured properly)
    print("[RSA] Generating synthetic embeddings...")
    np.random.seed(42)
    embeddings = {}

    # Create embeddings with structure (some images similar, some different)
    for i in range(n_images):
        # Each image has a "type" embedding + noise
        emb = np.random.randn(embedding_dim) * 0.5
        # Add structure based on image index (simulating similar types)
        if i % 5 < 3:  # 60% of images share some structure
            emb[:50] += np.random.randn(50) * 2

        emb = emb / np.linalg.norm(emb)
        embeddings[f"img_{i}"] = emb

    # Build embedding similarity matrix
    print("[RSA] Building embedding similarity matrix...")
    emb_sim = EmbeddingSimilarityMatrix(embeddings)
    image_ids = [f"img_{i}" for i in range(n_images)]
    embedding_matrix = emb_sim.build_matrix(image_ids, metric="cosine")

    # Run RSA comparison
    print("[RSA] Computing RSA correlation...")
    comparator = RSAComparator()
    result = comparator.compare(human_matrix, embedding_matrix, method="all")

    print(interpret_rsa_result(result))

    return result


# =============================================================================
# RSA Protocol for Real Human Data Collection
# =============================================================================

RSA_PROTOCOL = """
# RSA Protocol - Human Similarity Judgment Collection

## Objective
Collect human similarity judgments to validate embedding space.

## Images to Judge
Select N = 30-50 representative images stratified by:
- Different identities (avoid same person bias)
- Different visual attributes (hair length, style, face shape, etc.)
- Balanced across dataset

## Judgment Task
Show pairs of images and ask:
"How similar do these two people look in terms of overall appearance?"

Scale:
1 - Very different
2 - Somewhat different
3 - Neutral
4 - Somewhat similar
5 - Very similar

## Number of Judgments
For N images, all pairs = N*(N-1)/2
- N=30: 435 judgments (too many)
- Recommend: N=20, random 100-150 pairs per annotator

## Annotator Requirements
- Minimum 10 annotators per image set
- Exclude judges who fail attention checks
- Compute inter-rater reliability (ICC > 0.6)

## Output
CSV with columns:
user_id, image_A, image_B, similarity (1-5), timestamp

## Analysis
1. Build average human similarity matrix
2. Build embedding similarity matrix
3. Compute Spearman rho
4. Interpret:
   - rho < 0.3: STOP
   - rho 0.3-0.5: Caution
   - rho > 0.5: Validated
"""


if __name__ == "__main__":
    # Demo RSA with synthetic data
    result = run_rsa_with_synthetic_data(n_images=50, noise_level=0.2)
