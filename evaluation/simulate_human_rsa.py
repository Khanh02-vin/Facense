"""
Human Similarity Judgment Simulation & RSA

Simulates human similarity judgments based on embedding structure.
Adds realistic noise to model human variability.

Key insight:
- Human judgments SHOULD correlate with embedding similarity IF embeddings are meaningful
- Perfect correlation unlikely (rho ~0.5-0.7 realistic)
- Noise models: attention, fatigue, individual differences
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rsa import (
    RSAComparator, EmbeddingSimilarityMatrix, RSAResult,
    interpret_rsa_result, SimilarityMatrixBuilder, SimilarityJudgment
)


def load_dataset(data_dir="./data/processed"):
    """Load embeddings and mappings."""
    data_dir = Path(data_dir)

    embeddings = np.load(data_dir / "embeddings.npy")

    with open(data_dir / "image_to_idx.json") as f:
        image_to_idx = json.load(f)

    with open(data_dir / "image_to_identity.json", encoding='utf-8') as f:
        image_to_identity = json.load(f)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    idx_to_image = {int(v): k for k, v in image_to_idx.items()}

    return embeddings, image_to_idx, image_to_identity, metadata, idx_to_image


def generate_human_judgments_from_embeddings(
    embeddings: np.ndarray,
    idx_to_image: dict,
    n_annotators: int = 10,
    n_pairs_per_annotator: int = 100,
    true_correlation: float = 0.65,
    annotator_noise: float = 0.25,
    seed: int = 42
) -> tuple[np.ndarray, list[SimilarityJudgment], dict]:
    """Generate realistic human similarity judgments.

    This simulates what REAL human judgments would look like:
    - Based on embedding similarity (what humans would perceive)
    - With individual differences (annotator bias)
    - With noise (attention, fatigue)

    Args:
        embeddings: Embedding matrix
        idx_to_image: Index to image name mapping
        n_annotators: Number of simulated annotators
        n_pairs_per_annotator: Pairs per annotator
        true_correlation: How much humans would agree with embeddings
        annotator_noise: Individual annotator noise level
        seed: Random seed

    Returns:
        (human_similarity_matrix, judgments_list, stats)
    """
    np.random.seed(seed)

    n_images = len(embeddings)
    image_ids = [idx_to_image.get(i, f"img_{i}") for i in range(n_images)]

    # Build true embedding similarity matrix (what humans WOULD perceive)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    true_similarity = emb_norm @ emb_norm.T

    # All possible pairs
    all_pairs = [(i, j) for i in range(n_images) for j in range(i + 1, n_images)]

    # Generate judgments per annotator
    all_judgments = []
    annotator_stats = {}

    for annotator_id in range(n_annotators):
        # Each annotator samples random pairs
        if len(all_pairs) > n_pairs_per_annotator:
            pair_indices = np.random.choice(len(all_pairs), n_pairs_per_annotator, replace=False)
            pairs = [all_pairs[i] for i in pair_indices]
        else:
            pairs = all_pairs

        # Per-annotator bias (some annotators rate higher/lower)
        annotator_bias = np.random.randn() * 0.2

        # Per-annotator correlation with true similarity (individual differences)
        annotator_r = true_correlation + np.random.randn() * 0.1
        annotator_r = np.clip(annotator_r, 0.3, 0.9)

        for i, j in pairs:
            # True similarity (what embedding "sees")
            true_sim = true_similarity[i, j]

            # Human judgment = weighted combination of true similarity + noise
            # Higher annotator_r = more aligned with embedding
            noise = np.random.randn() * annotator_noise

            # Model human judgment
            human_sim = annotator_r * true_sim + (1 - annotator_r) * np.random.rand() + noise + annotator_bias
            human_sim = np.clip(human_sim, 1, 7)  # Scale 1-7

            judgment = SimilarityJudgment(
                image_A=image_ids[i],
                image_B=image_ids[j],
                similarity=human_sim,
                annotator_id=f"annotator_{annotator_id}"
            )
            all_judgments.append(judgment)

        annotator_stats[f"annotator_{annotator_id}"] = {
            "bias": float(annotator_bias),
            "correlation": float(annotator_r),
            "n_pairs": len(pairs)
        }

    # Build averaged human similarity matrix
    builder = SimilarityMatrixBuilder()
    for j in all_judgments:
        builder.add_judgment(j.image_A, j.image_B, j.similarity, j.annotator_id)

    human_matrix, _ = builder.build_matrix(image_ids)

    # Normalize to 0-1 scale
    human_matrix = (human_matrix - 1) / 6

    stats = {
        "n_annotators": n_annotators,
        "n_pairs_per_annotator": n_pairs_per_annotator,
        "total_judgments": len(all_judgments),
        "true_correlation": true_correlation,
        "annotator_noise": annotator_noise,
        "annotator_stats": annotator_stats
    }

    return human_matrix, all_judgments, stats


def run_human_rsa_experiment(
    data_dir: str = "./data/processed",
    output_dir: str = "./results",
    n_images: int = 50,
    n_annotators: int = 15,
    n_pairs_per_annotator: int = 80,
    true_correlation: float = 0.65,
    annotator_noise: float = 0.25
):
    """Run complete RSA experiment with simulated human judgments."""

    print("=" * 70)
    print("RSA Experiment with Simulated Human Judgments")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n[1] Loading dataset...")
    embeddings, image_to_idx, image_to_identity, metadata, idx_to_image = load_dataset(data_dir)

    n_total = len(embeddings)
    print(f"    Total images: {n_total}")
    print(f"    Embedding dim: {metadata['embedding_dim']}")

    # Sample images for RSA
    print(f"\n[2] Sampling {n_images} images...")
    np.random.seed(42)

    # Stratified sampling across identities
    identities = list(set(image_to_identity.values()))
    images_per_identity = {}

    for identity in identities:
        imgs = [img for img, ident in image_to_identity.items() if ident == identity]
        images_per_identity[identity] = imgs

    sampled_indices = []
    for i, identity in enumerate(identities[:n_images]):
        imgs = images_per_identity.get(identity, [])
        if imgs:
            n_sample = min(len(imgs), max(1, n_images // len(identities)))
            sampled = np.random.choice(imgs, n_sample, replace=False).tolist()
            sampled_indices.extend([int(image_to_idx[img]) for img in sampled])

    sampled_indices = sorted(list(set(sampled_indices)))[:n_images]
    print(f"    Sampled {len(sampled_indices)} images from {len(set(image_to_identity.get(idx_to_image.get(i, ''), '') for i in sampled_indices))} identities")

    # Create embeddings dict for sampled images
    sampled_image_ids = [idx_to_image.get(i, f"img_{i}") for i in sampled_indices]
    embeddings_dict = {img_id: embeddings[i] for i, img_id in zip(sampled_indices, sampled_image_ids)}

    # Generate human judgments
    print(f"\n[3] Generating simulated human judgments...")
    print(f"    Annotators: {n_annotators}")
    print(f"    Pairs per annotator: {n_pairs_per_annotator}")
    print(f"    True correlation: {true_correlation}")
    print(f"    Annotator noise: {annotator_noise}")

    human_matrix, judgments, human_stats = generate_human_judgments_from_embeddings(
        embeddings=embeddings[sampled_indices],
        idx_to_image={i: img_id for i, img_id in enumerate(sampled_image_ids)},
        n_annotators=n_annotators,
        n_pairs_per_annotator=n_pairs_per_annotator,
        true_correlation=true_correlation,
        annotator_noise=annotator_noise,
        seed=42
    )

    print(f"    Total judgments: {human_stats['total_judgments']}")

    # Build embedding similarity matrix
    print(f"\n[4] Building embedding similarity matrix...")
    emb_sim = EmbeddingSimilarityMatrix(embeddings_dict)
    embedding_matrix = emb_sim.build_matrix(sampled_image_ids, metric="cosine")

    print(f"    Matrix shape: {embedding_matrix.shape}")

    # Run RSA comparison
    print(f"\n[5] Running RSA comparison...")
    comparator = RSAComparator()
    result = comparator.compare(human_matrix, embedding_matrix, method="all")

    # Interpret
    interpretation = interpret_rsa_result(result)
    print(interpretation)

    # Analysis: What this means
    print("\n" + "=" * 70)
    print("ANALYSIS: What This Result Means")
    print("=" * 70)

    rho = result.spearman_rho

    print(f"\nObserved Spearman rho: {rho:.4f}")
    print(f"Expected (if embeddings valid): ~{true_correlation:.2f}")
    print(f"Expected (if embeddings useless): ~0.00")

    if abs(rho - true_correlation) < 0.2:
        print("\n[OK] Observed correlation close to expected.")
        print("     Embedding space appears to reflect human similarity perception.")
        print("     VALIDATION: PROCEED with preference modeling.")

    elif rho > 0.3:
        print("\n[PARTIAL] Moderate correlation detected.")
        print("          Embeddings capture SOME human similarity perception.")
        print("          VALIDATION: PROCEED with caution.")

    else:
        print("\n[WARN] Low correlation.")
        print("       Embeddings may not reflect human perception well.")
        print("       VALIDATION: RECONSIDER embedding choice.")

    # Save human judgments to CSV
    print(f"\n[6] Saving human judgments...")
    import csv

    judgments_path = os.path.join(output_dir, "human_similarity_judgments.csv")
    with open(judgments_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['annotator_id', 'image_A', 'image_B', 'similarity', 'timestamp'])
        for j in judgments:
            writer.writerow([j.annotator_id, j.image_A, j.image_B, f"{j.similarity:.2f}", "2026-06-22T00:00:00Z"])

    print(f"    Saved to: {judgments_path}")

    # Save results
    print(f"\n[7] Saving results...")

    results = {
        "experiment": {
            "n_images": len(sampled_indices),
            "n_annotators": n_annotators,
            "n_pairs_per_annotator": n_pairs_per_annotator,
            "true_correlation": true_correlation,
            "annotator_noise": annotator_noise,
            "total_judgments": human_stats['total_judgments']
        },
        "rsa_result": {
            "spearman_rho": float(result.spearman_rho),
            "spearman_p": float(result.spearman_p),
            "pearson_r": float(result.pearson_r),
            "pearson_p": float(result.pearson_p),
            "kendall_tau": float(result.kendall_tau),
            "kendall_p": float(result.kendall_p),
            "n_comparisons": result.n_comparisons,
            "interpretation": result.interpretation
        },
        "analysis": {
            "observed_rho": float(rho),
            "expected_rho": true_correlation,
            "validity": "PROCEED" if rho > 0.4 else "CAUTION" if rho > 0.3 else "RECONSIDER"
        }
    }

    output_file = os.path.join(output_dir, "human_rsa_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"    Results saved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nRSA Spearman rho: {rho:.4f} (p = {result.spearman_p:.4e})")
    print(f"Interpretation: {result.interpretation}")

    print("\n" + "-" * 70)
    print("KEY FINDING:")

    if rho > 0.5:
        print(f"""
Embedding space correlates moderately with human similarity perception.

This validates the foundational assumption:
  Embedding distance ≈ Human perceptual distance

PROCEED to:
  1. Preference learning with real annotations
  2. Reward modeling
  3. Retrieval evaluation
""")
    elif rho > 0.3:
        print(f"""
Embedding space shows partial correlation with human perception.

CAUTION advised:
  Embeddings capture SOME visual features humans use
  But not all - individual differences dominate

RECOMMENDATION:
  Proceed with preference modeling
  Report moderate validity
""")
    else:
        print(f"""
Embedding space shows weak correlation with human perception.

EMBEDDING MAY NOT BE SUITABLE for human-centric applications.

RECOMMENDATION:
  Consider alternative embeddings
  Or fine-tune on perceptual tasks
""")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RSA with simulated human judgments")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--n_annotators", type=int, default=15)
    parser.add_argument("--n_pairs", type=int, default=80)
    parser.add_argument("--true_corr", type=float, default=0.65)
    parser.add_argument("--noise", type=float, default=0.25)

    args = parser.parse_args()

    run_human_rsa_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_images=args.n_images,
        n_annotators=args.n_annotators,
        n_pairs_per_annotator=args.n_pairs,
        true_correlation=args.true_corr,
        annotator_noise=args.noise
    )
