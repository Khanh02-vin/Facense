"""
RSA Evaluation Script

Runs Representational Similarity Analysis on the dataset.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rsa import (
    RSAComparator, EmbeddingSimilarityMatrix, RSAResult,
    interpret_rsa_result, generate_synthetic_human_similarity
)


def load_dataset(data_dir="./data/processed"):
    """Load embeddings and metadata."""
    data_dir = Path(data_dir)

    embeddings = np.load(data_dir / "embeddings.npy")

    with open(data_dir / "image_to_idx.json") as f:
        image_to_idx = json.load(f)

    with open(data_dir / "image_to_identity.json", encoding='utf-8') as f:
        image_to_identity = json.load(f)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Build idx to image mapping
    idx_to_image = {int(v): k for k, v in image_to_idx.items()}

    return embeddings, image_to_idx, image_to_identity, metadata, idx_to_image


def generate_attribute_annotations(image_to_identity, idx_to_image):
    """Generate synthetic attribute annotations from identity groups.

    This creates synthetic attributes for RSA testing.
    In reality, attributes would come from human annotation.
    """
    # Group identities by "type" based on name patterns or random assignment
    attributes = {}

    identities = list(set(image_to_identity.values()))

    # Assign synthetic attributes based on identity patterns
    for img_name, identity in image_to_identity.items():
        attrs = {}

        # Extract some pseudo-attributes from identity names or assign randomly
        # In reality, these would come from human annotation
        identity_idx = identities.index(identity)
        attrs["style_type"] = identity_idx % 5  # 5 style types
        attrs["region"] = identity_idx % 3  # 3 regions
        attrs["era"] = identity_idx % 2  # 2 eras

        attributes[img_name] = attrs

    return attributes, identities


def run_rsa_evaluation(
    data_dir="./data/processed",
    output_dir="./results",
    n_sample_images: int = 50,
    synthetic_human_noise: float = 0.3
):
    """Run RSA evaluation on dataset.

    Args:
        data_dir: Path to processed data
        output_dir: Path to save results
        n_sample_images: Number of images to sample for RSA
        synthetic_human_noise: Noise level for synthetic human judgments
    """
    print("=" * 60)
    print("RSA Evaluation - Embedding vs Human Perception")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n[1] Loading dataset...")
    embeddings, image_to_idx, image_to_identity, metadata, idx_to_image = load_dataset(data_dir)

    n_total = len(embeddings)
    print(f"    Total images: {n_total}")
    print(f"    Embedding dim: {metadata['embedding_dim']}")

    # Sample images for RSA (need O(N^2) comparisons)
    print(f"\n[2] Sampling {n_sample_images} images for RSA...")
    sample_size = min(n_sample_images, n_total)

    # Stratified sampling: ensure variety of identities
    np.random.seed(42)

    identities = list(set(image_to_identity.values()))
    images_per_identity = n_total // len(identities)

    sampled_indices = []
    for identity in identities[:min(sample_size, len(identities))]:
        # Get images from this identity
        imgs = [img for img, ident in image_to_identity.items() if ident == identity]
        if imgs:
            # Sample 1-3 images per identity
            n_sample = min(len(imgs), max(1, sample_size // len(identities)))
            sampled = np.random.choice(imgs, n_sample, replace=False).tolist()
            sampled_indices.extend([int(image_to_idx[img]) for img in sampled])

    sampled_indices = sampled_indices[:sample_size]
    print(f"    Sampled {len(sampled_indices)} images from {len(set(image_to_identity.get(idx_to_image.get(i, ''), '') for i in sampled_indices))} identities")

    # Build embedding similarity matrix
    print("\n[3] Building embedding similarity matrix...")
    embeddings_dict = {
        idx_to_image.get(i, f"img_{i}"): embeddings[i]
        for i in sampled_indices
    }

    sample_image_ids = [idx_to_image.get(i, f"img_{i}") for i in sampled_indices]

    emb_sim = EmbeddingSimilarityMatrix(embeddings_dict)
    embedding_matrix = emb_sim.build_matrix(sample_image_ids, metric="cosine")
    print(f"    Matrix shape: {embedding_matrix.shape}")

    # Generate synthetic human similarity judgments
    # NOTE: In reality, this comes from human annotation
    print("\n[4] Generating synthetic human judgments (for demo)...")
    print("    NOTE: Real RSA requires human annotation, not synthetic!")

    human_matrix = generate_synthetic_human_similarity(
        n_images=len(sampled_indices),
        noise_level=synthetic_human_noise,
        seed=42
    )

    # Run RSA comparison
    print("\n[5] Running RSA comparison...")
    comparator = RSAComparator()
    result = comparator.compare(human_matrix, embedding_matrix, method="all")

    # Interpret
    interpretation = interpret_rsa_result(result)
    print(interpretation)

    # Additional analysis: correlation breakdown by similarity range
    print("\n[6] Correlation by similarity range...")

    human_vec = emb_sim._matrix_to_vector if hasattr(emb_sim, '_matrix_to_vector') else None

    # This would show if correlation is better for similar vs dissimilar pairs
    print("    (In real analysis, would break down by similarity quartiles)")

    # Build attributes for potential subset analysis
    print("\n[7] Generating synthetic attributes...")
    attributes, identities = generate_attribute_annotations(image_to_identity, idx_to_image)
    print(f"    Generated attributes for {len(attributes)} images")
    print(f"    Attributes: style_type (5 levels), region (3 levels), era (2 levels)")

    # Save results
    print("\n[8] Saving results...")

    results = {
        "metadata": {
            "n_images": len(sampled_indices),
            "n_total": n_total,
            "embedding_dim": metadata['embedding_dim'],
            "synthetic_human_noise": synthetic_human_noise,
            "note": "Human judgments are SYNTHETIC - real RSA requires human annotation"
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
        "interpretation_full": interpretation
    }

    output_file = os.path.join(output_dir, "rsa_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"    Results saved to: {output_file}")

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    rho = result.spearman_rho

    print(f"\nRSA Spearman rho: {rho:.4f}")

    if rho < 0:
        print("\n[WARN] NEGATIVE correlation detected!")
        print("  Embedding space may INVERT human perception.")
        print("  RECOMMENDATION: STOP or redesign embeddings.")

    elif rho < 0.3:
        print("\n[WARN] WEAK correlation (< 0.3)")
        print("  Embeddings don't strongly reflect human similarity judgments.")
        print("  RECOMMENDATION:")
        print("    1. Try alternative embeddings (SigLIP, CLIP, DINOv2)")
        print("    2. Fine-tune embeddings on perceptual similarity task")
        print("    3. Use face-specific features instead of full appearance")

    elif rho < 0.5:
        print("\n[CAUTION] MODERATE correlation (0.3-0.5)")
        print("  Partial validation of embedding space.")
        print("  RECOMMENDATION: Proceed with downstream tasks but be cautious.")

    else:
        print("\n[PASS] GOOD correlation (> 0.5)")
        print("  Embedding space reflects human perception.")
        print("  RECOMMENDATION: Proceed with preference modeling.")

    print("\n" + "-" * 60)
    print("NEXT STEPS:")
    print("  1. Collect REAL human similarity judgments")
    print("  2. Replace synthetic human_matrix with real annotations")
    print("  3. Re-run RSA with human data")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RSA evaluation")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n_images", type=int, default=50, help="Number of images for RSA")
    parser.add_argument("--noise", type=float, default=0.3, help="Synthetic human noise level")

    args = parser.parse_args()

    run_rsa_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_sample_images=args.n_images,
        synthetic_human_noise=args.noise
    )
