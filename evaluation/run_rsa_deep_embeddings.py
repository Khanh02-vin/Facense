"""
RSA Evaluation for Deep Embeddings

Runs RSA comparison for CLIP, SigLIP, DINOv2 embeddings.

RSA tests: Do embeddings reflect human perception?

Usage:
    python -m evaluation.run_rsa_deep_embeddings \
        --human_similarity_csv ./data/human_similarity.csv
    python -m evaluation.run_rsa_deep_embeddings --synthetic   # demo only
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preference_learning.rsa import RSAComparator, SimilarityMatrixBuilder, EmbeddingSimilarityMatrix


def _load_human_matrix(path: str):
    """Load human similarity judgments from CSV into a builder.

    Expected columns: image_A, image_B, similarity, annotator_id (optional).
    similarity should be in [0, 1] or [1, 5] (will be normalised).
    """
    builder = SimilarityMatrixBuilder()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim = float(row['similarity'])
            # Normalise 1-5 scale → 0-1 if applicable
            if sim > 1.0 + 1e-6:
                sim = (sim - 1.0) / 4.0
            sim = max(0.0, min(1.0, sim))
            builder.add_judgment(
                image_A=row['image_A'],
                image_B=row['image_B'],
                similarity=sim,
                annotator_id=row.get('annotator_id', 'unknown'),
            )
    return builder


def run_rsa_for_model(
    embeddings_file: str,
    model_name: str,
    human_builder: SimilarityMatrixBuilder | None = None,
    n_images: int = 60,
    seed: int = 42,
):
    """Run RSA for a specific embedding model."""
    print(f"\n{'=' * 60}")
    print(f"RSA for {model_name.upper()}")
    print(f"{'=' * 60}")

    # Load embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings: {embeddings.shape}")

    if embeddings.ndim == 4:
        embeddings = embeddings.mean(axis=2).squeeze(1)
    elif embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)
    print(f"Final shape: {embeddings.shape}")

    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings = embeddings / norms

    # Sample
    rng = np.random.default_rng(seed)
    if len(embeddings) > n_images:
        indices = rng.choice(len(embeddings), n_images, replace=False)
    else:
        indices = list(range(len(embeddings)))
    sampled_embeddings = embeddings[indices]
    image_ids = [f"img_{i}" for i in indices]

    # Embedding similarity matrix
    emb_sim = EmbeddingSimilarityMatrix(
        {f"img_{i}": sampled_embeddings[j] for j, i in enumerate(indices)}
    )
    embedding_matrix = emb_sim.build_matrix(image_ids, metric="cosine")

    # Build human similarity matrix
    if human_builder is not None:
        human_matrix, _ = human_builder.build_matrix(image_ids)
    else:
        # Without human data we cannot run meaningful RSA
        return {
            "model": model_name,
            "status": "not_available",
            "reason": "No human similarity data provided.",
        }

    # RSA comparison
    comparator = RSAComparator()
    result = comparator.compare(human_matrix, embedding_matrix, method="all")

    print(f"\nResults:")
    print(f"  Spearman rho: {result.spearman_rho:.4f} (p={result.spearman_p:.4e})")
    print(f"  Pearson r:    {result.pearson_r:.4f} (p={result.pearson_p:.4e})")
    print(f"  Kendall tau:  {result.kendall_tau:.4f}")

    rho = result.spearman_rho
    if rho < 0:
        interp = "NEGATIVE"
    elif rho < 0.3:
        interp = "WEAK"
    elif rho < 0.5:
        interp = "MODERATE"
    elif rho < 0.7:
        interp = "GOOD"
    else:
        interp = "STRONG"
    print(f"  Interpretation: {interp} (rho={rho:.3f})")

    return {
        "model": model_name,
        "status": "completed",
        "n_images": n_images,
        "embedding_dim": int(embeddings.shape[1]),
        "spearman_rho": float(result.spearman_rho),
        "spearman_p": float(result.spearman_p),
        "pearson_r": float(result.pearson_r),
        "pearson_p": float(result.pearson_p),
        "kendall_tau": float(result.kendall_tau),
        "interpretation": interp,
    }


def run_all_rsa(
    data_dir: str = "./data/processed",
    output_dir: str = "./results",
    human_similarity_csv: str | None = None,
    n_images: int = 60,
    synthetic: bool = False,
):
    """Run RSA for all deep embedding models."""
    print("=" * 60)
    print("RSA Evaluation for Deep Embeddings")
    print("=" * 60)
    print()

    os.makedirs(output_dir, exist_ok=True)
    models = ["clip", "siglip", "dinov2"]
    all_results = []

    # Load human judgments (required unless synthetic)
    human_builder = None
    if human_similarity_csv and os.path.exists(human_similarity_csv):
        human_builder = _load_human_matrix(human_similarity_csv)
        print(f"[+] Loaded human judgments from {human_similarity_csv}")
    elif synthetic:
        print("[!] Synthetic mode — human similarity will be generated from embeddings.")
    else:
        print("[!] No --human_similarity_csv provided.\n"
              "    RSA cannot run meaningfully without human similarity data.\n"
              "    Pass --synthetic for demo mode with synthetic data.")
        # Still try to report availability per model
        for model in models:
            emb_file = os.path.join(data_dir, f"embeddings_{model}.npy")
            all_results.append({
                "model": model,
                "status": "not_available",
                "reason": "No human similarity data (pass --human_similarity_csv or --synthetic).",
            })

    for model in models:
        embeddings_file = os.path.join(data_dir, f"embeddings_{model}.npy")
        if not os.path.exists(embeddings_file):
            print(f"\n{model.upper()}: No embeddings found")
            all_results.append({"model": model, "status": "no_embeddings"})
            continue

        if synthetic and human_builder is None:
            # Generate deterministic synthetic human judgments for demo
            from src.preference_learning.rsa import generate_synthetic_human_similarity
            syn = np.load(embeddings_file)
            n = min(n_images, len(syn))
            human_builder = SimilarityMatrixBuilder()
            h_mat = generate_synthetic_human_similarity(n, noise_level=0.25, seed=42)
            ids = [f"img_{i}" for i in range(n)]
            # Populate builder from matrix
            for i in range(n):
                for j in range(i + 1, n):
                    human_builder.add_judgment(ids[i], ids[j], float(h_mat[i, j]))
            print(f"[!] Synthetic human matrix generated ({n}x{n})")

        result = run_rsa_for_model(
            embeddings_file,
            model,
            human_builder=human_builder,
            n_images=n_images,
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: RSA Comparison")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Status':<20} {'rho':>10} {'Interpretation':<25}")
    print("-" * 70)

    valid = [r for r in all_results if r.get("status") == "completed"]
    for r in all_results:
        if r.get("status") == "completed":
            sig = "*" if r.get("spearman_p", 1) < 0.05 else " "
            print(f"{r['model']:<15} {'completed':<20} {r.get('spearman_rho', 0):>10.4f} {sig} {r.get('interpretation', ''):<25}")
        else:
            print(f"{r['model']:<15} {r.get('status', 'unknown'):<20} {'—':>10} {r.get('reason', ''):<25}")

    if valid:
        best = max(valid, key=lambda x: x.get("spearman_rho", 0))
        print(f"\nBest model: {best['model']} (rho = {best['spearman_rho']:.4f})")
        if best.get("spearman_rho", 0) >= 0.4:
            print("\n[PASS] Best embedding achieves RSA >= 0.4")
        else:
            print("\n[WARN] No embedding achieves RSA >= 0.4")

    output_file = os.path.join(output_dir, "deep_embedding_rsa_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSA for deep embeddings")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--human_similarity_csv", type=str, default=None,
                        help="CSV with columns: image_A, image_B, similarity [, annotator_id]")
    parser.add_argument("--n_images", type=int, default=60)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for demo/testing.")
    args = parser.parse_args()

    run_all_rsa(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        human_similarity_csv=args.human_similarity_csv,
        n_images=args.n_images,
        synthetic=args.synthetic,
    )
