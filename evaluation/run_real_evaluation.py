"""
Run Evaluation with Real Dataset

Uses extracted embeddings and generates synthetic preference labels
to test the full evaluation pipeline.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


def load_dataset(data_dir="./data/processed"):
    """Load embeddings and metadata."""
    data_dir = Path(data_dir)

    # Load embeddings
    embeddings = np.load(data_dir / "embeddings.npy")
    print(f"[INFO] Loaded embeddings: {embeddings.shape}")

    # Load mappings
    with open(data_dir / "image_to_idx.json") as f:
        image_to_idx = json.load(f)

    with open(data_dir / "image_to_identity.json", encoding='utf-8') as f:
        image_to_identity = json.load(f)

    with open(data_dir / "image_to_path.json", encoding='utf-8') as f:
        image_to_path = json.load(f)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    return embeddings, image_to_idx, image_to_identity, image_to_path, metadata


def generate_synthetic_preferences(
    embeddings,
    image_to_identity,
    image_to_idx,
    n_users=20,
    preferences_per_user=20
):
    """Generate synthetic preference labels.

    Creates preference labels based on embedding similarity to a "virtual preference vector".
    This simulates having actual user preference data.
    """
    np.random.seed(42)

    # image_to_idx maps image_name -> index in embeddings matrix
    idx_to_image = {int(v): k for k, v in image_to_idx.items()}
    identities = list(set(image_to_identity.values()))

    # Build index to identity mapping
    idx_to_identity = {}
    for img_name, identity in image_to_identity.items():
        if img_name in image_to_idx:
            idx = int(image_to_idx[img_name])
            idx_to_identity[idx] = identity

    # Create synthetic users, each with a "type preference"
    user_data = {}

    for user_id in range(n_users):
        # Each user prefers a random "type" (subset of identities)
        n_preferred_types = np.random.randint(3, 8)
        preferred_types = np.random.choice(identities, n_preferred_types, replace=False).tolist()

        # Get indices of preferred identities
        preferred_indices = [
            idx for idx, identity in idx_to_identity.items()
            if identity in preferred_types
        ]

        if len(preferred_indices) < 5:
            continue

        # Create preference vector from preferred images
        pref_emb = np.mean(embeddings[preferred_indices], axis=0)
        pref_emb = pref_emb / np.linalg.norm(pref_emb)

        # Generate labels based on similarity to preference vector
        similarities = embeddings @ pref_emb

        # Top 30% are "liked", bottom 30% are "disliked", rest is "neutral"
        threshold_high = np.percentile(similarities, 70)
        threshold_low = np.percentile(similarities, 30)

        labels = np.zeros(len(embeddings), dtype=int)
        labels[similarities >= threshold_high] = 1  # Liked
        labels[similarities <= threshold_low] = 0    # Disliked

        # Neutral (don't include in binary task)
        neutral_mask = (similarities > threshold_low) & (similarities < threshold_high)

        # Keep only liked and disliked
        binary_mask = ~neutral_mask
        binary_embeddings = embeddings[binary_mask]
        binary_labels = labels[binary_mask]
        binary_indices = np.where(binary_mask)[0]

        user_data[user_id] = {
            "user_id": f"user_{user_id}",
            "preferred_types": preferred_types,
            "preference_vector": pref_emb,
            "n_liked": int(np.sum(binary_labels == 1)),
            "n_disliked": int(np.sum(binary_labels == 0)),
            "indices": binary_indices.tolist(),
            "labels": binary_labels.tolist()
        }

    return user_data


def build_identity_groups(image_to_identity):
    """Build identity groups for identity control."""
    groups = {}
    for image_id, identity in image_to_identity.items():
        if identity not in groups:
            groups[identity] = []
        groups[identity].append(image_id)
    return groups


def run_evaluation_with_real_data(data_dir="./data/processed", output_dir="./results"):
    """Run full evaluation pipeline with real data."""
    print("=" * 60)
    print("Evaluation with Real Dataset")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n[1] Loading dataset...")
    embeddings, image_to_idx, image_to_identity, image_to_path, metadata = load_dataset(data_dir)

    n_images = metadata['n_images']
    n_identities = metadata['n_identities']
    embedding_dim = metadata['embedding_dim']

    print(f"    Images: {n_images}")
    print(f"    Identities: {n_identities}")
    print(f"    Embedding dim: {embedding_dim}")

    # Generate synthetic preferences
    print("\n[2] Generating synthetic preferences...")
    user_data = generate_synthetic_preferences(
        embeddings, image_to_identity, image_to_idx,
        n_users=20, preferences_per_user=20
    )

    total_samples = sum(len(d['indices']) for d in user_data.values())
    print(f"    Generated preferences for {len(user_data)} users")
    print(f"    Total labeled samples: {total_samples}")

    # Build identity groups
    identity_groups = build_identity_groups(image_to_identity)
    print(f"    Identity groups: {len(identity_groups)}")

    # Build idx_to_identity mapping
    idx_to_identity = {}
    for img_name, identity in image_to_identity.items():
        if img_name in image_to_idx:
            idx = int(image_to_idx[img_name])
            idx_to_identity[idx] = identity

    # Build idx_to_image mapping
    idx_to_image = {int(v): k for k, v in image_to_idx.items()}

    # Run evaluation phases
    from src.validation import RepresentationValidator, StabilityValidator
    from src.null_models import NullModelSuite, interpret_null_results
    from src.preference import BradleyTerryModel, PairwiseSample, MixtureOfPrototypes
    from src.retrieval import RetrievalEngine, RetrievalEvaluator, IdentityControlledRetrieval
    from src.cycle_diagnostics import CycleDetector, build_pairwise_matrix

    results = {
        "metadata": metadata,
        "dataset_summary": {
            "n_images": n_images,
            "n_identities": n_identities,
            "n_users": len(user_data),
            "total_labeled_samples": total_samples
        },
        "phases": {}
    }

    # Phase 1: Representation Validation
    print("\n[3] Phase 1: Representation Validation...")
    validator = RepresentationValidator(stability_threshold=0.85)

    # Use aggregated user preferences as labels (majority vote)
    all_labels = np.zeros(n_images, dtype=int)
    label_counts = np.zeros((n_images, 2))  # [disliked, liked] counts

    for user_id, data in user_data.items():
        for idx, label in zip(data['indices'], data['labels']):
            label_counts[idx, label] += 1

    # Majority vote
    all_labels = (label_counts[:, 1] > label_counts[:, 0]).astype(int)

    # User IDs for cross-user test (use identity as user proxy)
    all_user_ids = np.array([
        image_to_identity.get(idx_to_image.get(i, ""), "unknown")
        for i in range(n_images)
    ])

    val_results = validator.validate(
        embeddings={"real_data": embeddings},
        labels=all_labels
    )

    results["phases"]["representation_validation"] = {
        "status": "completed",
        "n_samples": n_images,
        "embedding_dim": embedding_dim
    }
    print("    [OK] Representation validation completed")

    # Phase 2: Null Models
    print("\n[4] Phase 2: Null Models Testing...")
    suite = NullModelSuite(n_permutations=500, alpha=0.05)
    null_results = suite.run_all(embeddings, all_labels, all_user_ids)

    interpretation = interpret_null_results(null_results)

    results["phases"]["null_models"] = {
        "interpretation": interpretation,
        "summary": null_results.get("summary", {})
    }

    # Count rejected
    n_rejected = sum(1 for k, r in null_results.items()
                     if isinstance(r, type(null_results.get("label_permutation")))
                     and hasattr(r, 'null_rejected') and r.null_rejected)

    print(f"    Label Permutation: {'REJECTED' if null_results['label_permutation'].null_rejected else 'NOT REJECTED'}")
    print(f"    Feature Shuffle: {'REJECTED' if null_results['feature_shuffle'].null_rejected else 'NOT REJECTED'}")
    print(f"    Cross User: {'REJECTED' if null_results['cross_user'].null_rejected else 'NOT REJECTED'}")

    # Phase 3: Pairwise Preference Learning
    print("\n[5] Phase 3: Pairwise Preference Learning...")

    # Generate pairwise data from synthetic preferences
    pairwise_samples = []
    for user_id, data in user_data.items():
        liked = [data['indices'][i] for i, l in enumerate(data['labels']) if l == 1]
        disliked = [data['indices'][i] for i, l in enumerate(data['labels']) if l == 0]

        # Create pairs: liked vs disliked
        for l_idx in liked[:5]:
            for d_idx in disliked[:5]:
                pairwise_samples.append(PairwiseSample(
                    user_id=f"user_{user_id}",
                    image_A=f"img_{l_idx}",
                    image_B=f"img_{d_idx}",
                    winner="A"
                ))

    # Bradley-Terry
    bt_model = BradleyTerryModel(max_iterations=100)
    bt_result = bt_model.fit(pairwise_samples)

    # Mixture of Prototypes
    positive_indices = []
    for data in user_data.values():
        positive_indices.extend([i for i, l in zip(data['indices'], data['labels']) if l == 1])

    positive_embeddings = embeddings[positive_indices]
    mixture = MixtureOfPrototypes(n_prototypes=5)
    mixture_result = mixture.fit(positive_embeddings)

    results["phases"]["preference_learning"] = {
        "bradley_terry": {
            "n_items": len(bt_result.item_scores),
            "converged": bt_result.convergence
        },
        "mixture_prototypes": {
            "n_prototypes": mixture_result["n_prototypes"]
        }
    }
    print(f"    Bradley-Terry: {len(bt_result.item_scores)} items, converged: {bt_result.convergence}")
    print(f"    Mixture of Prototypes: {mixture_result['n_prototypes']} prototypes")

    # Phase 4: Retrieval Evaluation
    print("\n[6] Phase 4: Retrieval Evaluation...")

    # Build image_id to embedding dict
    idx_to_image = {int(v): k for k, v in image_to_idx.items()}
    embeddings_dict = {
        f"img_{idx}": embeddings[idx]
        for idx in range(len(embeddings))
    }

    engine = RetrievalEngine(embeddings_dict)
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

    # Evaluate on synthetic preferences
    eval_results = []
    for user_id, data in list(user_data.items())[:10]:
        query_idx = data['indices'][0]
        query_id = f"img_{query_idx}"

        retrieved = engine.retrieve(query_id, k=10)

        # Get synthetic ratings
        ratings = []
        for r_id in retrieved:
            r_idx = int(r_id.replace("img_", ""))
            # Check if retrieved image is in user's liked or disliked
            if r_idx in data['indices']:
                label_idx = data['indices'].index(r_idx)
                rating = 5 if data['labels'][label_idx] == 1 else 2
            else:
                rating = 3  # Neutral
            ratings.append(rating)

        metrics = evaluator.evaluate(query_id, retrieved, ratings)
        from src.retrieval import RetrievalResult
        eval_results.append(RetrievalResult(
            query_id=query_id,
            candidate_pool=retrieved,
            retrieved_ids=retrieved,
            user_ratings=ratings,
            metrics=metrics
        ))

    batch_metrics = evaluator.evaluate_batch(eval_results)

    results["phases"]["retrieval"] = {
        "n_queries": len(eval_results),
        "metrics": batch_metrics
    }
    print(f"    Evaluated {len(eval_results)} queries")
    print(f"    Mean Hit@10: {batch_metrics.get('mean_hit_rate', 0):.4f}")
    print(f"    Mean MRR: {batch_metrics.get('mean_mrr', 0):.4f}")
    print(f"    Mean NDCG: {batch_metrics.get('mean_ndcg', 0):.4f}")

    # Phase 5: Identity Leakage Analysis
    print("\n[7] Phase 5: Identity Leakage Analysis...")

    # Check if retrieved images are from same identity as query
    identity_leakage = []
    for result in eval_results:
        query_id = result.query_id
        query_idx = int(query_id.replace("img_", ""))
        query_identity = idx_to_identity.get(query_idx, "")

        retrieved = engine.retrieve(query_id, k=10, exclude_ids=[query_id])
        same_identity_count = sum(
            1 for r_id in retrieved
            if idx_to_identity.get(int(r_id.replace("img_", "")), "") == query_identity
        )

        identity_leakage.append(same_identity_count / len(retrieved))

    results["phases"]["identity_analysis"] = {
        "mean_same_identity_ratio": float(np.mean(identity_leakage)),
        "interpretation": (
            "HIGH leakage (>0.5): retrieval relies on identity memorization"
            if np.mean(identity_leakage) > 0.5
            else "LOW leakage: retrieval captures genuine preference patterns"
        )
    }
    print(f"    Mean same-identity ratio: {np.mean(identity_leakage):.2%}")
    print(f"    Interpretation: {results['phases']['identity_analysis']['interpretation']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_nulls_rejected = all([
        null_results['label_permutation'].null_rejected,
        null_results['feature_shuffle'].null_rejected
    ])

    print(f"\nNull Models:")
    print(f"  Label Permutation: {'REJECTED' if null_results['label_permutation'].null_rejected else 'NOT REJECTED'}")
    print(f"  Feature Shuffle: {'REJECTED' if null_results['feature_shuffle'].null_rejected else 'NOT REJECTED'}")
    print(f"  Cross User: {'REJECTED' if null_results['cross_user'].null_rejected else 'NOT REJECTED'}")

    print(f"\nRetrieval:")
    print(f"  Hit@10: {batch_metrics.get('mean_hit_rate', 0):.4f}")
    print(f"  MRR: {batch_metrics.get('mean_mrr', 0):.4f}")

    print(f"\nIdentity Control:")
    print(f"  Same-identity ratio: {np.mean(identity_leakage):.2%}")

    # Recommendation
    print("\n" + "-" * 60)
    if all_nulls_rejected and np.mean(identity_leakage) < 0.3:
        print("[PASS] Strong signal with low identity leakage")
        results["recommendation"] = "Strong signal exists with good identity control"
    elif all_nulls_rejected:
        print("[WARN] Signal exists but with high identity leakage")
        results["recommendation"] = "Signal detected but identity control needed"
    else:
        print("[FAIL] Insufficient signal or weak results")
        results["recommendation"] = "Further investigation needed"

    # Save results
    output_file = os.path.join(output_dir, "real_evaluation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    run_evaluation_with_real_data(args.data_dir, args.output_dir)
