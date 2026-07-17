"""
Train Bradley-Terry Model with Real Preferences

Uses the centralized BradleyTerryModel from the preference module.
Outputs both detailed debug output and serve-ready bradley_terry_scores.json.

Usage:
    python evaluation/train_bradley_terry.py \
        --annotations ./data/annotations/annotations_result.json \
        --embeddings ./data/processed/embeddings.npz \
        --output ./data/processed/bradley_terry_scores.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preference_learning.preference import BradleyTerryModel, PairwiseSample


def load_annotations(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_embeddings_npz(path: Path) -> dict[str, np.ndarray]:
    """Load .npz and return {item_id: normalized_embedding}."""
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in data.files:
        vec = np.asarray(data[key], dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        out[key] = vec / norm if norm > 0 else vec
    return out


def build_paiwise_samples(
    annotations: list[dict],
    item_ids: set[str],
) -> list[PairwiseSample]:
    """Convert annotation records to PairwiseSample list.

    Expected annotation fields: identity_A, identity_B, choice ('A'/'B'/'equal').
    """
    samples: list[PairwiseSample] = []
    for a in annotations:
        winner = a.get('choice')
        if winner not in ('A', 'B'):
            continue  # skip equal / invalid
        id_a = a.get('identity_A', '')
        id_b = a.get('identity_B', '')
        if id_a not in item_ids or id_b not in item_ids:
            continue
        samples.append(
            PairwiseSample(
                user_id=a.get('annotator_id', 'unknown'),
                image_A=id_a,
                image_B=id_b,
                winner=winner,
            )
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Bradley-Terry Model")
    parser.add_argument('--annotations', type=Path,
                        default=Path('./data/annotations/annotations_result.json'))
    parser.add_argument('--embeddings', type=Path,
                        default=Path('./data/processed/embeddings.npz'))
    parser.add_argument('--output', type=Path,
                        default=Path('./data/processed/bradley_terry_scores.json'))
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='L2 regularisation strength')
    args = parser.parse_args()

    print("=" * 50)
    print("BRADLEY-TERRY MODEL TRAINING")
    print("=" * 50)

    # 1. Load data
    annotations = load_annotations(args.annotations)
    embeddings_dict = load_embeddings_npz(args.embeddings)
    item_ids = set(embeddings_dict.keys())

    print(f"\n[1] Data")
    print(f"    Annotations: {len(annotations)}")
    print(f"    Items in embedding index: {len(item_ids)}")

    # 2. Build pairwise samples
    samples = build_paiwise_samples(annotations, item_ids)
    print(f"    Valid pairwise samples: {len(samples)}")

    if not samples:
        print("[!] No valid pairs. Cannot train.")
        return 1

    # 3. Fit model
    print(f"\n[2] Fitting Bradley-Terry (alpha={args.alpha}) ...")
    bt = BradleyTerryModel(alpha=args.alpha)
    result = bt.fit(samples)

    print(f"    Converged: {result.convergence}")
    print(f"    Iterations: {result.n_iterations}")
    print(f"    Log-likelihood: {result.log_likelihood:.4f}")
    print(f"    Items scored: {len(result.item_scores)}")

    # 4. Show top / bottom
    sorted_items = sorted(result.item_scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n[3] Top 10 items")
    for name, score in sorted_items[:10]:
        print(f"    {name:<30s} {score:+.4f}")

    print(f"\n[4] Bottom 5 items")
    for name, score in sorted_items[-5:]:
        print(f"    {name:<30s} {score:+.4f}")

    # 5. Save serve-ready scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted_items), f, indent=2, ensure_ascii=False)
    print(f"\n[5] Saved serve-ready scores to: {args.output}")

    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
