"""
Pairwise Preference Annotation Tool

Collects real human annotations for preference pairs.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
import cv2
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

import torch

# Load SigLIP embeddings
SIGLIP_EMBEDDINGS = None
SIGLIP_IDENTITY_MAP = None
FRAME_TO_IDENTITY = None


def load_siglip_embeddings(data_dir="./data/processed"):
    """Load SigLIP embeddings (prefer multiframe version)."""
    global SIGLIP_EMBEDDINGS, SIGLIP_IDENTITY_MAP, FRAME_TO_IDENTITY

    if SIGLIP_EMBEDDINGS is None:
        # Prefer multiframe version
        multiframe_emb_file = os.path.join(data_dir, "embeddings_siglip_multiframe.npy")
        multiframe_id_file = os.path.join(data_dir, "image_to_identity_multiframe.json")
        frame_file = os.path.join(data_dir, "frame_to_identity.json")

        if os.path.exists(multiframe_emb_file):
            embeddings_file = multiframe_emb_file
            identity_file = multiframe_id_file
        else:
            embeddings_file = os.path.join(data_dir, "embeddings_siglip.npy")
            identity_file = os.path.join(data_dir, "image_to_identity_siglip.json")

        if os.path.exists(embeddings_file):
            SIGLIP_EMBEDDINGS = np.load(embeddings_file)
            # Handle 4D shape: (N, 1, seq, dim) -> (N, dim)
            if SIGLIP_EMBEDDINGS.ndim == 4:
                SIGLIP_EMBEDDINGS = SIGLIP_EMBEDDINGS.mean(axis=2).squeeze(1)

            # Normalize
            norms = np.linalg.norm(SIGLIP_EMBEDDINGS, axis=1, keepdims=True) + 1e-10
            SIGLIP_EMBEDDINGS = SIGLIP_EMBEDDINGS / norms

        if os.path.exists(identity_file):
            with open(identity_file, 'r', encoding='utf-8') as f:
                SIGLIP_IDENTITY_MAP = json.load(f)

        if os.path.exists(frame_file):
            with open(frame_file, 'r', encoding='utf-8') as f:
                FRAME_TO_IDENTITY = json.load(f)

    return SIGLIP_EMBEDDINGS, SIGLIP_IDENTITY_MAP, FRAME_TO_IDENTITY


@dataclass
class PreferencePair:
    """A single preference judgment."""
    image_A: str
    image_B: str
    preference: str  # "A", "B", or "equal"
    annotator_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding_distance: Optional[float] = None
    same_identity: bool = False


@dataclass
class AnnotationSession:
    """Collection of preference annotations."""
    session_id: str
    annotator_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pairs: List[PreferencePair] = field(default_factory=list)

    def add_pair(self, pair: PreferencePair):
        self.pairs.append(pair)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "created_at": self.created_at,
            "n_pairs": len(self.pairs),
            "pairs": [
                {
                    "image_A": p.image_A,
                    "image_B": p.image_B,
                    "preference": p.preference,
                    "timestamp": p.timestamp,
                    "same_identity": p.same_identity,
                    "embedding_distance": p.embedding_distance
                }
                for p in self.pairs
            ]
        }


class PairGenerator:
    """Generates pairs for annotation based on embedding similarity."""

    def __init__(self, embeddings, identity_map, frame_to_identity=None, n_pairs=100):
        self.embeddings = embeddings
        self.identity_map = identity_map
        self.frame_to_identity = frame_to_identity
        self.n_pairs = n_pairs

        # Build identity -> indices mapping
        self.identity_to_indices = {}
        for idx_str, identity in self.identity_map.items():
            idx = int(idx_str)
            if identity not in self.identity_to_indices:
                self.identity_to_indices[identity] = []
            self.identity_to_indices[identity].append(idx)

    def generate_balanced_pairs(self, seed=42):
        """Generate balanced pairs: some same-identity, some different."""
        random.seed(seed)
        np.random.seed(seed)

        pairs = []
        identities = list(self.identity_to_indices.keys())

        n_same = self.n_pairs // 3
        n_diff = self.n_pairs - n_same

        # Same-identity pairs (from different frames of same video)
        same_added = 0
        attempts = 0
        while same_added < n_same and attempts < n_same * 10:
            attempts += 1
            identity = random.choice(identities)
            indices = self.identity_to_indices[identity]
            if len(indices) >= 2:
                a, b = random.sample(indices, 2)
                pairs.append((a, b, True))
                same_added += 1

        # Different-identity pairs
        for _ in range(n_diff):
            id1, id2 = random.sample(identities, 2)
            a = random.choice(self.identity_to_indices[id1])
            b = random.choice(self.identity_to_indices[id2])
            pairs.append((a, b, False))

        random.shuffle(pairs)
        return pairs
        return pairs

    def generate_similarity_based_pairs(self, n_pairs=50, similarity_range=(0.3, 0.7)):
        """Generate pairs in specific similarity range (harder pairs)."""
        n = len(self.embeddings)
        sim_matrix = self.embeddings @ self.embeddings.T

        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if similarity_range[0] <= sim <= similarity_range[1]:
                    candidates.append((i, j, sim))

        if len(candidates) > n_pairs:
            candidates = random.sample(candidates, n_pairs)

        return [(i, j, False) for i, j, _ in candidates]


def create_annotation_ui(
    pairs: List[tuple],
    identity_map: dict,
    output_file: str = "./data/annotations/annotations.json"
):
    """Generate annotation instructions and data file for external annotation."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    annotation_data = {
        "n_pairs": len(pairs),
        "pairs": [],
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "embedding": "SigLIP"
        }
    }

    for i, (idx_a, idx_b, same_identity) in enumerate(pairs):
        identity_a = identity_map.get(str(idx_a), f"unknown_{idx_a}")
        identity_b = identity_map.get(str(idx_b), f"unknown_{idx_b}")

        annotation_data["pairs"].append({
            "pair_id": i,
            "image_A_idx": int(idx_a),
            "image_B_idx": int(idx_b),
            "identity_A": identity_a,
            "identity_B": identity_b,
            "same_identity": same_identity,
            "preference": None  # To be filled by annotator
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(pairs)} pairs for annotation")
    print(f"Output: {output_file}")

    return annotation_data


def generate_annotation_report(
    annotations_file: str,
    embeddings,
    identity_map
):
    """Generate statistics from collected annotations."""
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data["pairs"]
    completed = [p for p in pairs if p["preference"] is not None]

    if not completed:
        print("No completed annotations found")
        return

    # Count preferences
    prefs = {"A": 0, "B": 0, "equal": 0}
    for p in completed:
        prefs[p["preference"]] = prefs.get(p["preference"], 0) + 1

    # Same vs different identity agreement
    same_identity_prefs = {"A": 0, "B": 0, "equal": 0}
    diff_identity_prefs = {"A": 0, "B": 0, "equal": 0}

    for p in completed:
        if p["same_identity"]:
            same_identity_prefs[p["preference"]] += 1
        else:
            diff_identity_prefs[p["preference"]] += 1

    print("\n" + "=" * 60)
    print("Annotation Report")
    print("=" * 60)
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Completed: {len(completed)} ({100*len(completed)/len(pairs):.1f}%)")
    print(f"\nPreference distribution:")
    print(f"  A preferred:    {prefs['A']:>4} ({100*prefs['A']/len(completed):>5.1f}%)")
    print(f"  B preferred:   {prefs['B']:>4} ({100*prefs['B']/len(completed):>5.1f}%)")
    print(f"  Equal:         {prefs['equal']:>4} ({100*prefs['equal']/len(completed):>5.1f}%)")

    if sum(same_identity_prefs.values()) > 0:
        print(f"\nSame identity pairs ({sum(same_identity_prefs.values())} pairs):")
        for k, v in same_identity_prefs.items():
            print(f"  {k}: {v}")
        print(f"\nDifferent identity pairs ({sum(diff_identity_prefs.values())} pairs):")
        for k, v in diff_identity_prefs.items():
            print(f"  {k}: {v}")

    return data


def run_annotation_collection(
    n_pairs: int = 100,
    same_identity_ratio: float = 0.33,
    output_dir: str = "./data/annotations"
):
    """Run the annotation collection process."""
    print("=" * 60)
    print("Pairwise Preference Annotation Collection")
    print("=" * 60)

    # Load embeddings
    print("\n[1] Loading SigLIP embeddings...")
    embeddings, identity_map, frame_to_identity = load_siglip_embeddings()
    print(f"    Loaded {len(embeddings)} embeddings ({embeddings.shape[1]} dims)")

    # Generate pairs
    print(f"\n[2] Generating {n_pairs} pairs...")
    generator = PairGenerator(embeddings, identity_map, frame_to_identity, n_pairs=n_pairs)
    pairs = generator.generate_balanced_pairs(seed=42)
    print(f"    Generated {len(pairs)} pairs")
    print(f"    Same identity: {sum(1 for _, _, s in pairs if s)}")
    print(f"    Different:     {sum(1 for _, _, s in pairs if not s)}")

    # Generate annotation UI data
    print(f"\n[3] Creating annotation data...")
    output_file = os.path.join(output_dir, "annotation_pairs.json")
    annotation_data = create_annotation_ui(pairs, identity_map, output_file)

    # Create simple annotation form template
    print(f"\n[4] Annotation instructions:")
    print("""
To collect annotations:
1. Share the image pairs with annotators
2. Use format: Which face do you prefer? [A] or [B]
3. Record preferences in the annotation_pairs.json file
4. Run: python collect_pairwise_annotations.py --analyze

Example annotation:
{
  "preference": "A"  // or "B" or "equal"
}
""")

    return annotation_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect pairwise preference annotations")
    parser.add_argument("--n_pairs", type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("--output_dir", type=str, default="./data/annotations")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing annotations")

    args = parser.parse_args()

    if args.analyze:
        # Analyze existing annotations
        annotations_file = os.path.join(args.output_dir, "annotation_pairs.json")
        embeddings, identity_map, _ = load_siglip_embeddings()
        generate_annotation_report(annotations_file, embeddings, identity_map)
    else:
        run_annotation_collection(
            n_pairs=args.n_pairs,
            output_dir=args.output_dir
        )
