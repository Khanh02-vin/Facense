"""
Auto Label Remaining Clips
Dùng model để tự động label những clips chưa có rating
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("D:/Project/Face_project/data")
FEATURES_FILE = DATA_DIR / "dataset_processed/features.json"
LABELS_FILE = DATA_DIR / "user_labels.json"
OUTPUT_FILE = DATA_DIR / "auto_labels.json"


class AutoLabeler:
    """Auto-label clips dựa trên preference model."""

    FEATURE_NAMES = [
        'motion_energy', 'motion_peak', 'blur_score', 'brightness',
        'face_visibility', 'smile', 'eye_contact', 'face_symmetry',
        'head_yaw', 'head_pitch', 'head_roll'
    ]

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def load_features(self) -> Tuple[List[Dict], List[str]]:
        """Load all features."""
        with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
            features = json.load(f)
        clip_ids = [f['clip_name'] for f in features]
        return features, clip_ids

    def load_existing_labels(self) -> Dict[str, int]:
        """Load existing user labels."""
        if not LABELS_FILE.exists():
            return {}
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {l['clip_id']: l['rating'] for l in data.get('labels', [])}

    def extract_features(self, features_list: List[Dict]) -> np.ndarray:
        """Extract feature vectors."""
        X = []
        for f in features_list:
            vec = [f.get(name, 0.0) for name in self.FEATURE_NAMES]
            X.append(vec)
        return np.array(X)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train model on labeled data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_NAMES, importance)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def auto_label(self, confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Auto-label all clips.

        Returns list of predictions with confidence scores.
        """
        # Load data
        all_features, clip_ids = self.load_features()
        existing_labels = self.load_existing_labels()

        print("="*60)
        print("AUTO LABEL CLIPS")
        print("="*60)
        print(f"Total clips: {len(clip_ids)}")
        print(f"Already labeled: {len(existing_labels)}")
        print(f"Need to label: {len(clip_ids) - len(existing_labels)}")

        # Filter unlabeled
        unlabeled_indices = [
            i for i, cid in enumerate(clip_ids) if cid not in existing_labels
        ]

        if len(unlabeled_indices) == 0:
            print("\n✅ All clips already labeled!")
            return []

        # Prepare data
        X_all = self.extract_features(all_features)

        if len(existing_labels) >= 10:
            # Train on existing labels
            X_train = []
            y_train = []
            for i, cid in enumerate(clip_ids):
                if cid in existing_labels:
                    X_train.append(X_all[i])
                    y_train.append(existing_labels[cid])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            print(f"\n[MODEL] Training on {len(X_train)} labeled samples...")
            self.train(X_train, y_train)

            # Predict unlabeled
            X_unlabeled = X_all[unlabeled_indices]
            predictions = self.predict(X_unlabeled)

            # Calculate confidence (based on prediction variance from RF trees)
            pred_std = np.std([
                tree.predict(self.scaler.transform(X_unlabeled))
                for tree in self.model.estimators_
            ], axis=0)

            confidence = 1 / (1 + pred_std)

        else:
            # Not enough labeled data - use heuristic-based prediction
            print("\n[WARN] Not enough labeled data for ML model.")
            print("       Using feature-based heuristics...")

            X_unlabeled = X_all[unlabeled_indices]
            predictions = self._heuristic_predict(X_unlabeled)
            confidence = np.ones(len(predictions)) * 0.5

        # Build results
        results = []
        for idx, (unlabeled_idx, pred, conf) in enumerate(
            zip(unlabeled_indices, predictions, confidence)
        ):
            clip_id = clip_ids[unlabeled_idx]
            rating = max(1, min(5, round(pred)))

            results.append({
                'clip_id': clip_id,
                'predicted_rating': float(pred),
                'rounded_rating': rating,
                'confidence': float(conf),
                'identity': all_features[unlabeled_idx].get('identity', 'unknown'),
                'auto_labeled': True
            })

        # Sort by confidence (highest first for review)
        results.sort(key=lambda x: -x['confidence'])

        # Stats
        ratings = [r['rounded_rating'] for r in results]
        print(f"\n[RESULTS]")
        print(f"  Auto-labeled: {len(results)} clips")
        print(f"  Rating distribution:")
        for r in range(1, 6):
            count = ratings.count(r)
            pct = count / len(ratings) * 100
            bar = "█" * int(pct / 5)
            print(f"    {r}: {bar} {count} ({pct:.1f}%)")

        high_conf = sum(1 for c in confidence if c >= confidence_threshold)
        print(f"\n  High confidence (≥{confidence_threshold}): {high_conf}/{len(results)}")

        return results

    def _heuristic_predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback heuristic-based prediction."""
        # Simple heuristics based on features
        scores = np.zeros(len(X))

        # Smile is positive
        scores += X[:, 5] * 2  # smile

        # Eye contact is positive
        scores += X[:, 6] * 1.5  # eye_contact

        # Face symmetry is positive
        scores += X[:, 7] * 1  # face_symmetry

        # Motion too high is negative
        scores -= np.clip(X[:, 0] / 50, 0, 2)  # motion_energy

        # Blur is negative
        scores -= np.clip(X[:, 2] / 200, 0, 1)  # blur_score

        # Normalize to 1-5 scale
        scores_min, scores_max = scores.min(), scores.max()
        if scores_max > scores_min:
            normalized = (scores - scores_min) / (scores_max - scores_min)
        else:
            normalized = np.ones(len(scores)) * 3

        return normalized * 4 + 1  # Scale to 1-5

    def save(self, results: List[Dict]):
        """Save predictions."""
        output = {
            'total': len(results),
            'high_confidence': sum(1 for r in results if r['confidence'] >= 0.7),
            'predictions': results
        }

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved to: {OUTPUT_FILE}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Auto Label Clips')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold (default: 0.7)')
    args = parser.parse_args()

    labeler = AutoLabeler()
    results = labeler.auto_label(confidence_threshold=args.threshold)

    if results:
        labeler.save(results)
        print("\n" + "="*60)
        print("AUTO LABELING COMPLETE")
        print("="*60)
        print(f"\nNext steps:")
        print(f"  1. Review: python src/review_auto_labels.py")
        print(f"  2. Merge: python src/merge_labels.py")
    else:
        print("\n✅ No clips need auto-labeling!")


if __name__ == "__main__":
    main()