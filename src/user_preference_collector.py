"""
User Preference Collector

Thu thập preference data từ user để system hiểu "ý bạn"

Data types:
1. Pairwise comparisons (priority)
2. Star ratings
3. Feature feedback
4. Implicit behavior
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime
import random


@dataclass
class PairwiseAnnotation:
    """Một cặp so sánh preference."""
    pair_id: int
    video_a_id: str
    video_b_id: str
    choice: str  # "A", "B", "equal", "skip"
    confidence: int  # 1-5
    time_seconds: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StarRating:
    """Rating cho một video."""
    video_id: str
    rating: int  # 1-5
    watched_fully: bool
    liked: bool
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureFeedback:
    """Feedback về features của video."""
    video_id: str
    features: List[str]  # ["smile", "eyes", "movement", ...]
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UserPreferenceData:
    """Tất cả preference data của một user."""
    user_id: str
    pairwise: List[PairwiseAnnotation]
    ratings: List[StarRating]
    feature_feedback: List[FeatureFeedback]
    implicit_actions: List[dict]  # Views, clicks, etc.

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "pairwise": [p.to_dict() for p in self.pairwise],
            "ratings": [r.to_dict() for r in self.ratings],
            "feature_feedback": [f.to_dict() for f in self.feature_feedback],
            "implicit_actions": self.implicit_actions
        }


class PreferenceCollector:
    """
    Thu thập preference data từ user.

    Usage:
    collector = PreferenceCollector(user_id="user_001")
    collector.add_pairwise("video_a", "video_b", "A", confidence=4)
    collector.add_rating("video_1", rating=5, watched_fully=True)
    collector.add_feature("video_1", features=["smile", "eyes"])
    collector.save()
    """

    FEATURES = [
        "smile",      # Nụ cười
        "eyes",       # Mắt đẹp
        "hair",       # Tóc
        "movement",   # Cách di chuyển
        "expression", # Biểu cảm
        "face_shape", # Khuôn mặt
        "voice",      # Giọng nói
        "style",      # Phong cách
        "pose",       # Tư thế
        "energy"      # Năng lượng
    ]

    def __init__(self, user_id: str, output_dir: str = "./data/user_preferences"):
        self.user_id = user_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pairwise: List[PairwiseAnnotation] = []
        self.ratings: List[StarRating] = []
        self.feature_feedback: List[FeatureFeedback] = []
        self.implicit_actions: List[dict] = []

    def add_pairwise(
        self,
        video_a_id: str,
        video_b_id: str,
        choice: str,
        confidence: int = 3,
        time_seconds: float = 0.0
    ) -> PairwiseAnnotation:
        """Thêm một pairwise comparison."""
        ann = PairwiseAnnotation(
            pair_id=len(self.pairwise),
            video_a_id=video_a_id,
            video_b_id=video_b_id,
            choice=choice,
            confidence=confidence,
            time_seconds=time_seconds,
            timestamp=datetime.now().isoformat()
        )
        self.pairwise.append(ann)
        return ann

    def add_rating(
        self,
        video_id: str,
        rating: int,
        watched_fully: bool = False,
        liked: bool = False
    ) -> StarRating:
        """Thêm một star rating."""
        rating_obj = StarRating(
            video_id=video_id,
            rating=rating,
            watched_fully=watched_fully,
            liked=liked,
            timestamp=datetime.now().isoformat()
        )
        self.ratings.append(rating_obj)
        return rating_obj

    def add_feature(
        self,
        video_id: str,
        features: List[str]
    ) -> FeatureFeedback:
        """Thêm feature feedback."""
        # Validate features
        valid_features = [f for f in features if f in self.FEATURES]

        fb = FeatureFeedback(
            video_id=video_id,
            features=valid_features,
            timestamp=datetime.now().isoformat()
        )
        self.feature_feedback.append(fb)
        return fb

    def add_implicit_action(
        self,
        video_id: str,
        action_type: str,  # "view", "click", "save", "share", "replay"
        duration_seconds: float = 0.0
    ) -> dict:
        """Thêm implicit action (tự động)."""
        action = {
            "video_id": video_id,
            "action_type": action_type,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        }
        self.implicit_actions.append(action)
        return action

    def generate_pairs(self, video_ids: List[str], n_pairs: int = 50) -> List[dict]:
        """Tạo random pairs từ video list."""
        pairs = []
        video_ids = list(set(video_ids))  # Remove duplicates

        if len(video_ids) < 2:
            return []

        # Generate random pairs
        for _ in range(n_pairs):
            a, b = random.sample(video_ids, 2)
            pairs.append({
                "video_a": a,
                "video_b": b,
                "position": random.choice(["A_left", "B_left"])  # Randomize position
            })

        return pairs

    def get_preference_model_input(self) -> dict:
        """
        Convert sang format cho preference model.

        Returns:
            dict với:
            - pairs: List of (video_a, video_b, prefer_a) tuples
            - ratings: Dict[video_id, rating]
            - features: Dict[video_id, List[features]]
        """
        # Convert pairwise to Bradley-Terry format
        pairs = []
        for ann in self.pairwise:
            if ann.choice in ["A", "B"]:
                prefer_a = 1 if ann.choice == "A" else 0
                pairs.append((ann.video_a_id, ann.video_b_id, prefer_a))

        # Convert ratings
        ratings = {r.video_id: r.rating for r in self.ratings}

        # Convert features
        features = {f.video_id: f.features for f in self.feature_feedback}

        # Aggregate implicit
        implicit_scores = {}
        for action in self.implicit_actions:
            vid = action["video_id"]
            score = self._implicit_to_score(action["action_type"])
            implicit_scores[vid] = implicit_scores.get(vid, 0) + score

        return {
            "user_id": self.user_id,
            "n_pairwise": len(pairs),
            "n_ratings": len(ratings),
            "n_features": len(features),
            "n_implicit": len(implicit_scores),
            "pairs": pairs,
            "ratings": ratings,
            "features": features,
            "implicit_scores": implicit_scores
        }

    def _implicit_to_score(self, action_type: str) -> float:
        """Convert implicit action to score."""
        scores = {
            "view": 0.1,
            "click": 0.2,
            "save": 0.5,
            "share": 0.5,
            "replay": 0.3,
            "watch_full": 0.4
        }
        return scores.get(action_type, 0.0)

    def save(self) -> str:
        """Lưu tất cả data."""
        data = self.get_preference_model_input()

        output_path = self.output_dir / f"{self.user_id}_preferences.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def load(self, user_id: str = None) -> bool:
        """Load existing data."""
        load_id = user_id or self.user_id
        path = self.output_dir / f"{load_id}_preferences.json"

        if not path.exists():
            return False

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Restore pairwise
        for pair in data.get("pairs", []):
            self.add_pairwise(
                video_a_id=pair[0],
                video_b_id=pair[1],
                choice="A" if pair[2] == 1 else "B"
            )

        # Restore ratings
        for vid, rating in data.get("ratings", {}).items():
            self.add_rating(video_id=vid, rating=rating)

        # Restore features
        for vid, features in data.get("features", {}).items():
            self.add_feature(video_id=vid, features=features)

        return True

    def get_stats(self) -> dict:
        """Get statistics về data đã thu thập."""
        return {
            "user_id": self.user_id,
            "n_pairwise": len(self.pairwise),
            "n_ratings": len(self.ratings),
            "n_feature_feedback": len(self.feature_feedback),
            "n_implicit_actions": len(self.implicit_actions),
            "avg_confidence": sum(p.confidence for p in self.pairwise) / max(1, len(self.pairwise)),
            "rating_distribution": self._rating_distribution(),
            "feature_frequency": self._feature_frequency()
        }

    def _rating_distribution(self) -> dict:
        """Rating distribution."""
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in self.ratings:
            dist[r.rating] = dist.get(r.rating, 0) + 1
        return dist

    def _feature_frequency(self) -> dict:
        """Feature frequency."""
        freq = {f: 0 for f in self.FEATURES}
        for fb in self.feature_feedback:
            for feat in fb.features:
                freq[feat] = freq.get(feat, 0) + 1
        return freq


# ============================================================
# USAGE EXAMPLE
# ============================================================

def example_usage():
    """Ví dụ sử dụng."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Initialize collector cho user
    collector = PreferenceCollector(user_id="user_demo_001")

    # VIDEO LIST (thay bằng video IDs thực tế)
    video_ids = [f"video_{i:03d}" for i in range(100)]

    # STEP 1: Pairwise comparisons (20 pairs)
    print("STEP 1: Pairwise Comparisons")
    pairs = collector.generate_pairs(video_ids, n_pairs=20)

    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}: {pair['video_a']} vs {pair['video_b']}")

        # User choice (simulated)
        choice = random.choice(["A", "B"])
        collector.add_pairwise(
            video_a_id=pair['video_a'],
            video_b_id=pair['video_b'],
            choice=choice,
            confidence=random.randint(3, 5),
            time_seconds=random.uniform(1.0, 5.0)
        )

    # STEP 2: Star ratings (10 videos)
    print("\nSTEP 2: Star Ratings")
    sample_videos = random.sample(video_ids, 10)

    for vid in sample_videos:
        rating = random.randint(3, 5)
        collector.add_rating(
            video_id=vid,
            rating=rating,
            watched_fully=random.choice([True, False]),
            liked=rating >= 4
        )
        print(f"  {vid}: {rating} stars")

    # STEP 3: Feature feedback (5 videos)
    print("\nSTEP 3: Feature Feedback")
    feature_videos = random.sample(video_ids, 5)

    for vid in feature_videos:
        features = random.sample(PreferenceCollector.FEATURES, k=random.randint(1, 3))
        collector.add_feature(video_id=vid, features=features)
        print(f"  {vid}: {features}")

    # STEP 4: Implicit actions (simulated)
    print("\nSTEP 4: Implicit Actions")
    for vid in random.sample(video_ids, 20):
        action = random.choice(["view", "click", "save", "replay"])
        collector.add_implicit_action(
            video_id=vid,
            action_type=action,
            duration_seconds=random.uniform(5.0, 60.0)
        )

    # Save and get stats
    output_path = collector.save()
    print(f"\nSaved to: {output_path}")

    stats = collector.get_stats()
    print(f"\nStats:")
    print(f"  Pairwise comparisons: {stats['n_pairwise']}")
    print(f"  Star ratings: {stats['n_ratings']}")
    print(f"  Feature feedback: {stats['n_feature_feedback']}")
    print(f"  Implicit actions: {stats['n_implicit_actions']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"\n  Rating distribution: {stats['rating_distribution']}")
    print(f"\n  Feature frequency: {stats['feature_frequency']}")

    # Get model input
    model_input = collector.get_preference_model_input()
    print(f"\nModel input:")
    print(f"  Pairs for Bradley-Terry: {len(model_input['pairs'])}")
    print(f"  Videos with ratings: {len(model_input['ratings'])}")
    print(f"  Videos with features: {len(model_input['features'])}")


if __name__ == "__main__":
    example_usage()
