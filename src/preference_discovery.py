"""
Preference Discovery from Liked Videos

Scenario: User có dataset videos họ THÍCH
- Không biết VÌ SAO thích
- Muốn KHÁM PHÁ gu của mình

Approach:
1. Extract features từ videos
2. Find patterns (clustering, frequency, centroid)
3. Explain preference
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import random

# For clustering
CLUSTERING_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    CLUSTERING_AVAILABLE = True
except ImportError:
    pass


@dataclass
class VideoFeatures:
    """Features extracted from one video."""
    video_id: str

    # Face features
    face_embedding: Optional[np.ndarray] = None
    smile_intensity: float = 0.0  # 0-1
    eye_contact_score: float = 0.0  # 0-1
    face_clarity: float = 0.0  # 0-1
    skin_tone: str = "unknown"  # light/medium/dark

    # Style features
    shot_type: str = "unknown"  # close_up/wide/medium
    lighting: str = "unknown"  # bright/dim/natural
    background: str = "unknown"  # simple/complex/outdoor

    # Motion features
    movement_level: float = 0.0  # 0-1
    energy_level: float = 0.0  # 0-1
    camera_stability: float = 0.0  # 0-1

    # Aesthetic
    attractiveness_score: float = 0.0  # 0-1
    blur_score: float = 0.0  # 0-1

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for analysis."""
        # Numeric features only
        return np.array([
            self.smile_intensity,
            self.eye_contact_score,
            self.face_clarity,
            self.movement_level,
            self.energy_level,
            self.camera_stability,
            self.attractiveness_score,
            self.blur_score
        ])

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "smile_intensity": self.smile_intensity,
            "eye_contact_score": self.eye_contact_score,
            "face_clarity": self.face_clarity,
            "shot_type": self.shot_type,
            "lighting": self.lighting,
            "movement_level": self.movement_level,
            "energy_level": self.energy_level,
            "attractiveness_score": self.attractiveness_score
        }


@dataclass
class PreferenceProfile:
    """User's discovered preference profile."""
    user_id: str
    n_videos_analyzed: int

    # Top features
    top_features: List[Tuple[str, float]]  # [(feature, score), ...]

    # Preference types/clusters
    preference_types: List[Dict]  # [{"type": "Warm", "percentage": 0.6}, ...]

    # Feature percentiles
    feature_percentiles: Dict[str, float]

    # Summary
    summary: str
    discovery_notes: List[str]


class PreferenceDiscoverer:
    """
    Discover user's preference pattern từ videos họ thích.

    Input: Dataset of videos (all are "liked")
    Output: Preference profile explaining WHAT they like
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id = user_id
        self.videos: List[VideoFeatures] = []

    def add_video(self, features: VideoFeatures):
        """Add video với extracted features."""
        self.videos.append(features)

    def analyze(self) -> PreferenceProfile:
        """
        Analyze all videos to discover preference pattern.

        Returns:
            PreferenceProfile với discovered patterns
        """
        if len(self.videos) < 3:
            raise ValueError("Need at least 3 videos for analysis")

        print(f"\n{'='*60}")
        print("PREFERENCE DISCOVERY ANALYSIS")
        print(f"{'='*60}")
        print(f"Videos analyzed: {len(self.videos)}")

        # Step 1: Extract feature vectors
        feature_vectors = []
        for v in self.videos:
            vec = v.to_vector()
            feature_vectors.append(vec)

        X = np.array(feature_vectors)
        print(f"Feature matrix: {X.shape}")

        # Step 2: Frequency Analysis
        print(f"\n{'='*60}")
        print("[1] FREQUENCY ANALYSIS")
        print(f"{'='*60}")
        freq_analysis = self._frequency_analysis()
        for key, value in freq_analysis.items():
            print(f"  {key}: {value}")

        # Step 3: Feature Percentiles
        print(f"\n{'='*60}")
        print("[2] FEATURE PERCENTILES (vs Dataset)")
        print(f"{'='*60}")
        percentiles = self._compute_percentiles(X)
        for feat, pct in sorted(percentiles.items(), key=lambda x: -x[1]):
            level = "HIGH" if pct > 70 else "MEDIUM" if pct > 40 else "LOW"
            print(f"  {feat}: {pct:.1f}th percentile ({level})")

        # Step 4: Clustering (if available)
        print(f"\n{'='*60}")
        print("[3] PREFERENCE TYPES DISCOVERY")
        print(f"{'='*60}")
        types = self._discover_types(X)
        for t in types:
            print(f"  Type: {t['name']}")
            print(f"  Percentage: {t['percentage']:.0%}")
            print(f"  Characteristics: {t['characteristics']}")
            print()

        # Step 5: Centroid Analysis
        print(f"{'='*60}")
        print("[4] PREFERENCE CENTROID")
        print(f"{'='*60}")
        centroid = X.mean(axis=0)
        feature_names = ['smile', 'eye_contact', 'face_clarity', 'movement',
                        'energy', 'camera_stability', 'attractiveness', 'blur']

        print("Average feature values:")
        for i, name in enumerate(feature_names):
            val = centroid[i]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {name:15s}: {bar} {val:.2f}")

        # Step 6: Generate Summary
        summary = self._generate_summary(percentiles, types)

        return PreferenceProfile(
            user_id=self.user_id,
            n_videos_analyzed=len(self.videos),
            top_features=self._get_top_features(percentiles),
            preference_types=types,
            feature_percentiles=percentiles,
            summary=summary,
            discovery_notes=self._generate_notes(percentiles, types)
        )

    def _frequency_analysis(self) -> Dict:
        """Analyze frequency of categorical features."""
        shot_types = Counter(v.shot_type for v in self.videos)
        lighting_types = Counter(v.lighting for v in self.videos)

        return {
            "Close-up shots": sum(1 for v in self.videos if v.shot_type == "close_up") / len(self.videos),
            "Wide shots": sum(1 for v in self.videos if v.shot_type == "wide") / len(self.videos),
            "Bright lighting": sum(1 for v in self.videos if v.lighting == "bright") / len(self.videos),
            "Natural lighting": sum(1 for v in self.videos if v.lighting == "natural") / len(self.videos),
        }

    def _compute_percentiles(self, X: np.ndarray) -> Dict[str, float]:
        """Compute percentile of each feature."""
        feature_names = ['smile', 'eye_contact', 'face_clarity', 'movement',
                        'energy', 'camera_stability', 'attractiveness', 'blur']

        percentiles = {}
        for i, name in enumerate(feature_names):
            mean_val = X[:, i].mean()
            # Assume uniform distribution for percentile (would need population data)
            percentiles[name] = mean_val * 100  # Placeholder

        return percentiles

    def _discover_types(self, X: np.ndarray) -> List[Dict]:
        """Cluster videos to find preference types."""
        if not CLUSTERING_AVAILABLE:
            return self._discover_types_simple(X)

        # K-means clustering
        n_clusters = min(3, len(self.videos))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        types = []
        feature_names = ['smile', 'eye_contact', 'face_clarity', 'movement',
                        'energy', 'camera_stability', 'attractiveness', 'blur']

        type_names = ["Warm & Approachable", "Dynamic & Energetic", "Calm & Serene"]

        for i in range(n_clusters):
            mask = labels == i
            count = mask.sum()
            pct = count / len(self.videos)

            # Find top characteristics
            cluster_mean = X[mask].mean(axis=0)
            sorted_idx = np.argsort(cluster_mean)[::-1]

            chars = []
            for idx in sorted_idx[:3]:
                if cluster_mean[idx] > 0.6:
                    chars.append(f"{feature_names[idx]}: high")

            types.append({
                "name": type_names[i] if i < len(type_names) else f"Type {i+1}",
                "percentage": pct,
                "characteristics": chars,
                "size": count
            })

        # Sort by size
        types.sort(key=lambda x: -x["percentage"])
        return types

    def _discover_types_simple(self, X: np.ndarray) -> List[Dict]:
        """Simple clustering without sklearn."""
        feature_names = ['smile', 'eye_contact', 'face_clarity', 'movement',
                        'energy', 'camera_stability', 'attractiveness', 'blur']

        # Simple split by attractiveness
        sorted_idx = np.argsort(X[:, 6])[::-1]  # Sort by attractiveness

        n = len(self.videos)
        types = []

        # High attractiveness
        high_mask = sorted_idx[:n//3]
        high_mean = X[high_mask].mean(axis=0)
        high_chars = [feature_names[i] for i in np.argsort(high_mean)[::-1][:2] if high_mean[i] > 0.5]

        types.append({
            "name": "Highly Attractive",
            "percentage": len(high_mask) / n,
            "characteristics": high_chars,
            "size": len(high_mask)
        })

        # Medium attractiveness
        mid_mask = sorted_idx[n//3:2*n//3]
        mid_mean = X[mid_mask].mean(axis=0)
        mid_chars = [feature_names[i] for i in np.argsort(mid_mean)[::-1][:2] if mid_mean[i] > 0.5]

        types.append({
            "name": "Moderately Attractive",
            "percentage": len(mid_mask) / n,
            "characteristics": mid_chars,
            "size": len(mid_mask)
        })

        # Low attractiveness
        low_mask = sorted_idx[2*n//3:]
        low_mean = X[low_mask].mean(axis=0)
        low_chars = [feature_names[i] for i in np.argsort(low_mean)[::-1][:2] if low_mean[i] > 0.5]

        types.append({
            "name": "Other",
            "percentage": len(low_mask) / n,
            "characteristics": low_chars,
            "size": len(low_mask)
        })

        return types

    def _get_top_features(self, percentiles: Dict) -> List[Tuple[str, float]]:
        """Get top features by percentile."""
        sorted_features = sorted(percentiles.items(), key=lambda x: -x[1])
        return [(name, pct) for name, pct in sorted_features[:5]]

    def _generate_summary(self, percentiles: Dict, types: List) -> str:
        """Generate human-readable summary."""
        top = sorted(percentiles.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join([f"{name} ({pct:.0f}%)" for name, pct in top])

        main_type = types[0]["name"] if types else "Unknown"

        return f"""
Your Preference Profile:
- Primary preference: {top_str}
- Main type: {main_type}
- Based on {len(self.videos)} videos analyzed
""".strip()

    def _generate_notes(self, percentiles: Dict, types: List) -> List[str]:
        """Generate discovery notes."""
        notes = []

        # Check for strong preferences
        high_features = [name for name, pct in percentiles.items() if pct > 70]
        if high_features:
            notes.append(f"You strongly prefer videos with: {', '.join(high_features)}")

        low_features = [name for name, pct in percentiles.items() if pct < 30]
        if low_features:
            notes.append(f"You tend to avoid: {', '.join(low_features)}")

        # Check for dominant type
        if types and types[0]["percentage"] > 0.5:
            notes.append(f"Your preference is dominated by '{types[0]['name']}' type ({types[0]['percentage']:.0%})")

        # Notes about variance
        notes.append(f"Variety in your preferences: {len(types)} distinct types identified")

        return notes


def simulate_video_features(video_ids: List[str]) -> List[VideoFeatures]:
    """Simulate feature extraction (replace with real extraction)."""
    features = []

    for vid in video_ids:
        # Simulate based on "some" pattern
        # In reality, this would be ML model inference

        # Create varied but somewhat consistent features
        base = random.uniform(0.4, 0.8)

        f = VideoFeatures(
            video_id=vid,
            smile_intensity=random.uniform(0.3, 0.9),
            eye_contact_score=random.uniform(0.4, 0.9),
            face_clarity=random.uniform(0.5, 0.9),
            shot_type=random.choice(["close_up", "close_up", "close_up", "wide", "medium"]),
            lighting=random.choice(["bright", "natural", "natural"]),
            movement_level=random.uniform(0.2, 0.7),
            energy_level=random.uniform(0.3, 0.8),
            attractiveness_score=base,
            blur_score=random.uniform(0.0, 0.3)
        )

        features.append(f)

    return features


def main():
    """Example usage."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("PREFERENCE DISCOVERY")
    print("=" * 60)
    print("""
Scenario: You have a dataset of videos YOU LIKE
Goal: Discover WHAT makes you like them
""")

    # Simulate: User có 30 videos họ thích
    print("[1] Simulating 30 videos you like...")

    video_ids = [f"video_{i:03d}" for i in range(30)]
    videos = simulate_video_features(video_ids)

    print(f"  Added {len(videos)} videos to analysis")

    # Step 2: Analyze
    print("\n[2] Analyzing preference patterns...")

    discoverer = PreferenceDiscoverer(user_id="user_001")
    for v in videos:
        discoverer.add_video(v)

    profile = discoverer.analyze()

    # Step 3: Output
    print("\n" + "=" * 60)
    print("DISCOVERED PREFERENCE PROFILE")
    print("=" * 60)

    print(f"\n📊 ANALYZED: {profile.n_videos_analyzed} videos")

    print(f"\n🎯 TOP FEATURES YOU PREFER:")
    for i, (feat, pct) in enumerate(profile.top_features, 1):
        level = "🔴 HIGH" if pct > 70 else "🟡 MEDIUM" if pct > 40 else "🟢 LOW"
        print(f"   {i}. {feat}: {pct:.0f}th percentile {level}")

    print(f"\n📁 YOUR PREFERENCE TYPES:")
    for t in profile.preference_types:
        bar = "█" * int(t["percentage"] * 20)
        print(f"   {t['name']:20s}: {bar:20s} {t['percentage']:.0%}")

    print(f"\n💡 DISCOVERY INSIGHTS:")
    for note in profile.discovery_notes:
        print(f"   • {note}")

    print(f"\n📝 SUMMARY:")
    print(f"   {profile.summary}")

    # Save - convert all numpy types to Python native
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, int):
            return int(obj)
        return obj

    output = {
        "user_id": profile.user_id,
        "n_videos": int(profile.n_videos_analyzed),
        "top_features": [{"feature": str(f), "percentile": float(p)} for f, p in profile.top_features],
        "types": convert_to_serializable(profile.preference_types),
        "notes": [str(n) for n in profile.discovery_notes],
        "summary": str(profile.summary)
    }

    output_path = "./data/preference_profile.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to: {output_path}")

    return profile


if __name__ == "__main__":
    main()
