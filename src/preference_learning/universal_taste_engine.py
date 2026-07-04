"""
Universal Taste Engine - Xử lý khi user thích TẤT CẢ mọi người

4 Approach:
1. Negative Sampling - Chọn người KHÔNG phải gu
2. Trait-based - Đánh giá traits thay vì người
3. Diversity-based - Recommend đa dạng nhất
4. Universal - Uniform distribution (thật sự universal taste)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass, field
from collections import Counter
import random


# =============================================================================
# TRAIT DEFINITIONS
# =============================================================================

TRAITS = {
    # Appearance
    "youthful": {"name": "Youthful", "category": "appearance"},
    "mature": {"name": "Mature", "category": "appearance"},
    "baby_face": {"name": "Baby Face", "category": "appearance"},
    "soft_features": {"name": "Soft Features", "category": "appearance"},
    "sharp_features": {"name": "Sharp Features", "category": "appearance"},
    "pale_skin": {"name": "Pale Skin", "category": "appearance"},
    "tan_skin": {"name": "Tan Skin", "category": "appearance"},
    "petite": {"name": "Petite", "category": "appearance"},
    "voluptuous": {"name": "Voluptuous", "category": "appearance"},

    # Expression
    "smiling": {"name": "Smiling", "category": "expression"},
    "serious": {"name": "Serious", "category": "expression"},
    "mysterious": {"name": "Mysterious", "category": "expression"},
    "innocent": {"name": "Innocent", "category": "expression"},
    "confident": {"name": "Confident", "category": "expression"},
    "warm": {"name": "Warm", "category": "expression"},
    "cool": {"name": "Cool", "category": "expression"},

    # Style
    "natural": {"name": "Natural", "category": "style"},
    "glamorous": {"name": "Glamorous", "category": "style"},
    "elegant": {"name": "Elegant", "category": "style"},
    "cute": {"name": "Cute", "category": "style"},
    "sexy": {"name": "Sexy", "category": "style"},
    "sporty": {"name": "Sporty", "category": "style"},

    # Ethnicity/Region (for clustering)
    "east_asian": {"name": "East Asian (JP/KR/CN)", "category": "region"},
    "southeast_asian": {"name": "SE Asian", "category": "region"},
    "western": {"name": "Western", "category": "region"},
    "latina": {"name": "Latina", "category": "region"},
    "middle_eastern": {"name": "Middle Eastern", "category": "region"},
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class NegativeSample:
    """Người user chọn là KHÔNG phải gu."""
    identity: str
    reason: str = ""
    timestamp: float = 0.0


@dataclass
class TraitRating:
    """Rating cho một trait (1-5 sao)."""
    trait: str
    rating: int  # 1-5
    examples: List[str] = field(default_factory=list)  # Người minh họa trait này


@dataclass
class DiversityBucket:
    """Một nhóm đa dạng."""
    name: str
    description: str
    members: List[str]
    region: str


@dataclass
class UniversalTasteProfile:
    """Profile khi user thích TẤT CẢ."""
    user_id: str

    # Approach 1: Negative sampling
    negatives: List[str] = field(default_factory=list)  # Người KHÔNG thích
    negative_regions: Dict[str, int] = field(default_factory=dict)

    # Approach 2: Trait preferences
    trait_ratings: Dict[str, int] = field(default_factory=dict)  # trait -> rating

    # Approach 3: Diversity config
    prefer_diversity: bool = True
    diversity_weights: Dict[str, float] = field(default_factory=dict)

    # Approach 4: Universality score
    universality_score: float = 1.0  # 1.0 = hoàn toàn universal

    # Final recommendation strategy
    recommendation_strategy: str = "diversity"  # negative/trait/diversity/universal


# =============================================================================
# APPROACH 1: NEGATIVE SAMPLING
# =============================================================================

class NegativeSamplingEngine:
    """
    Thay vì chọn người HAY HƠN, user chọn người KHÔNG PHẢI GU.

    Workflow:
    1. User chọn vài người họ KHÔNG thích
    2. Model học: "Những người này ≠ user's taste"
    3. Retrieval: Recommend NHỮNG NGƯỜI KHÁC những người bị loại
    """

    def __init__(self, identity_groups: Dict[str, List[str]]):
        self.identity_groups = identity_groups
        self.negatives: List[str] = []

    def add_negative(self, identity: str, reason: str = "") -> NegativeSample:
        """Thêm một negative sample."""
        neg = NegativeSample(identity=identity, reason=reason)
        self.negatives.append(neg)
        return neg

    def add_negatives_batch(self, identities: List[str]) -> List[NegativeSample]:
        """Thêm nhiều negative samples cùng lúc."""
        return [self.add_negative(ident) for ident in identities]

    def get_recommendations(
        self,
        embedding_dict: Dict[str, np.ndarray],
        identity_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations that are DIFFERENT from negatives.

        Args:
            embedding_dict: image_id -> embedding
            identity_embeddings: identity -> avg embedding
            top_k: Số lượng recommendations

        Returns:
            List of (identity, score) recommended
        """
        if not self.negatives:
            # No negatives = recommend random
            identities = list(identity_embeddings.keys())
            return [(i, 1.0/len(identities)) for i in random.sample(identities, min(top_k, len(identities)))]

        # Compute negative centroid
        neg_embs = []
        for neg in self.negatives:
            if neg.identity in identity_embeddings:
                neg_embs.append(identity_embeddings[neg.identity])

        if not neg_embs:
            return []

        negative_centroid = np.mean(neg_embs, axis=0)

        # Score each identity by DISTANCE from negative centroid
        # Higher distance = more different from what user doesn't like
        scores = []
        for identity, emb in identity_embeddings.items():
            # Cosine distance from negative centroid
            similarity = np.dot(emb, negative_centroid)
            distance_score = 1 - similarity  # Higher = more different
            scores.append((identity, float(distance_score)))

        # Sort by distance (most different first)
        scores.sort(key=lambda x: -x[1])

        return scores[:top_k]

    def get_stats(self) -> Dict:
        """Get statistics về negatives."""
        return {
            "n_negatives": len(self.negatives),
            "negatives": [n.identity for n in self.negatives],
            "reason_summary": Counter([n.reason for n in self.negatives if n.reason])
        }


# =============================================================================
# APPROACH 2: TRAIT-BASED PREFERENCE
# =============================================================================

class TraitBasedEngine:
    """
    User đánh giá TRAITS thay vì so sánh người.

    Ví dụ:
    - Youthful: ★★★★☆ (4/5)
    - Soft features: ★★★☆☆ (3/5)
    - Confident: ★★★☆☆ (3/5)

    Model: Preference = weighted sum của traits
    """

    def __init__(self):
        self.trait_ratings: Dict[str, int] = {}  # trait -> rating (1-5)
        self.trait_examples: Dict[str, List[str]] = {}  # trait -> [identities]

    def rate_trait(self, trait: str, rating: int, examples: List[str] = None) -> TraitRating:
        """Đánh giá một trait."""
        rating = max(1, min(5, rating))  # Clamp to 1-5
        self.trait_ratings[trait] = rating

        if examples:
            if trait not in self.trait_examples:
                self.trait_examples[trait] = []
            self.trait_examples[trait].extend(examples)

        return TraitRating(trait=trait, rating=rating, examples=examples or [])

    def get_preference_vector(self) -> np.ndarray:
        """Convert trait ratings to preference vector."""
        traits_list = list(TRAITS.keys())
        vector = np.zeros(len(traits_list))

        for i, trait in enumerate(traits_list):
            if trait in self.trait_ratings:
                vector[i] = self.trait_ratings[trait] / 5.0  # Normalize to 0-1

        return vector

    def get_top_traits(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get top N traits the user likes most."""
        sorted_traits = sorted(
            self.trait_ratings.items(),
            key=lambda x: -x[1]
        )
        return sorted_traits[:n]

    def get_low_traits(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get bottom N traits (what user doesn't like)."""
        sorted_traits = sorted(
            self.trait_ratings.items(),
            key=lambda x: x[1]
        )
        return sorted_traits[:n]

    def recommend_by_traits(
        self,
        identity_traits: Dict[str, Dict[str, float]],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Recommend identities based on trait match.

        Args:
            identity_traits: {identity: {trait: score_0_to_1}}

        Returns:
            List of (identity, match_score)
        """
        pref_vec = self.get_preference_vector()
        traits_list = list(TRAITS.keys())

        scores = []
        for identity, traits_dict in identity_traits.items():
            # Compute match score
            total_score = 0.0
            n_rated = 0

            for i, trait in enumerate(traits_list):
                if trait in self.trait_ratings and trait in traits_dict:
                    # Weighted by user rating
                    weight = self.trait_ratings[trait] / 5.0
                    total_score += weight * traits_dict[trait]
                    n_rated += 1

            if n_rated > 0:
                avg_score = total_score / n_rated
            else:
                avg_score = 0.5  # Neutral

            scores.append((identity, avg_score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def get_preference_profile(self) -> Dict:
        """Get summary of trait preferences."""
        top = self.get_top_traits(5)
        low = self.get_low_traits(5)

        return {
            "top_traits": [{"trait": t, "name": TRAITS[t]["name"], "rating": r} for t, r in top],
            "low_traits": [{"trait": t, "name": TRAITS[t]["name"], "rating": r} for t, r in low],
            "total_rated": len(self.trait_ratings),
            "profile_summary": self._generate_summary()
        }

    def _generate_summary(self) -> str:
        """Generate text summary of preferences."""
        if not self.trait_ratings:
            return "Chưa đánh giá traits nào."

        top3 = self.get_top_traits(3)
        low3 = self.get_low_traits(3)

        likes = ", ".join([TRAITS[t]["name"] for t, _ in top3])
        dislikes = ", ".join([TRAITS[t]["name"] for t, _ in low3])

        return f"Thích: {likes}. Không thích: {dislikes}."


# =============================================================================
# APPROACH 3: DIVERSITY-BASED RECOMMENDATION
# =============================================================================

class DiversityEngine:
    """
    Khi user thích TẤT CẢ, recommend DIVERSE (đa dạng nhất).

    Strategy: Mỗi bucket lấy 1-2 người đại diện
    - 1 East Asian (Japanese/Korean)
    - 1 Western
    - 1 Latina
    - 1 Southeast Asian
    - ...
    """

    REGION_BUCKETS = {
        "east_asian": {
            "name": "Đông Á",
            "description": "Nhật Bản, Hàn Quốc, Trung Quốc",
            "keywords": ["japanese", "korean", "chinese", "hamabe", "aoi", "arimura",
                        "asuka", "anya", "bae", "kim", "han"]
        },
        "western": {
            "name": "Phương Tây",
            "description": "Mỹ, Châu Âu",
            "keywords": ["anne", "alexandra", "blake", "lily", "kate", "emily",
                        "margot", "natalie", "scarlett", "jennifer"]
        },
        "southeast_asian": {
            "name": "Đông Nam Á",
            "description": "Việt Nam, Thái Lan, Philippines",
            "keywords": ["davika", "lam duan", "mai", "can", "bao thuong"]
        },
        "latina": {
            "name": "Latin",
            "description": "Brazil, Mexico, Tây Ban Nha",
            "keywords": ["ana de armas", "eiza", "barbie", "sofia", "pene"]
        },
        "slavic": {
            "name": "Đông Âu/Nga",
            "description": "Nga, Ukraine, Eastern Europe",
            "keywords": ["elena", "mila", "irina", "alesandrova", "anna chipovskaya"]
        },
        "middle_eastern": {
            "name": "Trung Đông",
            "description": "Lebanon, Iran, Ả Rập",
            "keywords": ["mona", "nancy", "yara"]
        }
    }

    def __init__(self):
        self.identities: List[str] = []
        self.identity_embeddings: Dict[str, np.ndarray] = {}

    def load_identities(self, identities: List[str]):
        """Load danh sách identities."""
        self.identities = identities

    def load_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Load embeddings."""
        self.identity_embeddings = embeddings

    def assign_region(self, identity: str) -> str:
        """Tự động assign region dựa trên tên."""
        identity_lower = identity.lower()

        for region, info in self.REGION_BUCKETS.items():
            for keyword in info["keywords"]:
                if keyword in identity_lower:
                    return region

        # Default based on common patterns
        if any(c in identity_lower for c in ['japanese', 'aoi', 'aragaki', 'hamabe', 'asuka']):
            return "east_asian"
        elif any(c in identity_lower for c in ['korean', 'bae', 'han', 'kim', 'jessica']):
            return "east_asian"
        elif any(c in identity_lower for c in ['anne', 'blake', 'margot', 'natalie', 'scarlett']):
            return "western"

        return "western"  # Default fallback

    def bucket_identities(self) -> Dict[str, DiversityBucket]:
        """Phân chia identities vào buckets."""
        buckets = {}

        for identity in self.identities:
            region = self.assign_region(identity)

            if region not in buckets:
                buckets[region] = DiversityBucket(
                    name=self.REGION_BUCKETS[region]["name"],
                    description=self.REGION_BUCKETS[region]["description"],
                    members=[],
                    region=region
                )

            buckets[region].members.append(identity)

        return buckets

    def get_diverse_recommendations(
        self,
        n_per_bucket: int = 1,
        use_embedding_order: bool = True
    ) -> List[Dict]:
        """
        Get diverse recommendations (1-2 from each region).

        Returns:
            List of recommendation dicts with identity và region info
        """
        buckets = self.bucket_identities()

        recommendations = []

        for region, bucket in buckets.items():
            if not bucket.members:
                continue

            if use_embedding_order and bucket.members[0] in self.identity_embeddings:
                # Sort by embedding magnitude as proxy for "prototypical"
                sorted_members = sorted(
                    bucket.members,
                    key=lambda x: np.linalg.norm(self.identity_embeddings.get(x, np.zeros(768)))
                )
            else:
                sorted_members = bucket.members

            # Take n_per_bucket from each bucket
            selected = sorted_members[:n_per_bucket]

            for ident in selected:
                recommendations.append({
                    "identity": ident,
                    "region": region,
                    "region_name": bucket.name,
                    "description": bucket.description,
                    "bucket_description": f"1 {bucket.name} đại diện"
                })

        return recommendations

    def get_diversity_matrix(self) -> Dict:
        """Get diversity coverage matrix."""
        buckets = self.bucket_identities()

        return {
            region: {
                "name": info["name"],
                "count": len(bucket.members),
                "members": bucket.members[:3]  # Sample
            }
            for region, (bucket, info) in [(r, (b, self.REGION_BUCKETS[r]))
                                            for r, b in buckets.items()]
        }


# =============================================================================
# APPROACH 4: UNIVERSAL PREFERENCE
# =============================================================================

class UniversalPreferenceEngine:
    """
    Khi user THẬT SỰ thích TẤT CẢ đều như nhau.

    Model: Uniform distribution
    - Mỗi người có xác suất = 1/N
    - Recommend random (có seed để reproducible)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        self.identities: List[str] = []
        self.n_recommendations = 0

    def load_identities(self, identities: List[str]):
        """Load danh sách identities."""
        self.identities = identities

    def get_uniform_recommendation(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get uniform random recommendations.

        Returns:
            List of (identity, probability) - all equal probability
        """
        n = len(self.identities)
        if n == 0:
            return []

        prob = 1.0 / n
        self.n_recommendations += 1

        # Shuffle with seed for reproducibility
        shuffled = self.identities.copy()
        self.random.seed(self.seed + self.n_recommendations)
        self.random.shuffle(shuffled)

        return [(ident, prob) for ident in shuffled[:top_k]]

    def get_underrepresented_recommendation(
        self,
        popularity_scores: Dict[str, float],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Recommend những người ÍT được recommend nhất.
        (Inverse popularity)
        """
        if not popularity_scores:
            return self.get_uniform_recommendation(top_k)

        # Inverse popularity
        inverse_scores = {}
        for ident, pop in popularity_scores.items():
            inverse_scores[ident] = 1.0 / (pop + 0.1)  # Add smoothing

        # Normalize
        total = sum(inverse_scores.values())
        for ident in inverse_scores:
            inverse_scores[ident] /= total

        # Sort by inverse score (least popular first)
        sorted_ids = sorted(inverse_scores.items(), key=lambda x: -x[1])

        return sorted_ids[:top_k]

    def compute_universality_score(
        self,
        pairwise_data: List[Tuple[str, str, str]]
    ) -> float:
        """
        Compute universality score từ pairwise data.

        Nếu tất cả pairs đều là "equal" hoặc "both good":
        - Universality score cao (0.8-1.0)

        Nếu có clear preferences:
        - Universality score thấp (< 0.5)
        """
        if not pairwise_data:
            return 1.0  # Default = universal

        equal_count = sum(1 for _, _, result in pairwise_data if result == "equal")
        total = len(pairwise_data)

        if total == 0:
            return 1.0

        return equal_count / total

    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            "n_identities": len(self.identities),
            "uniform_probability": 1.0 / max(len(self.identities), 1),
            "n_recommendations_made": self.n_recommendations,
            "strategy": "uniform_random"
        }


# =============================================================================
# MAIN: UNIVERSAL TASTE ORCHESTRATOR
# =============================================================================

class UniversalTasteEngine:
    """
    Orchestrator cho cả 4 approaches.

    Workflow:
    1. User chọn: "Tôi thích TẤT CẢ"
    2. System hỏi: "Bạn muốn đa dạng hay tìm pattern ẩn?"
    3. Tùy theo câu trả lời, sử dụng approach phù hợp
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id = user_id
        self.profile = UniversalTasteProfile(user_id=user_id)

        # Initialize all engines
        self.negative_engine = None
        self.trait_engine = TraitBasedEngine()
        self.diversity_engine = DiversityEngine()
        self.universal_engine = UniversalPreferenceEngine()

        # State
        self.identities: List[str] = []
        self.identity_embeddings: Dict[str, np.ndarray] = {}
        self.identity_traits: Dict[str, Dict[str, float]] = {}

        # Recommendation history
        self.recommendation_history: List[Dict] = []

    def load_dataset(
        self,
        identities: List[str],
        embeddings: Dict[str, np.ndarray] = None,
        traits: Dict[str, Dict[str, float]] = None
    ):
        """Load dataset vào engine."""
        self.identities = identities

        if embeddings:
            # Create identity-level embeddings (average of frames)
            self.identity_embeddings = self._aggregate_embeddings(embeddings)

        if traits:
            self.identity_traits = traits

        # Initialize negative engine
        identity_to_frames = {}
        for ident in identities:
            identity_to_frames[ident] = [f"{ident}_{i}.jpg" for i in range(5)]

        self.negative_engine = NegativeSamplingEngine(identity_to_frames)
        self.diversity_engine.load_identities(identities)
        self.diversity_engine.load_embeddings(self.identity_embeddings)
        self.universal_engine.load_identities(identities)

    def _aggregate_embeddings(self, frame_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Aggregate frame embeddings to identity level."""
        identity_embs = {}

        for frame_id, emb in frame_embeddings.items():
            # Extract identity from frame_id: "Hamabe Minami_0.jpg" -> "Hamabe Minami"
            identity = frame_id.rsplit("_", 1)[0]
            if identity not in identity_embs:
                identity_embs[identity] = []
            identity_embs[identity].append(emb)

        # Average
        for identity in identity_embs:
            identity_embs[identity] = np.mean(identity_embs[identity], axis=0)

        return identity_embs

    # -------------------------------------------------------------------------
    # APPROACH 1: NEGATIVE SAMPLING
    # -------------------------------------------------------------------------

    def add_negative_preference(self, identity: str, reason: str = "") -> NegativeSample:
        """Thêm người KHÔNG phải gu."""
        self.profile.negatives.append(identity)
        return self.negative_engine.add_negative(identity, reason)

    def get_negative_recommendations(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Recommend những người KHÁC người bị negative."""
        return self.negative_engine.get_recommendations(
            embedding_dict={},  # Not needed for identity-level
            identity_embeddings=self.identity_embeddings,
            top_k=top_k
        )

    # -------------------------------------------------------------------------
    # APPROACH 2: TRAIT-BASED
    # -------------------------------------------------------------------------

    def rate_trait(self, trait: str, rating: int, examples: List[str] = None) -> TraitRating:
        """Đánh giá một trait."""
        return self.trait_engine.rate_trait(trait, rating, examples)

    def get_trait_profile(self) -> Dict:
        """Get trait preference profile."""
        return self.trait_engine.get_preference_profile()

    def get_trait_recommendations(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Recommend based on trait match."""
        return self.trait_engine.recommend_by_traits(self.identity_traits, top_k)

    # -------------------------------------------------------------------------
    # APPROACH 3: DIVERSITY-BASED
    # -------------------------------------------------------------------------

    def get_diverse_recommendations(self, n_per_bucket: int = 1) -> List[Dict]:
        """Get diverse recommendations (1 from each region)."""
        return self.diversity_engine.get_diverse_recommendations(n_per_bucket)

    def get_diversity_coverage(self) -> Dict:
        """Get diversity coverage matrix."""
        return self.diversity_engine.get_diversity_matrix()

    # -------------------------------------------------------------------------
    # APPROACH 4: UNIVERSAL
    # -------------------------------------------------------------------------

    def get_universal_recommendations(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get random uniform recommendations."""
        return self.universal_engine.get_uniform_recommendation(top_k)

    # -------------------------------------------------------------------------
    # MASTER: GET FINAL RECOMMENDATIONS
    # -------------------------------------------------------------------------

    def get_recommendations(
        self,
        strategy: Literal["negative", "trait", "diversity", "universal"] = "diversity",
        top_k: int = 10
    ) -> List:
        """
        Get recommendations theo strategy.

        Args:
            strategy: Recommendation strategy
            top_k: Number of recommendations

        Returns:
            List of recommendations
        """
        self.profile.recommendation_strategy = strategy

        if strategy == "negative":
            recs = self.get_negative_recommendations(top_k)
            return [{"identity": ident, "score": score, "strategy": "negative"} for ident, score in recs]

        elif strategy == "trait":
            recs = self.get_trait_recommendations(top_k)
            return [{"identity": ident, "score": score, "strategy": "trait"} for ident, score in recs]

        elif strategy == "diversity":
            recs = self.get_diverse_recommendations(n_per_bucket=1)
            return recs

        elif strategy == "universal":
            recs = self.get_universal_recommendations(top_k)
            return [{"identity": ident, "probability": prob, "strategy": "universal"} for ident, prob in recs]

        else:
            # Default to diversity
            return self.get_diverse_recommendations(n_per_bucket=1)

    def get_profile_summary(self) -> Dict:
        """Get complete profile summary."""
        return {
            "user_id": self.user_id,
            "strategy": self.profile.recommendation_strategy,
            "negatives": self.profile.negatives,
            "traits": self.trait_engine.get_preference_profile(),
            "diversity_coverage": self.get_diversity_coverage(),
            "universal_stats": self.universal_engine.get_stats()
        }


# =============================================================================
# DEMO / CLI
# =============================================================================

def demo():
    """Demo tất cả 4 approaches."""
    print("=" * 70)
    print("UNIVERSAL TASTE ENGINE - DEMO")
    print("=" * 70)

    # Sample identities
    sample_identities = [
        "Hamabe Minami", "Aoi Yu", "Anya Taylor Joy", "Anne Hathaway",
        "Alexandra Daddario", "Ana De Armas", "Blake Lively", "Bae Suzy",
        "Lily Collins", "Margot Robbie"
    ]

    # Initialize engine
    engine = UniversalTasteEngine(user_id="demo_user")
    engine.load_dataset(sample_identities)

    # Approach 1: Negative Sampling
    print("\n" + "=" * 70)
    print("APPROACH 1: NEGATIVE SAMPLING")
    print("User chọn: Anne Hathaway, Blake Lively KHÔNG phải gu")
    print("=" * 70)

    engine.add_negative_preference("Anne Hathaway", reason="Too western")
    engine.add_negative_preference("Blake Lively", reason="Too tall")

    neg_recs = engine.get_negative_recommendations(top_k=5)
    print("\nRecommendations (những người KHÁC Anne Hathaway, Blake Lively):")
    for ident, score in neg_recs:
        print(f"  {ident}: {score:.3f}")

    # Approach 2: Trait-based
    print("\n" + "=" * 70)
    print("APPROACH 2: TRAIT-BASED")
    print("User đánh giá traits:")
    print("  Youthful: ★★★★☆ (4)")
    print("  Soft features: ★★★★★ (5)")
    print("  Mature: ★★☆☆☆ (2)")
    print("  Baby-faced: ★★★★☆ (4)")
    print("  East Asian look: ★★★★★ (5)")
    print("=" * 70)

    engine.rate_trait("youthful", 4)
    engine.rate_trait("soft_features", 5)
    engine.rate_trait("mature", 2)
    engine.rate_trait("baby_face", 4)
    engine.rate_trait("east_asian", 5)

    trait_profile = engine.get_trait_profile()
    print(f"\nTop traits: {[t['name'] for t in trait_profile['top_traits']]}")
    print(f"Summary: {trait_profile['profile_summary']}")

    # Approach 3: Diversity-based
    print("\n" + "=" * 70)
    print("APPROACH 3: DIVERSITY-BASED")
    print("=" * 70)

    diversity_recs = engine.get_diverse_recommendations(n_per_bucket=1)
    print("\nDiverse Recommendations (1 from each region):")
    for rec in diversity_recs:
        print(f"  {rec['identity']:20s} - {rec['region_name']:15s} - {rec['description']}")

    # Approach 4: Universal
    print("\n" + "=" * 70)
    print("APPROACH 4: UNIVERSAL")
    print("User thích TẤT CẢ đều như nhau (universal taste)")
    print("=" * 70)

    universal_recs = engine.get_universal_recommendations(top_k=5)
    print("\nUniform Random Recommendations:")
    for ident, prob in universal_recs:
        print(f"  {ident}: {prob:.3f} (equal probability)")

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    summary = engine.get_profile_summary()
    print(f"\nStrategy: {summary['strategy']}")
    print(f"Negatives: {summary['negatives']}")
    print(f"Top traits: {[t['name'] for t in summary['traits']['top_traits']]}")


if __name__ == "__main__":
    demo()
