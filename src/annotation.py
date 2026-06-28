"""
Annotation Module - Human Preference Annotation

Protocols:
- Attribute-level ratings (face, hair, outfit, expression)
- Preference level (ordinal 1-5)
- Context annotation
"""

import csv
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class PreferenceAnnotation:
    """Single image preference annotation."""
    user_id: str
    image_id: str
    timestamp: str
    context: str = ""

    # Overall preference
    preference_level: int = 3  # 1-5 ordinal scale
    overall_score: int = 3    # 1-5 overall attractiveness

    # Attribute scores (1-5 each)
    face_score: int = 0
    hair_score: int = 0
    outfit_score: int = 0
    expression_score: int = 0

    # Optional text comment
    comment: str = ""

    # Quality flags
    annotator_flag: str = "pass"  # 'pass', 'check', 'fail'
    seen_before: bool = False
    fan_of_subject: bool = False

    # Context scores (optional)
    trend_score: float = 0.0  # 0-1
    celebrity_score: float = 0.0  # 0-1
    capture_quality: int = 3  # 1-5


@dataclass
class AnnotationCollection:
    """Collection of annotations with statistics."""
    annotations: list[PreferenceAnnotation] = field(default_factory=list)

    def add(self, annotation: PreferenceAnnotation):
        """Add annotation to collection."""
        self.annotations.append(annotation)

    def filter_by_user(self, user_id: str) -> "AnnotationCollection":
        """Get annotations for specific user."""
        filtered = AnnotationCollection()
        filtered.annotations = [
            a for a in self.annotations if a.user_id == user_id
        ]
        return filtered

    def filter_by_context(self, context: str) -> "AnnotationCollection":
        """Get annotations for specific context."""
        filtered = AnnotationCollection()
        filtered.annotations = [
            a for a in self.annotations if a.context == context
        ]
        return filtered

    def get_preference_levels(self) -> dict[str, list[int]]:
        """Get preference levels per user."""
        levels = {}
        for ann in self.annotations:
            if ann.user_id not in levels:
                levels[ann.user_id] = []
            levels[ann.user_id].append(ann.preference_level)
        return levels

    def compute_attribute_importance(self) -> dict[str, float]:
        """Compute correlation between attributes and overall score."""
        if not self.annotations:
            return {}

        import numpy as np
        from scipy.stats import pearsonr

        attributes = ["face_score", "hair_score", "outfit_score", "expression_score"]
        importance = {}

        for attr in attributes:
            values = [getattr(a, attr) for a in self.annotations]
            overall = [a.overall_score for a in self.annotations]

            # Filter out zeros (missing)
            mask = [(v > 0 and o > 0) for v, o in zip(values, overall)]
            if sum(mask) < 10:
                continue

            valid_values = [v for v, m in zip(values, mask) if m]
            valid_overall = [o for o, m in zip(overall, mask) if m]

            try:
                corr, p = pearsonr(valid_values, valid_overall)
                importance[attr] = corr
            except:
                importance[attr] = 0.0

        return importance

    def to_csv(self, filepath: str):
        """Save annotations to CSV."""
        if not self.annotations:
            return

        fieldnames = [
            "user_id", "image_id", "timestamp", "context",
            "preference_level", "overall_score",
            "face_score", "hair_score", "outfit_score", "expression_score",
            "comment", "annotator_flag", "seen_before", "fan_of_subject",
            "trend_score", "celebrity_score", "capture_quality"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ann in self.annotations:
                row = asdict(ann)
                row["seen_before"] = "true" if ann.seen_before else "false"
                row["fan_of_subject"] = "true" if ann.fan_of_subject else "false"
                writer.writerow(row)

    @classmethod
    def from_csv(cls, filepath: str) -> "AnnotationCollection":
        """Load annotations from CSV."""
        collection = cls()

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ann = PreferenceAnnotation(
                    user_id=row["user_id"],
                    image_id=row["image_id"],
                    timestamp=row["timestamp"],
                    context=row.get("context", ""),
                    preference_level=int(row.get("preference_level", 3)),
                    overall_score=int(row.get("overall_score", 3)),
                    face_score=int(row.get("face_score", 0)),
                    hair_score=int(row.get("hair_score", 0)),
                    outfit_score=int(row.get("outfit_score", 0)),
                    expression_score=int(row.get("expression_score", 0)),
                    comment=row.get("comment", ""),
                    annotator_flag=row.get("annotator_flag", "pass"),
                    seen_before=row.get("seen_before", "false").lower() == "true",
                    fan_of_subject=row.get("fan_of_subject", "false").lower() == "true",
                    trend_score=float(row.get("trend_score", 0.0)),
                    celebrity_score=float(row.get("celebrity_score", 0.0)),
                    capture_quality=int(row.get("capture_quality", 3))
                )
                collection.add(ann)

        return collection


class AnnotationProtocol:
    """Protocol for collecting human preference annotations."""

    def __init__(
        self,
        min_annotations_per_user: int = 30,
        inter_rater_threshold: float = 0.6,
        gold_check_count: int = 5
    ):
        """
        Args:
            min_annotations_per_user: Minimum annotations to include user
            inter_rater_threshold: Krippendorff's alpha threshold
            gold_check_count: Number of gold-standard images per batch
        """
        self.min_annotations = min_annotations_per_user
        self.inter_rater_threshold = inter_rater_threshold
        self.gold_check_count = gold_check_count

    def validate_annotation_quality(
        self,
        annotations: AnnotationCollection
    ) -> dict:
        """Validate quality of annotations.

        Returns:
            Quality report dict
        """
        users = set(a.user_id for a in annotations.annotations)

        quality_report = {
            "total_annotations": len(annotations.annotations),
            "total_users": len(users),
            "users_meeting_minimum": 0,
            "annotation_distribution": {},
            "recommendations": []
        }

        # Check per-user counts
        for user_id in users:
            user_anns = annotations.filter_by_user(user_id)
            count = len(user_anns.annotations)
            quality_report["annotation_distribution"][user_id] = count

            if count >= self.min_annotations:
                quality_report["users_meeting_minimum"] += 1

        # Quality recommendations
        if quality_report["users_meeting_minimum"] < len(users) * 0.8:
            quality_report["recommendations"].append(
                f"Only {quality_report['users_meeting_minimum']}/{len(users)} "
                f"users meet minimum annotation threshold of {self.min_annotations}"
            )

        if len(annotations.annotations) < 200:
            quality_report["recommendations"].append(
                "Consider collecting at least 200 total annotations for stable estimates"
            )

        return quality_report

    def create_annotation_ui_spec(self) -> dict:
        """Generate UI specification for annotation collection.

        Returns:
            UI spec dict
        """
        return {
            "type": "annotation_ui",
            "version": "1.0",
            "components": [
                {
                    "id": "image_display",
                    "type": "image",
                    "description": "Single image to annotate"
                },
                {
                    "id": "preference_level",
                    "type": "ordinal_scale",
                    "label": "How much do you like this image?",
                    "options": [
                        {"value": 1, "label": "Strong Dislike"},
                        {"value": 2, "label": "Weak Dislike"},
                        {"value": 3, "label": "Neutral"},
                        {"value": 4, "label": "Weak Like"},
                        {"value": 5, "label": "Strong Like"}
                    ]
                },
                {
                    "id": "face_score",
                    "type": "slider",
                    "label": "Face attractiveness",
                    "min": 1, "max": 5
                },
                {
                    "id": "hair_score",
                    "type": "slider",
                    "label": "Hair attractiveness",
                    "min": 1, "max": 5
                },
                {
                    "id": "outfit_score",
                    "type": "slider",
                    "label": "Outfit attractiveness",
                    "min": 1, "max": 5
                },
                {
                    "id": "expression_score",
                    "type": "slider",
                    "label": "Expression / vibe",
                    "min": 1, "max": 5
                },
                {
                    "id": "comment",
                    "type": "text",
                    "label": "Why did you feel this way? (optional)",
                    "max_length": 500
                }
            ],
            "context_questions": [
                {
                    "id": "seen_before",
                    "type": "boolean",
                    "label": "Have you seen this image before?"
                },
                {
                    "id": "fan_of_subject",
                    "type": "boolean",
                    "label": "Are you a fan of the person in this image?"
                }
            ]
        }


class PositiveUnlabeledAdapter:
    """Adapter for Positive-Unlabeled learning settings.

    Converts annotation collection to pairwise comparisons for PU learning.
    """

    def __init__(self):
        pass

    def to_pairwise(
        self,
        annotations: AnnotationCollection,
        user_id: str
    ) -> list[tuple[str, str, Literal["A", "B"]]]:
        """Convert ordinal annotations to pairwise comparisons.

        Args:
            annotations: Annotation collection
            user_id: User to extract pairs for

        Returns:
            List of (image_A, image_B, winner) where winner is the preferred one
        """
        user_anns = annotations.filter_by_user(user_id)
        items = [(a.image_id, a.preference_level) for a in user_anns.annotations]

        pairs = []

        # Create pairs from ordinal comparisons
        for i, (img_a, level_a) in enumerate(items):
            for j, (img_b, level_b) in enumerate(items):
                if i >= j:
                    continue

                if level_a > level_b:
                    pairs.append((img_a, img_b, "A"))
                elif level_b > level_a:
                    pairs.append((img_a, img_b, "B"))
                # Skip equal levels

        return pairs

    def get_grade_bins(
        self,
        annotations: AnnotationCollection,
        user_id: str
    ) -> dict[int, list[str]]:
        """Group images by preference grade.

        Returns:
            Dict mapping grade (1-5) to list of image IDs
        """
        user_anns = annotations.filter_by_user(user_id)

        bins = {1: [], 2: [], 3: [], 4: [], 5: []}

        for ann in user_anns.annotations:
            grade = ann.preference_level
            if grade in bins:
                bins[grade].append(ann.image_id)

        return bins

    def compute_pu_metrics(
        self,
        annotations: AnnotationCollection
    ) -> dict:
        """Compute metrics relevant to PU learning.

        Returns:
            Dict with PU-specific metrics
        """
        import numpy as np

        # Distribution of preference levels
        levels = [a.preference_level for a in annotations.annotations]

        bins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for level in levels:
            if level in bins:
                bins[level] += 1

        total = len(levels)

        return {
            "n_annotations": total,
            "positive_rate": (bins[4] + bins[5]) / total if total > 0 else 0,
            "negative_rate": (bins[1] + bins[2]) / total if total > 0 else 0,
            "neutral_rate": bins[3] / total if total > 0 else 0,
            "grade_distribution": bins,
            "pu_note": (
                "High neutral rate suggests PU learning approach may be beneficial. "
                "Consider treating level 1-2 as negatives, 4-5 as positives, "
                "and 3 as unlabeled for PU risk estimation."
            )
        }
