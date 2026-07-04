"""
Frame Sampler Module

Fast frame sampling for video frame extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.frame_extraction.adaptive_threshold import AdaptiveVideoThreshold
@dataclass(slots=True)
class FrameInfo:
    """Information about a sampled frame."""

    frame_index: int
    timestamp: float
    quality_score: float = 0.0


class FrameSampler:
    """Fast frame sampler."""

    def __init__(
        self,
        strategy: str = "combined",
        max_frames: int = 50,
        k: float = 1.5,
        warmup: int = 10,
    ):
        """
        Args:
            strategy:
                even | motion | quality | combined
            max_frames:
                Maximum frames returned.
            k:
                Adaptive threshold multiplier.
            warmup:
                Skip first N frames.
        """

        if strategy not in {
            "even",
            "motion",
            "quality",
            "combined",
        }:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self.max_frames = max_frames
        self.warmup = warmup

        self.threshold = AdaptiveVideoThreshold(
            k=k,
            method="median_iqr",
        )

    def sample(self, video_path: str) -> List[FrameInfo]:
        """
        Sample frames from a video.

        Returns:
            List[FrameInfo]
        """

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_gray = None

        scores = []

        for frame_index in range(total_frames):

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index < self.warmup:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            quality = self._quality_score(gray)

            if prev_gray is None:
                motion = 0.0
            else:
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.mean(diff) / 255.0

            prev_gray = gray

            if self.strategy == "quality":
                score = quality

            elif self.strategy == "motion":
                score = motion

            elif self.strategy == "even":
                score = 1.0

            else:
                score = quality * 0.6 + motion * 0.4

            scores.append(
                (
                    frame_index,
                    score,
                )
            )

        cap.release()

        if not scores:
            return []

        values = [s for _, s in scores]

        self.threshold.fit(values)

        threshold = self.threshold.get_threshold()

        selected = [
            item
            for item in scores
            if item[1] >= threshold
        ]

        if len(selected) > self.max_frames:
            selected = sorted(
                selected,
                key=lambda x: x[1],
                reverse=True,
            )[: self.max_frames]

        selected.sort(key=lambda x: x[0])

        return [
            FrameInfo(
                frame_index=index,
                timestamp=index / fps,
                quality_score=score,
            )
            for index, score in selected
        ]

    def _quality_score(self, gray: np.ndarray) -> float:
        """
        Calculate frame quality score.

        Returns:
            Score in range [0, 1]
        """

        # Sharpness
        blur = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
        blur = min(blur / 1000.0, 1.0)

        # Brightness
        brightness = gray.mean() / 255.0
        brightness = 1.0 - abs(brightness - 0.5) * 2.0

        # Contrast
        contrast = min(gray.std() / 64.0, 1.0)

        return (
            blur * 0.5
            + brightness * 0.3
            + contrast * 0.2
        )