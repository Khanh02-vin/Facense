"""
Quality Checker Module.

Gates frames on four criteria:
    1. Blur  — Laplacian variance (adaptive threshold)
    2. Lighting — brightness within a safe range
    3. Face    — face detection (MediaPipe or OpenCV Haar)
    4. Duplicate — histogram similarity vs. previously seen frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.frame_extraction.adaptive_threshold import AdaptiveVideoThreshold


@dataclass
class QualityConfig:
    """Thresholds and options used by every sub-checker."""

    min_blur_score: float = 0.10
    min_face_size_ratio: float = 0.02
    brightness_range: Tuple[float, float] = (0.2, 0.8)
    duplicate_threshold: float = 0.95
    face_detector: str = "mediapipe"
    use_adaptive_blur: bool = True
    blur_k: float = 1.5


@dataclass
class QualityResult:
    """Outcome of a single frame quality check."""

    passed: bool
    blur_score: float = 0.0
    brightness_score: float = 0.0
    has_face: bool = False
    face_size_ratio: float = 0.0
    face_count: int = 0
    is_duplicate: bool = False
    failed_checks: List[str] = field(default_factory=list)


class BlurChecker:
    """Blur detection using Laplacian variance."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        self.adaptive = AdaptiveVideoThreshold(k=config.blur_k, method="median_iqr")

    def score(self, frame: np.ndarray) -> float:
        """Return blur score in [0, 1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(variance / 500.0, 1.0)

    def check(
        self, frame: np.ndarray, history: Optional[List[float]] = None
    ) -> Tuple[float, bool]:
        """Return (blur_score, passed)."""
        blur_score = self.score(frame)

        if self.config.use_adaptive_blur and history:
            self.adaptive.fit(history)
            threshold = min(self.adaptive.get_threshold() / 500.0, 1.0)
        else:
            threshold = self.config.min_blur_score

        return blur_score, blur_score >= threshold


class LightingChecker:
    """Brightness range check."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def check(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Return (brightness_score, passed).

        Score = 1 when mean brightness is 0.5; lower at extremes.
        Passed when mean is inside ``brightness_range``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray) / 255.0
        score = float(np.clip(1.0 - abs(mean - 0.5) * 2, 0.0, 1.0))
        low, high = self.config.brightness_range
        return score, low <= mean <= high


class DuplicateChecker:
    """Detect near-identical frames via histogram correlation."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        self.histograms: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear history for a new video."""
        self.histograms.clear()

    @staticmethod
    def _histogram(gray: np.ndarray) -> np.ndarray:
        """Return a L1-normalised grayscale histogram (256 bins)."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def check(self, frame: np.ndarray) -> bool:
        """Return True if the frame is a duplicate of any previously seen frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = self._histogram(gray)

        for prev in self.histograms:
            similarity = cv2.compareHist(
                hist.astype(np.float32),
                prev.astype(np.float32),
                cv2.HISTCMP_CORREL,
            )
            if similarity >= self.config.duplicate_threshold:
                return True

        self.histograms.append(hist)
        return False


class FaceChecker:
    """Face detection via MediaPipe Tasks API (default) or OpenCV Haar cascade."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        self.detector = self._create_detector()

    def _create_detector(self) -> "cv2.CascadeClassifier | object":
        from mediapipe.tasks.python import vision

        if self.config.face_detector == "opencv":
            return cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # MediaPipe >= 0.10 — use Tasks Vision API
        from mediapipe.tasks.python import BaseOptions

        model_path = "/usr/local/share/mediapipe/face_detection_short_range.tflite"

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        return vision.FaceDetector.create_from_options(options)

    def check(self, frame: np.ndarray) -> Tuple[bool, float, int]:
        """
        Return (has_face, face_size_ratio, face_count).

        face_size_ratio is the area of the largest detected face
        relative to the frame area.
        """
        h, w = frame.shape[:2]

        if self.config.face_detector == "opencv":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                return False, 0.0, 0

            areas = [fw * fh for _, _, fw, fh in faces]
            ratio = max(areas) / (w * h)
            return ratio >= self.config.min_face_size_ratio, ratio, len(faces)

        import mediapipe as mp
        from mediapipe.tasks.python import vision

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.detections:
            return False, 0.0, 0

        ratios = []
        for det in result.detections:
            bbox = det.bounding_box
            ratios.append(
                (bbox.width * bbox.height) / (w * h)
            )

        ratio = max(ratios)
        return ratio >= self.config.min_face_size_ratio, ratio, len(ratios)


class QualityChecker:
    """Orchestrates all per-frame quality checks."""

    def __init__(self, config: Optional[QualityConfig] = None) -> None:
        self.config = config or QualityConfig()
        self.blur_checker = BlurChecker(self.config)
        self.light_checker = LightingChecker(self.config)
        self.face_checker = FaceChecker(self.config)
        self.duplicate_checker = DuplicateChecker(self.config)
        self.blur_history: List[float] = []

    def reset(self) -> None:
        """Reset state for a new video."""
        self.blur_history.clear()
        self.duplicate_checker.reset()

    def check_frame(self, frame: np.ndarray) -> QualityResult:
        """
        Evaluate a single frame against all quality gates.

        Returns:
            QualityResult with scores and pass/fail breakdown.
        """
        blur_score, blur_ok = self.blur_checker.check(frame, self.blur_history)
        brightness_score, light_ok = self.light_checker.check(frame)
        is_duplicate = self.duplicate_checker.check(frame)
        has_face, face_ratio, face_count = self.face_checker.check(frame)

        self.blur_history.append(blur_score)

        failed: List[str] = []
        if not blur_ok:
            failed.append("blur")
        if not light_ok:
            failed.append("lighting")
        if not has_face:
            failed.append("no_face")
        if is_duplicate:
            failed.append("duplicate")

        return QualityResult(
            passed=blur_ok and light_ok and has_face and not is_duplicate,
            blur_score=blur_score,
            brightness_score=brightness_score,
            has_face=has_face,
            face_size_ratio=face_ratio,
            face_count=face_count,
            is_duplicate=is_duplicate,
            failed_checks=failed,
        )

    def check_frames_batch(
        self,
        frames: List[np.ndarray],
        start_index: int = 0,
        fps: float = 30.0,
    ) -> List[QualityResult]:
        """
        Evaluate a list of frames sequentially.

        Args:
            frames:      List of BGR frames.
            start_index: Starting frame index (only used for future extensions).
            fps:         Video FPS (only used for future extensions).

        Returns:
            List of QualityResult, one per input frame.
        """
        return [self.check_frame(frame) for frame in frames]
