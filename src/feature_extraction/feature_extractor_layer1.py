"""
Layer 1 Feature Extractor - MVP

Basic objective metrics:
1. Motion Energy (MAD)
2. Blur Level (Variance of Laplacian)
3. Brightness (Mean pixel)
4. Face Visibility (Bounding box ratio)

Low-cost, high-value features.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import json


@dataclass
class VideoFeatures:
    """Features extracted from video/clip."""
    video_id: str

    # Motion features
    motion_energy: float = 0.0  # Mean frame difference
    motion_peak: float = 0.0  # Max frame difference
    motion_variance: float = 0.0  # Std of frame differences

    # Quality features
    blur_score: float = 0.0  # Variance of Laplacian
    brightness: float = 0.0  # Mean pixel value (0-255)
    brightness_std: float = 0.0

    # Face features
    face_visibility: float = 0.0  # Face area / frame area
    face_detected: bool = False
    face_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    # Derived
    n_frames: int = 0
    duration_seconds: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Feature vector for ML model."""
        return np.array([
            self.motion_energy,
            self.motion_peak,
            self.motion_variance,
            self.blur_score,
            self.brightness,
            self.brightness_std,
            self.face_visibility,
            1.0 if self.face_detected else 0.0,
        ])

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "motion_energy": float(self.motion_energy),
            "motion_peak": float(self.motion_peak),
            "blur_score": float(self.blur_score),
            "brightness": float(self.brightness),
            "face_visibility": float(self.face_visibility),
            "face_detected": bool(self.face_detected),
        }


class FeatureExtractorLayer1:
    """
    Extract Layer 1 (MVP) features from video/frames.

    Features:
    - Motion Energy: Mean Absolute Difference between frames
    - Blur Level: Variance of Laplacian
    - Brightness: Mean pixel intensity
    - Face Visibility: Face bbox ratio
    """

    def __init__(self):
        # Load face cascade (OpenCV built-in)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_from_video(self, video_path: str) -> VideoFeatures:
        """Extract features from video file."""
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        return self.extract_from_frames(frames, Path(video_path).stem)

    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        video_id: str = "unknown"
    ) -> VideoFeatures:
        """Extract features from list of frames."""
        if len(frames) < 2:
            return VideoFeatures(video_id=video_id, n_frames=len(frames))

        features = VideoFeatures(video_id=video_id)
        features.n_frames = len(frames)

        # Convert to grayscale for processing
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # 1. Motion Energy (MAD)
        motion_scores = self._compute_motion_energy(gray_frames)
        features.motion_energy = float(np.mean(motion_scores))
        features.motion_peak = float(np.max(motion_scores))
        features.motion_variance = float(np.std(motion_scores))

        # 2. Blur Level (Variance of Laplacian)
        blur_scores = [self._compute_blur(g) for g in gray_frames]
        features.blur_score = float(np.mean(blur_scores))

        # 3. Brightness
        brightness_values = [np.mean(g) for g in gray_frames]
        features.brightness = float(np.mean(brightness_values))
        features.brightness_std = float(np.std(brightness_values))

        # 4. Face Detection
        face_result = self._detect_face(frames)
        features.face_detected = face_result["detected"]
        features.face_visibility = face_result["visibility"]
        features.face_bbox = face_result["bbox"]

        return features

    def _compute_motion_energy(self, gray_frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute Motion Energy = Mean Absolute Difference between consecutive frames.

        M_t = (1/WH) * sum|x,y| |I_t(x,y) - I_{t-1}(x,y)|
        """
        motion_scores = []

        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            mad = np.mean(diff)
            motion_scores.append(mad)

        return np.array(motion_scores)

    def _compute_blur(self, gray_frame: np.ndarray) -> float:
        """
        Compute Blur Level = Variance of Laplacian.

        V_lap = (1/WH) * sum (L(x,y) - mean_L)^2

        Higher = sharper, Lower = blurrier
        """
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def _detect_face(self, frames: List[np.ndarray]) -> dict:
        """
        Detect face and compute visibility ratio.

        R_face = A_face / A_frame
        """
        if not frames:
            return {"detected": False, "visibility": 0.0, "bbox": (0, 0, 0, 0)}

        # Use middle frame for face detection
        mid_frame = frames[len(frames) // 2]
        gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return {"detected": False, "visibility": 0.0, "bbox": (0, 0, 0, 0)}

        # Use largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        # Compute visibility ratio
        frame_area = mid_frame.shape[0] * mid_frame.shape[1]
        face_area = w * h
        visibility = face_area / frame_area

        return {
            "detected": True,
            "visibility": float(visibility),
            "bbox": (int(x), int(y), int(w), int(h))
        }

    def extract_batch(
        self,
        video_paths: List[str]
    ) -> List[VideoFeatures]:
        """Extract features from multiple videos."""
        results = []
        for path in video_paths:
            try:
                features = self.extract_from_video(path)
                results.append(features)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return results


class FeatureVisualizer:
    """Visualize extracted features."""

    @staticmethod
    def print_features(features: VideoFeatures):
        """Print features in readable format."""
        print(f"\n{'='*50}")
        print(f"FEATURES: {features.video_id}")
        print(f"{'='*50}")

        print("\n[MOTION]")
        print(f"  Energy:     {features.motion_energy:.2f}")
        print(f"  Peak:       {features.motion_peak:.2f}")
        print(f"  Variance:   {features.motion_variance:.2f}")

        print("\n[QUALITY]")
        print(f"  Blur:       {features.blur_score:.2f}")
        print(f"  Brightness: {features.brightness:.1f} ± {features.brightness_std:.1f}")

        print("\n[FACE]")
        print(f"  Detected:   {'Yes' if features.face_detected else 'No'}")
        print(f"  Visibility: {features.face_visibility:.4f} ({features.face_visibility*100:.1f}%)")

        print(f"\n[INFO]")
        print(f"  Frames:     {features.n_frames}")

        print(f"\n[FEATURE VECTOR]")
        vec = features.to_vector()
        names = ['motion_energy', 'motion_peak', 'motion_var',
                'blur', 'brightness', 'brightness_std', 'face_vis', 'face_det']
        for name, val in zip(names, vec):
            bar = '█' * int(val / 10) if val < 100 else '█' * 10
            print(f"  {name:15s}: {bar:10s} {val:.2f}")

    @staticmethod
    def compare_features(
        features_list: List[VideoFeatures],
        labels: List[int] = None
    ):
        """Compare features across multiple videos."""
        if not features_list:
            return

        vectors = np.array([f.to_vector() for f in features_list])
        names = ['motion_energy', 'motion_peak', 'motion_var',
                'blur', 'brightness', 'brightness_std', 'face_vis', 'face_det']

        print(f"\n{'='*80}")
        print("FEATURE COMPARISON")
        print(f"{'='*80}")
        print(f"{'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print('-' * 55)

        for i, name in enumerate(names):
            col = vectors[:, i]
            print(f"{name:<15} {np.mean(col):>10.2f} {np.std(col):>10.2f} "
                  f"{np.min(col):>10.2f} {np.max(col):>10.2f}")

        if labels is not None:
            print(f"\n[BY LABEL]")
            for label in sorted(set(labels)):
                mask = np.array(labels) == label
                print(f"\nLabel {label} (n={mask.sum()}):")
                for name, col in zip(names, vectors.T):
                    print(f"  {name:<15}: {np.mean(col[mask]):.2f} ± {np.std(col[mask]):.2f}")


def demo():
    """Demo with sample data."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "="*60)
    print("LAYER 1 FEATURE EXTRACTOR - DEMO")
    print("="*60)

    extractor = FeatureExtractorLayer1()

    # Demo: Create synthetic frames with different properties
    print("\n[1] Testing with synthetic frames...")

    # Low motion, sharp, bright
    low_motion_frames = []
    base = np.random.randint(100, 150, (480, 640), dtype=np.uint8)
    for i in range(30):
        noise = np.random.randint(-5, 5, (480, 640), dtype=np.int16)
        frame = np.clip(base + noise, 0, 255).astype(np.uint8)
        low_motion_frames.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    f1 = extractor.extract_from_frames(low_motion_frames, "low_motion_sharp")
    FeatureVisualizer.print_features(f1)

    # High motion, blurry, dark
    high_motion_frames = []
    for i in range(30):
        offset = i * 5
        frame = np.roll(base, offset, axis=1)
        noise = np.random.randint(-20, 20, (480, 640), dtype=np.int16)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        high_motion_frames.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    f2 = extractor.extract_from_frames(high_motion_frames, "high_motion_blur")
    FeatureVisualizer.print_features(f2)

    # Compare
    print("\n[2] Feature Comparison:")
    FeatureVisualizer.compare_features([f1, f2], labels=[5, 1])

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)

    return [f1, f2]


if __name__ == "__main__":
    demo()
