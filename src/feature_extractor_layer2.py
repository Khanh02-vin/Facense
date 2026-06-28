"""
Layer 2 Feature Extractor - Advanced Face Analysis

Advanced features using MediaPipe FaceLandmarker:
1. Smile Intensity
2. Eye Contact Score
3. Head Pose
4. Facial Symmetry

Usage:
    extractor = FeatureExtractorLayer2()
    features = extractor.extract_from_frames(frames)
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional
import json


# Try to import MediaPipe
MEDIAPIPE_AVAILABLE = False
try:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[INFO] MediaPipe not available. Layer 2 features will use approximations.")


@dataclass
class FaceMeshFeatures:
    """Advanced face features from MediaPipe."""
    frame_idx: int

    # Smile
    smile_score: float = 0.0  # 0-1
    mouth_openness: float = 0.0

    # Eye
    eye_contact_score: float = 0.0  # 0-1
    pupil_visibility_left: float = 0.0
    pupil_visibility_right: float = 0.0

    # Head pose (approximate)
    head_yaw: float = 0.0  # Left/Right
    head_pitch: float = 0.0  # Up/Down
    head_roll: float = 0.0  # Tilt

    # Face
    face_symmetry: float = 0.0  # 0-1
    face_clarity: float = 0.0

    # Confidence
    confidence: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.smile_score,
            self.mouth_openness,
            self.eye_contact_score,
            self.pupil_visibility_left,
            self.pupil_visibility_right,
            self.head_yaw,
            self.head_pitch,
            self.head_roll,
            self.face_symmetry,
            self.face_clarity,
        ])

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "smile_score": float(self.smile_score),
            "mouth_openness": float(self.mouth_openness),
            "eye_contact_score": float(self.eye_contact_score),
            "head_yaw": float(self.head_yaw),
            "head_pitch": float(self.head_pitch),
            "head_roll": float(self.head_roll),
            "face_symmetry": float(self.face_symmetry),
            "confidence": float(self.confidence),
        }


class FeatureExtractorLayer2:
    """
    Extract Layer 2 (Advanced) features using MediaPipe FaceLandmarker.

    Features:
    - Smile Intensity: Based on mouth landmark distances
    - Eye Contact: Based on iris position relative to eye
    - Head Pose: Approximated from landmark positions
    - Face Symmetry: Left-right landmark symmetry

    If MediaPipe is not available, uses approximate features.
    """

    # MediaPipe FaceLandmarker landmark indices (approximate)
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270]
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 165, 92]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    NOSE_TIP = 4
    FOREHEAD = 10

    def __init__(self, model_path: str = None):
        self.landmark_model = None

        if MEDIAPIPE_AVAILABLE:
            try:
                from mediapipe.tasks.python.components.containers import BaseOptions

                base_options = BaseOptions(model_asset_path=model_path or 'face_landmarker.task')
                options = FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.landmark_model = FaceLandmarker.create_from_options(options)
                print("[OK] MediaPipe FaceLandmarker loaded")
            except Exception as e:
                print(f"[WARN] MediaPipe initialization failed: {e}")
                print("[INFO] Using approximate features instead")
                self.landmark_model = None
        else:
            print("[INFO] MediaPipe not available. Using approximate features.")

    def extract_from_frame(self, frame: np.ndarray) -> Optional[FaceMeshFeatures]:
        """Extract features from single frame."""
        if self.landmark_model is not None:
            return self._extract_with_mediapipe(frame)
        else:
            return self._extract_approximate(frame)

    def _extract_with_mediapipe(self, frame: np.ndarray) -> Optional[FaceMeshFeatures]:
        """Extract features using MediaPipe FaceLandmarker."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            from mediapipe.framework.formats import ImageFrame
            from mediapipe import Image

            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb.tobytes())
            mp_image.width = rgb.shape[1]
            mp_image.height = rgb.shape[0]

            results = self.landmark_model.detect(mp_image)
        except Exception as e:
            print(f"[WARN] MediaPipe detection failed: {e}")
            return self._extract_approximate(frame)

        if not results or not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        features = FaceMeshFeatures(frame_idx=0)
        features.confidence = 1.0

        # Compute all features
        features.smile_score = self._compute_smile(landmarks)
        features.mouth_openness = self._compute_mouth_openness(landmarks)
        features.eye_contact_score = self._compute_eye_contact(landmarks)

        yaw, pitch, roll = self._compute_head_pose(landmarks)
        features.head_yaw = yaw
        features.head_pitch = pitch
        features.head_roll = roll

        features.face_symmetry = self._compute_symmetry(landmarks)

        return features

    def _extract_approximate(self, frame: np.ndarray) -> Optional[FaceMeshFeatures]:
        """
        Extract approximate features when MediaPipe is not available.

        Uses simple image processing as fallback.
        """
        features = FaceMeshFeatures(frame_idx=0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face region (crude approximation)
        face_region = self._detect_face_region(gray)

        if face_region is None:
            features.confidence = 0.0
            return features

        features.confidence = 0.5  # Lower confidence for approximate

        # Approximate smile from mouth region brightness
        features.smile_score = self._approximate_smile(gray, face_region)

        # Approximate eye contact from face centering
        features.eye_contact_score = self._approximate_eye_contact(gray, face_region)

        # Approximate head pose from face position
        yaw, pitch = self._approximate_head_pose(gray, face_region)
        features.head_yaw = yaw
        features.head_pitch = pitch

        # Approximate symmetry from face structure
        features.face_symmetry = self._approximate_symmetry(gray, face_region)

        # Mouth openness not available without landmarks
        features.mouth_openness = 0.0

        return features

    def _detect_face_region(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect approximate face region using simple methods."""
        # Use face cascade as fallback
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

        if len(faces) > 0:
            return tuple(faces[0])
        return None

    def _approximate_smile(self, gray: np.ndarray, face: Tuple) -> float:
        """Approximate smile from mouth region brightness."""
        x, y, w, h = face

        # Mouth region is in lower third of face
        mouth_y = y + int(h * 0.7)
        mouth_h = int(h * 0.2)

        mouth_region = gray[mouth_y:mouth_y+mouth_h, x:x+w]

        if mouth_region.size == 0:
            return 0.0

        # Higher brightness variance in mouth = more open = possibly smiling
        brightness_var = np.var(mouth_region)
        smile_score = np.clip(brightness_var / 1000, 0, 1)

        return float(smile_score)

    def _approximate_eye_contact(self, gray: np.ndarray, face: Tuple) -> float:
        """Approximate eye contact from face centering."""
        x, y, w, h = face

        # Check if face is centered
        frame_h, frame_w = gray.shape
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2

        # Distance from center
        dist_x = abs(face_center_x - frame_center_x) / (frame_w / 2)
        dist_y = abs(face_center_y - frame_center_y) / (frame_h / 2)

        # Higher score = more centered = better eye contact
        centering = 1.0 - np.sqrt(dist_x**2 + dist_y**2)
        eye_contact = np.clip(centering, 0, 1)

        return float(eye_contact)

    def _approximate_head_pose(self, gray: np.ndarray, face: Tuple) -> Tuple[float, float]:
        """Approximate head pose from face position and size."""
        x, y, w, h = face

        # Yaw: horizontal position
        frame_w = gray.shape[1]
        yaw = (x + w / 2 - frame_w / 2) / (frame_w / 2)

        # Pitch: vertical position
        frame_h = gray.shape[0]
        pitch = (y + h / 2 - frame_h / 2) / (frame_h / 2)

        return float(yaw), float(pitch)

    def _approximate_symmetry(self, gray: np.ndarray, face: Tuple) -> float:
        """Approximate symmetry from face mirroring."""
        x, y, w, h = face

        # Take upper half of face for symmetry
        face_roi = gray[y:y+int(h*0.6), x:x+w]

        if face_roi.size == 0:
            return 0.5

        h, w = face_roi.shape
        mid = w // 2

        # Left and right halves
        left = face_roi[:, :mid] if mid > 0 else np.array([])
        right = face_roi[:, -mid:] if mid > 0 else np.array([])

        if left.size == 0 or right.size == 0:
            return 0.5

        # Flip right and compare
        right_flipped = np.fliplr(right)

        # Resize to match
        min_w = min(left.shape[1], right_flipped.shape[1])
        left_resized = cv2.resize(left, (min_w, left.shape[0]))
        right_resized = cv2.resize(right_flipped, (min_w, right_flipped.shape[0]))

        # Similarity
        diff = np.abs(left_resized.astype(float) - right_resized.astype(float))
        similarity = 1.0 - np.mean(diff) / 255.0

        return float(np.clip(similarity, 0, 1))

    def extract_from_frames(self, frames: List[np.ndarray]) -> List[FaceMeshFeatures]:
        """Extract features from list of frames."""
        all_features = []

        for i, frame in enumerate(frames):
            features = self.extract_from_frame(frame)
            if features:
                features.frame_idx = i
                all_features.append(features)

        return all_features

    def _compute_smile(self, landmarks) -> float:
        """Compute smile intensity from landmarks."""
        def get_lm(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y])

        try:
            # Mouth corners
            left_corner = get_lm(61)
            right_corner = get_lm(291)

            # Upper and lower lip
            upper_lip = get_lm(13)
            lower_lip = get_lm(14)

            # Mouth width and height
            mouth_width = np.linalg.norm(right_corner - left_corner)
            mouth_height = np.linalg.norm(lower_lip - upper_lip)

            if mouth_width < 0.001:
                return 0.0

            # Smile ratio
            smile_ratio = mouth_height / (mouth_width + 0.001)
            return float(np.clip(smile_ratio * 5, 0, 1))
        except:
            return 0.0

    def _compute_mouth_openness(self, landmarks) -> float:
        """Compute mouth openness."""
        try:
            upper = np.array([landmarks[13].x, landmarks[13].y])
            lower = np.array([landmarks[14].x, landmarks[14].y])
            openness = np.linalg.norm(lower - upper)
            return float(np.clip(openness * 20, 0, 1))
        except:
            return 0.0

    def _compute_eye_contact(self, landmarks) -> float:
        """Compute eye contact score."""
        try:
            def get_lm(idx):
                return np.array([landmarks[idx].x, landmarks[idx].y])

            # Eye centers
            left_eye = np.mean([get_lm(i) for i in self.LEFT_EYE[:6]], axis=0)
            right_eye = np.mean([get_lm(i) for i in self.RIGHT_EYE[:6]], axis=0)

            # Iris centers
            left_iris = np.mean([get_lm(i) for i in self.LEFT_IRIS], axis=0)
            right_iris = np.mean([get_lm(i) for i in self.RIGHT_IRIS], axis=0)

            # Offsets
            left_offset = np.linalg.norm(left_iris - left_eye)
            right_offset = np.linalg.norm(right_iris - right_eye)

            # Eye contact = centered iris
            avg_offset = (left_offset + right_offset) / 2
            return float(1.0 - np.clip(avg_offset * 10, 0, 1))
        except:
            return 0.0

    def _compute_head_pose(self, landmarks) -> Tuple[float, float, float]:
        """Compute head pose from landmarks."""
        try:
            def get_lm(idx):
                return np.array([landmarks[idx].x, landmarks[idx].y])

            nose = get_lm(self.NOSE_TIP)
            forehead = get_lm(self.FOREHEAD)
            left_eye = get_lm(33)
            right_eye = get_lm(263)

            # Yaw
            yaw = nose[0] - 0.5

            # Pitch
            pitch = nose[1] - forehead[1]

            # Roll
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            roll = np.arctan2(dy, dx + 0.001)

            return float(yaw), float(pitch), float(roll)
        except:
            return 0.0, 0.0, 0.0

    def _compute_symmetry(self, landmarks) -> float:
        """Compute face symmetry."""
        try:
            def get_lm(idx):
                return np.array([landmarks[idx].x, landmarks[idx].y])

            pairs = [(33, 263), (133, 362), (61, 291), (234, 454)]
            symmetries = []

            for left_idx, right_idx in pairs:
                left = get_lm(left_idx)
                right = get_lm(right_idx)
                mirrored = np.array([1 - right[0], right[1]])

                dist = np.linalg.norm(left - mirrored)
                symmetries.append(1.0 - np.clip(dist * 5, 0, 1))

            return float(np.mean(symmetries))
        except:
            return 0.5


class Layer1andLayer2:
    """Combined Layer 1 + Layer 2 feature extractor."""

    def __init__(self):
        from feature_extractor_layer1 import FeatureExtractorLayer1

        self.layer1 = FeatureExtractorLayer1()
        self.layer2 = FeatureExtractorLayer2()

    def extract(self, frames: List[np.ndarray], video_id: str = "unknown") -> dict:
        """Extract combined features from frames."""
        # Layer 1 features
        layer1_feats = self.layer1.extract_from_frames(frames, video_id)

        # Layer 2 features
        layer2_feats_list = self.layer2.extract_from_frames(frames)

        # Aggregate Layer 2 (mean across frames)
        layer2_agg = {}
        if layer2_feats_list:
            layer2_vecs = np.array([f.to_vector() for f in layer2_feats_list])
            layer2_names = [
                'smile', 'mouth_open', 'eye_contact',
                'pupil_l', 'pupil_r',
                'head_yaw', 'head_pitch', 'head_roll',
                'symmetry', 'clarity'
            ]
            for i, name in enumerate(layer2_names):
                layer2_agg[name] = float(np.mean(layer2_vecs[:, i]))
                layer2_agg[f'{name}_std'] = float(np.std(layer2_vecs[:, i]))

        # Combine
        combined = {
            'video_id': video_id,
            'layer1': layer1_feats.to_dict(),
            'layer2': layer2_agg,
            'combined_dim': 8 + len(layer2_agg),
        }

        return combined


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demo Layer 2 feature extraction."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("="*60)
    print("LAYER 2 FEATURE EXTRACTOR - DEMO")
    print("="*60)

    extractor = FeatureExtractorLayer2()

    # Create synthetic frames
    print("\n[1] Creating synthetic frames...")
    frames = []
    for i in range(30):
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)

    # Test extraction
    print("\n[2] Testing feature extraction...")
    features_list = extractor.extract_from_frames(frames)

    if features_list:
        print(f"  Extracted features from {len(features_list)} frames")

        # Show sample
        f = features_list[0]
        print(f"\n  Sample features (frame 0):")
        print(f"    Smile: {f.smile_score:.3f}")
        print(f"    Eye Contact: {f.eye_contact_score:.3f}")
        print(f"    Head Yaw: {f.head_yaw:.3f}")
        print(f"    Head Pitch: {f.head_pitch:.3f}")
        print(f"    Face Symmetry: {f.face_symmetry:.3f}")
        print(f"    Confidence: {f.confidence:.1f}")

        # Aggregate
        print(f"\n  Aggregated (mean across frames):")
        all_feats = np.array([f.to_vector() for f in features_list])
        names = ['smile', 'mouth_open', 'eye_contact', 'pupil_l', 'pupil_r',
                'head_yaw', 'head_pitch', 'head_roll', 'symmetry', 'clarity']
        for i, name in enumerate(names):
            print(f"    {name}: {np.mean(all_feats[:, i]):.3f}")
    else:
        print("  No features extracted")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    demo()
