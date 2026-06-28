"""
Video Segmentation: Frames vs Clips

Tách video thành clips ngắn (1-5 giây)
Mỗi clip = 1 hành động "hoàn chỉnh"

3 detection methods:
1. Motion-based (automatic)
2. Face-tracking-based (practical)
3. User-defined rules (subjective)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class Clip:
    """Một clip ngắn từ video."""
    start_frame: int
    end_frame: int
    duration_frames: int
    quality_score: float
    face_visible: bool
    motion_level: float

    @property
    def duration_seconds(self, fps: float = 30) -> float:
        return self.duration_frames / fps


@dataclass
class SegmentationConfig:
    """Config cho segmentation."""
    # Minimum/Maximum clip duration
    min_duration_frames: int = 30  # 1 second at 30fps
    max_duration_frames: int = 300  # 10 seconds at 30fps

    # Motion thresholds
    motion_threshold_low: float = 5.0  # Below = static
    motion_threshold_high: float = 50.0  # Above = too chaotic

    # Face visibility
    min_face_visibility: float = 0.8  # 80%
    min_face_size: int = 64  # pixels

    # Quality
    min_laplacian: float = 50.0  # Blur detection


class VideoSegmenter:
    """
    Tách video thành clips.

    Methods:
    1. motion_based - Dựa vào optical flow
    2. face_tracking - Dựa vào face detection
    3. hybrid - Kết hợp cả hai
    """

    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()

    def segment_motion_based(self, video_path: str) -> List[Clip]:
        """
        Tách video dựa vào motion (optical flow).

        Logic:
        - Compute optical flow between frames
        - Detect motion onset/offset (motion boundaries)
        - Segment at boundaries
        """
        cap = cv2.VideoCapture(video_path)

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < self.config.min_duration_frames:
            return []

        # Compute optical flow magnitude
        flow_magnitudes = self._compute_flow_magnitudes(frames)

        # Smooth the flow curve
        flow_smooth = self._smooth_motion(flow_magnitudes)

        # Find motion boundaries (onset/offset)
        boundaries = self._find_motion_boundaries(flow_smooth)

        # Create clips from boundaries
        clips = self._create_clips(frames, flow_smooth, boundaries)

        return clips

    def segment_face_tracking(self, video_path: str) -> List[Clip]:
        """
        Tách video dựa vào face tracking.

        Logic:
        - Detect face in each frame
        - Track face visibility
        - Segment when face enters/exits or significant change
        """
        cap = cv2.VideoCapture(video_path)

        frames = []
        face_boxes = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Simple face detection (using cascades)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces_simple(gray)

            frames.append(frame)
            face_boxes.append(faces)

            frame_idx += 1

        cap.release()

        if len(frames) < self.config.min_duration_frames:
            return []

        # Compute face visibility over time
        visibility = self._compute_face_visibility(face_boxes, frames[0].shape)

        # Smooth visibility
        visibility_smooth = self._smooth_motion(visibility)

        # Find face enter/exit boundaries
        boundaries = self._find_face_boundaries(visibility_smooth)

        # Create clips
        clips = self._create_clips(frames, visibility_smooth, boundaries)

        return clips

    def segment_hybrid(self, video_path: str) -> List[Clip]:
        """
        Kết hợp motion + face tracking.
        """
        cap = cv2.VideoCapture(video_path)

        frames = []
        face_boxes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces_simple(gray)

            frames.append(frame)
            face_boxes.append(faces)

        cap.release()

        if len(frames) < self.config.min_duration_frames:
            return []

        # Motion
        flow_magnitudes = self._compute_flow_magnitudes(frames)
        flow_smooth = self._smooth_motion(flow_magnitudes)

        # Face
        visibility = self._compute_face_visibility(face_boxes, frames[0].shape)
        visibility_smooth = self._smooth_motion(visibility)

        # Combine signals (weighted)
        combined = 0.4 * self._normalize(flow_smooth) + 0.6 * visibility_smooth

        # Find boundaries
        boundaries = self._find_motion_boundaries(combined)

        # Create clips with both metrics
        clips = self._create_clips_with_quality(frames, flow_smooth, visibility_smooth, boundaries)

        return clips

    def _compute_flow_magnitudes(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute optical flow magnitude for each frame."""
        magnitudes = []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Magnitude
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            magnitudes.append(mag.mean())

            prev_gray = gray

        # First frame has no flow
        magnitudes.insert(0, 0)

        return np.array(magnitudes)

    def _detect_faces_simple(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Simple face detection using Haar cascades."""
        # Load cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        try:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [tuple(f) for f in faces]
        except:
            return []

    def _compute_face_visibility(self, face_boxes: List, frame_shape: Tuple) -> np.ndarray:
        """Compute face visibility score per frame."""
        h, w = frame_shape[:2]
        frame_area = h * w

        visibility = []
        for boxes in face_boxes:
            if len(boxes) == 0:
                visibility.append(0)
            else:
                # Take largest face
                largest = max(boxes, key=lambda b: b[2] * b[3])
                x, y, bw, bh = largest
                face_area = bw * bh
                # Visibility = face area / frame area
                vis = min(face_area / frame_area * 10, 1.0)  # Scale up
                visibility.append(vis)

        return np.array(visibility)

    def _smooth_motion(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth values with moving average."""
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode='same')

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range."""
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def _find_motion_boundaries(self, motion: np.ndarray) -> List[int]:
        """Find frame indices where motion changes significantly."""
        # Compute gradient (rate of change)
        gradient = np.abs(np.diff(motion))
        gradient = np.concatenate([[0], gradient])

        # Find peaks in gradient (boundaries)
        threshold = np.percentile(gradient, 90)  # Top 10% changes

        boundaries = [0]  # Always start at frame 0

        for i in range(1, len(gradient)):
            if gradient[i] > threshold:
                # Check if far enough from last boundary
                if i - boundaries[-1] >= self.config.min_duration_frames:
                    boundaries.append(i)

        boundaries.append(len(motion))  # Always end at last frame

        return boundaries

    def _find_face_boundaries(self, visibility: np.ndarray) -> List[int]:
        """Find where face enters/exits."""
        # Binary: face visible or not
        binary = (visibility > self.config.min_face_visibility).astype(int)

        boundaries = [0]

        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                # Face appeared or disappeared
                if i - boundaries[-1] >= self.config.min_duration_frames:
                    boundaries.append(i)

        boundaries.append(len(visibility))

        return boundaries

    def _create_clips(
        self,
        frames: List[np.ndarray],
        motion: np.ndarray,
        boundaries: List[int]
    ) -> List[Clip]:
        """Create clips from boundaries."""
        clips = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Check duration
            duration = end - start
            if duration < self.config.min_duration_frames:
                continue
            if duration > self.config.max_duration_frames:
                # Split long clips
                sub_clips = self._split_long_clip(start, end, motion)
                clips.extend(sub_clips)
            else:
                clip_motion = motion[start:end].mean()
                quality = self._compute_quality(frames[start:end])

                clips.append(Clip(
                    start_frame=start,
                    end_frame=end,
                    duration_frames=duration,
                    quality_score=quality,
                    face_visible=True,  # Simplified
                    motion_level=clip_motion
                ))

        return clips

    def _create_clips_with_quality(
        self,
        frames: List[np.ndarray],
        motion: np.ndarray,
        visibility: np.ndarray,
        boundaries: List[int]
    ) -> List[Clip]:
        """Create clips with quality metrics."""
        clips = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            duration = end - start
            if duration < self.config.min_duration_frames:
                continue

            # Compute metrics for this clip
            avg_motion = motion[start:end].mean()
            avg_visibility = visibility[start:end].mean()
            quality = self._compute_quality(frames[start:end])

            # Face visible if > 80% of frames have face
            face_visible = avg_visibility > 0.5

            clips.append(Clip(
                start_frame=start,
                end_frame=end,
                duration_frames=duration,
                quality_score=quality,
                face_visible=face_visible,
                motion_level=avg_motion
            ))

        return clips

    def _compute_quality(self, frames: List[np.ndarray]) -> float:
        """Compute quality score for a sequence of frames."""
        scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur (Laplacian variance)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(lap_var / 100, 1.0)

            # Brightness
            brightness = gray.mean() / 255.0
            bright_score = 1.0 - abs(brightness - 0.5) * 2

            scores.append((blur_score + bright_score) / 2)

        return np.mean(scores)

    def _split_long_clip(
        self,
        start: int,
        end: int,
        motion: np.ndarray
    ) -> List[Clip]:
        """Split a long clip at natural motion boundaries."""
        clip_motion = motion[start:end]

        # Find local minima (good split points)
        from scipy.signal import find_peaks

        inverted = -clip_motion
        peaks, _ = find_peaks(inverted, distance=self.config.min_duration_frames)

        if len(peaks) == 0:
            return []  # Can't split

        split_points = start + peaks

        # Create sub-clips
        sub_clips = []
        prev = start
        for sp in split_points:
            if sp - prev >= self.config.min_duration_frames:
                sub_clips.append(Clip(
                    start_frame=prev,
                    end_frame=sp,
                    duration_frames=sp - prev,
                    quality_score=0.5,  # Placeholder
                    face_visible=True,
                    motion_level=motion[prev:sp].mean()
                ))
                prev = sp

        return sub_clips


def extract_clip_frames(video_path: str, clip: Clip, n_frames: int = 5) -> List[np.ndarray]:
    """
    Extract N representative frames from a clip.

    Strategies:
    1. Uniform sampling (best for short clips)
    2. Quality-weighted (best for diverse clips)
    3. Key frame detection (most different from neighbors)
    """
    cap = cv2.VideoCapture(video_path)

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)

    frames = []
    total = clip.end_frame - clip.start_frame

    for i in range(n_frames):
        # Uniform sampling
        target_frame = clip.start_frame + int(i * total / n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    return frames


# ============================================================
# USAGE EXAMPLE
# ============================================================

def main():
    """Example usage."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    config = SegmentationConfig(
        min_duration_frames=30,  # 1 second
        max_duration_frames=300,  # 10 seconds
        min_face_visibility=0.5
    )

    segmenter = VideoSegmenter(config)

    # Example video
    video_path = "./data/videos/sample.mp4"

    if Path(video_path).exists():
        print("=" * 60)
        print("VIDEO SEGMENTATION")
        print("=" * 60)

        # Try all methods
        for method in ["motion", "face", "hybrid"]:
            print(f"\n[{method.upper()}] Segmentation:")

            if method == "motion":
                clips = segmenter.segment_motion_based(video_path)
            elif method == "face":
                clips = segmenter.segment_face_tracking(video_path)
            else:
                clips = segmenter.segment_hybrid(video_path)

            print(f"  Found {len(clips)} clips")

            for i, clip in enumerate(clips[:5]):  # Show first 5
                print(f"  Clip {i+1}: frames {clip.start_frame}-{clip.end_frame}, "
                      f"duration={clip.duration_seconds:.1f}s, "
                      f"quality={clip.quality_score:.2f}, "
                      f"motion={clip.motion_level:.1f}")
    else:
        print(f"Video not found: {video_path}")
        print("Using sample data...")


if __name__ == "__main__":
    main()
