"""
Video Pre-Filter Module

Validate videos before frame extraction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

@dataclass(slots=True)
class VideoInfo:
    """Basic information of a validated video."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    is_valid: bool = True
    reason: Optional[str] = None


class VideoPreFilter:
    """Validate videos before frame extraction."""

    def __init__(
        self,
        min_frames: int = 10,
        min_resolution: Tuple[int, int] = (480, 480),
        min_duration: float = 1.0,
        max_duration: float = 3600.0,
        skip_duplicate: bool = True,
        hash_sample_frames: int = 10,
    ):
        self.min_frames = min_frames
        self.min_resolution = min_resolution

        self.min_duration = min_duration
        self.max_duration = max_duration

        self.skip_duplicate = skip_duplicate
        self.hash_sample_frames = hash_sample_frames

        self._seen_hashes: Dict[str, str] = {}

    def _invalid(
        self,
        path: str,
        reason: str,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        frame_count: int = 0,
        duration: float = 0.0,
    ) -> VideoInfo:
        """Create an invalid VideoInfo."""

        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            is_valid=False,
            reason=reason,
        )

    def check_video(self, video_path: str) -> VideoInfo:
        """
        Validate a video before processing.
        """

        video_path = str(video_path)

        if not Path(video_path).exists():
            return self._invalid(
                video_path,
                "FILE_NOT_FOUND",
            )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return self._invalid(
                video_path,
                "CANNOT_OPEN",
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = (
            frame_count / fps
            if fps > 0
            else 0.0
        )
        if width < self.min_resolution[0] or \
           height < self.min_resolution[1]:

            cap.release()

            return self._invalid(
                video_path,
                "LOW_RESOLUTION",
                width,
                height,
                fps,
                frame_count,
                duration,
            )

        if frame_count < self.min_frames:

            cap.release()

            return self._invalid(
                video_path,
                "TOO_FEW_FRAMES",
                width,
                height,
                fps,
                frame_count,
                duration,
            )

        if duration < self.min_duration:

            cap.release()

            return self._invalid(
                video_path,
                "TOO_SHORT",
                width,
                height,
                fps,
                frame_count,
                duration,
            )

        if duration > self.max_duration:

            cap.release()

            return self._invalid(
                video_path,
                "TOO_LONG",
                width,
                height,
                fps,
                frame_count,
                duration,
            )

        if self.skip_duplicate:

            video_hash = self._compute_video_hash(
                video_path,
            )

            if video_hash in self._seen_hashes:

                cap.release()

                return self._invalid(
                    video_path,
                    "DUPLICATE_VIDEO",
                    width,
                    height,
                    fps,
                    frame_count,
                    duration,
                )

            self._seen_hashes[video_hash] = video_path

        cap.release()

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            is_valid=True,
            reason=None,
        )

    def _compute_video_hash(
        self,
        video_path: str,
    ) -> str:
        """
        Compute a lightweight hash for duplicate detection.
        """

        cap = cv2.VideoCapture(video_path)

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        step = max(
            frame_count // self.hash_sample_frames,
            1,
        )

        md5 = hashlib.md5()

        for i in range(self.hash_sample_frames):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                i * step,
            )

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(
                frame,
                (64, 64),
            )

            md5.update(frame.tobytes())

        cap.release()

        return md5.hexdigest()

    def reset(self) -> None:
        """
        Reset duplicate cache.
        """

        self._seen_hashes.clear()

    def get_statistics(self) -> Dict[str, int]:
        """
        Return current prefilter statistics.
        """

        return {
            "cached_hashes": len(self._seen_hashes),
            "min_frames": self.min_frames,
            "min_width": self.min_resolution[0],
            "min_height": self.min_resolution[1],
        }