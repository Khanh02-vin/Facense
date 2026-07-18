"""
Frame Extraction Module.

Public API:
    Sampling
        FrameSampler, FrameInfo

    Quality
        QualityChecker, QualityConfig, QualityResult
        BlurChecker, LightingChecker, DuplicateChecker, FaceChecker

    Pre-filter
        VideoPreFilter, VideoInfo

    Threshold
        AdaptiveVideoThreshold
"""

from src.frame_extraction.adaptive_threshold import AdaptiveVideoThreshold
from src.frame_extraction.frame_sampler import FrameInfo, FrameSampler
from src.frame_extraction.prefilter import VideoInfo, VideoPreFilter
from src.frame_extraction.quality_checker import (
    BlurChecker,
    DuplicateChecker,
    FaceChecker,
    LightingChecker,
    QualityChecker,
    QualityConfig,
    QualityResult,
)
