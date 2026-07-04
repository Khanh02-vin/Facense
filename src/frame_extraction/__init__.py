"""
Frame Extraction Module.

Public API:
    Pipeline
        FrameExtractionPipeline, PipelineConfig
        VideoProcessingResult, PipelineReport, ExtractedFrame

    Quality
        QualityChecker, QualityConfig, QualityResult
        BlurChecker, LightingChecker, DuplicateChecker, FaceChecker

    Sampling
        FrameSampler, FrameInfo

    Pre-filter
        VideoPreFilter, VideoInfo

    Threshold
        AdaptiveVideoThreshold
"""

from src.frame_extraction.adaptive_threshold import AdaptiveVideoThreshold
from src.frame_extraction.frame_extraction_pipeline import (
    ExtractedFrame,
    FrameExtractionPipeline,
    PipelineConfig,
    PipelineReport,
    VideoProcessingResult,
)
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
