"""
Sample Ranker - Xếp hạng và Chọn Representative Samples

Dùng để:
1. Xếp hạng ALL frames từ dataset vào 5 nhóm:
   - Quá tốt (excellent)
   - Tốt (good)
   - Trung bình (average)
   - Tệ (bad)
   - Quá tệ (terrible)

2. Chọn representative samples từ mỗi nhóm
   (5-10 samples mỗi nhóm = 25-50 samples total)

3. Output danh sách để human label

Input:
- dataset_statistics.json (từ dataset_scanner.py)
- baseline_thresholds.json (từ threshold_calibrator.py)
- Hoặc dùng trực tiếp frames từ dataset

Output: samples_to_label.json

References:
- idea.md: Phần "Representative samples để human label"
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import logging

from src.quality_checker import QualityChecker, QualityConfig
from src.adaptive_threshold import AdaptiveVideoThreshold

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================
# Data Classes
# ========================

@dataclass
class SampleInfo:
    """Thông tin về một sample."""
    path: str
    video_name: str
    person_name: str
    frame_index: int
    timestamp: float

    # Quality scores
    blur_score: float = 0.0
    brightness_score: float = 0.0
    face_size_score: float = 0.0
    motion_score: float = 0.0

    # Combined score
    overall_score: float = 0.0

    # Group
    group: str = ""

    def to_dict(self) -> Dict:
        return {
            'path': self.path,
            'video_name': self.video_name,
            'person_name': self.person_name,
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
            'blur_score': self.blur_score,
            'brightness_score': self.brightness_score,
            'face_size_score': self.face_size_score,
            'motion_score': self.motion_score,
            'overall_score': self.overall_score,
            'group': self.group,
        }


@dataclass
class RankingGroup:
    """Một nhóm xếp hạng."""
    name: str
    description: str
    threshold_min: float
    threshold_max: float
    samples: List[SampleInfo] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'threshold_min': self.threshold_min,
            'threshold_max': self.threshold_max,
            'count': self.count,
            'samples': [s.to_dict() for s in self.samples],
        }


@dataclass
class RankingReport:
    """Report đầy đủ của ranking."""
    ranking_date: str = field(default_factory=lambda: datetime.now().isoformat())
    total_frames_scanned: int = 0
    ranking_method: str = "median_iqr"

    # Groups
    excellent_group: Dict = field(default_factory=dict)
    good_group: Dict = field(default_factory=dict)
    average_group: Dict = field(default_factory=dict)
    bad_group: Dict = field(default_factory=dict)
    terrible_group: Dict = field(default_factory=dict)

    # All samples to label
    samples_to_label: List[Dict] = field(default_factory=list)

    # Threshold stats
    thresholds: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'ranking_date': self.ranking_date,
            'total_frames_scanned': self.total_frames_scanned,
            'ranking_method': self.ranking_method,
            'excellent_group': self.excellent_group,
            'good_group': self.good_group,
            'average_group': self.average_group,
            'bad_group': self.bad_group,
            'terrible_group': self.terrible_group,
            'samples_to_label': self.samples_to_label,
            'thresholds': self.thresholds,
        }


# ========================
# Sample Ranker
# ========================

class SampleRanker:
    """
    Xếp hạng frames và chọn representative samples.

    Sử dụng Median + IQR để xác định outliers:
    - excellent: > Q3 + 1.5*IQR (quá tốt - outliers trên)
    - good: Q3 → Q3 + 1.5*IQR (tốt)
    - average: Q1 → Q3 (trung bình)
    - bad: Q1 - 1.5*IQR → Q1 (tệ)
    - terrible: < Q1 - 1.5*IQR (quá tệ - outliers dưới)

    Usage:
        ranker = SampleRanker(dataset_path, thresholds)
        report = ranker.rank()
        report.save("samples_to_label.json")
    """

    def __init__(
        self,
        dataset_path: str,
        thresholds: Optional[Dict] = None,
        k: float = 1.5,
        samples_per_group: int = 10,
        quality_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
    ):
        """
        Args:
            dataset_path: Đường dẫn đến dataset
            thresholds: Baseline thresholds (từ threshold_calibrator.py)
            k: Multiplier cho IQR
            samples_per_group: Số samples muốn chọn mỗi nhóm
            quality_weights: (blur, brightness, face, motion)
        """
        self.dataset_path = Path(dataset_path)
        self.thresholds = thresholds or {}
        self.k = k
        self.samples_per_group = samples_per_group
        self.quality_weights = quality_weights

        # Statistics từ dataset
        self.stats = {
            'blur': [],
            'brightness': [],
            'face_size': [],
            'motion': []
        }

        # All samples
        self.all_samples: List[SampleInfo] = []

        # Ranking groups
        self.groups: Dict[str, RankingGroup] = {}

    def load_thresholds(self, thresholds_path: str) -> bool:
        """Load thresholds từ baseline_thresholds.json."""
        if not Path(thresholds_path).exists():
            logger.warning(f"Thresholds file not found: {thresholds_path}")
            return False

        with open(thresholds_path, 'r') as f:
            self.thresholds = json.load(f)

        return True

    def calculate_combined_score(
        self,
        blur_score: float,
        brightness_score: float,
        face_size_score: float,
        motion_score: float = 0.0
    ) -> float:
        """Tính combined quality score."""
        w = self.quality_weights
        return (
            blur_score * w[0] +
            brightness_score * w[1] +
            face_size_score * w[2] +
            motion_score * w[3]
        )

    def scan_frames(self, limit: int = None) -> int:
        """
        Scan tất cả frames trong dataset.

        Args:
            limit: Giới hạn số frames (cho testing)

        Returns:
            Số frames đã scan
        """
        logger.info(f"Scanning frames in: {self.dataset_path}")

        frame_count = 0
        quality_checker = QualityChecker(QualityConfig())

        # Duyệt qua tất cả videos
        for person_folder in self.dataset_path.iterdir():
            if not person_folder.is_dir():
                continue

            person_name = person_folder.name

            for video_file in person_folder.glob("*"):
                if video_file.suffix.lower() not in ['.mp4', '.avi', '.mkv', '.mov']:
                    continue

                video_name = video_file.stem
                cap = cv2.VideoCapture(str(video_file))

                if not cap.isOpened():
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Tính quality scores
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur_score = self._calculate_blur_score(gray)
                    brightness_score = self._calculate_brightness_score(gray)
                    face_size_score = self._calculate_face_size_score(frame)
                    motion_score = 0  # Cần frame trước đó

                    combined_score = self.calculate_combined_score(
                        blur_score, brightness_score, face_size_score, motion_score
                    )

                    # Lưu vào stats
                    self.stats['blur'].append(blur_score)
                    self.stats['brightness'].append(brightness_score)
                    self.stats['face_size'].append(face_size_score)

                    # Tạo sample
                    sample = SampleInfo(
                        path=str(video_file),
                        video_name=video_name,
                        person_name=person_name,
                        frame_index=frame_idx,
                        timestamp=frame_idx / fps if fps > 0 else 0,
                        blur_score=blur_score,
                        brightness_score=brightness_score,
                        face_size_score=face_size_score,
                        motion_score=motion_score,
                        overall_score=combined_score
                    )
                    self.all_samples.append(sample)

                    frame_idx += 1
                    frame_count += 1

                    if limit and frame_count >= limit:
                        cap.release()
                        return frame_count

                cap.release()

                if limit and frame_count >= limit:
                    break

        logger.info(f"Scanned {frame_count} frames")
        return frame_count

    def _calculate_blur_score(self, gray_frame: np.ndarray) -> float:
        """Tính blur score (Laplacian variance)."""
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        variance = laplacian.var()
        return min(variance / 500, 1.0)  # Normalize

    def _calculate_brightness_score(self, gray_frame: np.ndarray) -> float:
        """Tính brightness score (0.5 = tối ưu)."""
        mean = np.mean(gray_frame)
        brightness = mean / 255.0

        # Optimal = 0.5, penalty cho quá tối hoặc quá sáng
        optimal = 0.5
        score = 1.0 - abs(brightness - optimal) * 2
        return max(0, min(1, score))

    def _calculate_face_size_score(self, frame: np.ndarray) -> float:
        """Tính face size score (giả lập - cần face detection thực)."""
        h, w = frame.shape[:2]
        # Giả định: face size score = frame area / some baseline
        # Thực tế cần dùng QualityChecker để detect face
        return 0.5  # Placeholder

    def compute_thresholds_from_stats(self) -> Dict:
        """Tính thresholds từ statistics đã scan."""
        thresholds = {}

        for key in ['blur', 'brightness', 'face_size', 'motion']:
            values = np.array(self.stats[key])
            if len(values) == 0:
                continue

            median = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            thresholds[key] = {
                'median': float(median),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'threshold_excellent': float(q3 + self.k * iqr),
                'threshold_terrible': float(q1 - self.k * iqr)
            }

        self.thresholds = thresholds
        return thresholds

    def rank_samples(self) -> Dict[str, RankingGroup]:
        """Xếp hạng samples vào các nhóm."""
        if not self.all_samples:
            raise ValueError("Chưa scan frames. Gọi scan_frames() trước.")

        if not self.thresholds:
            self.compute_thresholds_from_stats()

        # Khởi tạo groups
        self.groups = {
            'excellent': RankingGroup(
                name='excellent',
                description='Quá tốt (> Q3 + 1.5×IQR)',
                threshold_min=self.thresholds['blur']['threshold_excellent'],
                threshold_max=float('inf')
            ),
            'good': RankingGroup(
                name='good',
                description='Tốt (Q3 → Q3 + 1.5×IQR)',
                threshold_min=self.thresholds['blur']['q3'],
                threshold_max=self.thresholds['blur']['threshold_excellent']
            ),
            'average': RankingGroup(
                name='average',
                description='Trung bình (Q1 → Q3)',
                threshold_min=self.thresholds['blur']['q1'],
                threshold_max=self.thresholds['blur']['q3']
            ),
            'bad': RankingGroup(
                name='bad',
                description='Tệ (Q1 - 1.5×IQR → Q1)',
                threshold_min=self.thresholds['blur']['threshold_terrible'],
                threshold_max=self.thresholds['blur']['q1']
            ),
            'terrible': RankingGroup(
                name='terrible',
                description='Quá tệ (< Q1 - 1.5×IQR)',
                threshold_min=0,
                threshold_max=self.thresholds['blur']['threshold_terrible']
            ),
        }

        # Phân loại samples
        for sample in self.all_samples:
            score = sample.overall_score

            if score > self.groups['excellent'].threshold_min:
                sample.group = 'excellent'
                self.groups['excellent'].samples.append(sample)
            elif score > self.groups['good'].threshold_min:
                sample.group = 'good'
                self.groups['good'].samples.append(sample)
            elif score > self.groups['average'].threshold_min:
                sample.group = 'average'
                self.groups['average'].samples.append(sample)
            elif score > self.groups['bad'].threshold_min:
                sample.group = 'bad'
                self.groups['bad'].samples.append(sample)
            else:
                sample.group = 'terrible'
                self.groups['terrible'].samples.append(sample)

        # Log distribution
        logger.info("Ranking distribution:")
        for name, group in self.groups.items():
            logger.info(f"  {name}: {group.count} samples")

        return self.groups

    def select_representative_samples(self) -> List[SampleInfo]:
        """Chọn representative samples từ mỗi nhóm."""
        selected = []

        for name, group in self.groups.items():
            if group.count == 0:
                continue

            # Chọn samples đại diện
            n_select = min(self.samples_per_group, group.count)

            # Nếu ít hơn cần thiết, lấy tất cả
            if group.count <= n_select:
                selected.extend(group.samples)
            else:
                # Chọn ngẫu nhiên nhưng đa dạng
                # Ưu tiên: top/bottom của nhóm
                sorted_samples = sorted(group.samples, key=lambda s: s.overall_score)

                # Lấy từ các phần của distribution
                indices = np.linspace(0, len(sorted_samples) - 1, n_select, dtype=int)
                for idx in indices:
                    selected.append(sorted_samples[idx])

        # Shuffle để không bias theo group
        random.shuffle(selected)

        # Set group cho selected samples
        for sample in selected:
            sample.group = next(g.name for g in self.groups.values() if sample in g.samples)

        return selected

    def create_report(self) -> RankingReport:
        """Tạo ranking report."""
        report = RankingReport(
            total_frames_scanned=len(self.all_samples),
            ranking_method='median_iqr',
            thresholds=self.thresholds
        )

        # Fill groups
        for name in ['excellent', 'good', 'average', 'bad', 'terrible']:
            group = self.groups.get(name)
            if group:
                setattr(report, f'{name}_group', group.to_dict())

        # Samples to label
        selected = self.select_representative_samples()
        report.samples_to_label = [s.to_dict() for s in selected]

        return report

    def rank(self, limit: int = None) -> RankingReport:
        """
        Full ranking pipeline.

        Args:
            limit: Giới hạn frames (cho testing)

        Returns:
            RankingReport
        """
        # Scan frames
        self.scan_frames(limit)

        # Rank
        self.rank_samples()

        # Create report
        report = self.create_report()

        return report

    def save_report(self, report: RankingReport, output_path: str):
        """Save report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved ranking report to: {output_path}")


# ========================
# Quick Functions
# ========================

def quick_rank(
    dataset_path: str,
    thresholds_path: str = None,
    output_path: str = "samples_to_label.json",
    samples_per_group: int = 10,
    limit: int = None
) -> RankingReport:
    """
    Quick ranking function.

    Args:
        dataset_path: Đường dẫn dataset
        thresholds_path: Path đến baseline_thresholds.json
        output_path: Output path
        samples_per_group: Samples mỗi nhóm
        limit: Giới hạn frames

    Returns:
        RankingReport
    """
    # Load thresholds
    thresholds = {}
    if thresholds_path and Path(thresholds_path).exists():
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)

    # Create ranker
    ranker = SampleRanker(
        dataset_path=dataset_path,
        thresholds=thresholds,
        samples_per_group=samples_per_group
    )

    # Rank
    report = ranker.rank(limit=limit)

    # Save
    ranker.save_report(report, output_path)

    return report


def print_ranking_summary(report: RankingReport):
    """In tóm tắt ranking."""
    print("\n" + "=" * 60)
    print("SAMPLE RANKING SUMMARY")
    print("=" * 60)

    print(f"\n📊 Total frames scanned: {report.total_frames_scanned}")
    print(f"📋 Ranking method: {report.ranking_method}")

    print(f"\n📦 Groups distribution:")
    for name in ['excellent', 'good', 'average', 'bad', 'terrible']:
        group = getattr(report, f'{name}_group', {})
        count = group.get('count', 0)
        threshold = group.get('description', '')
        print(f"   {name:10}: {count:5} frames | {threshold}")

    print(f"\n🎯 Samples to label: {len(report.samples_to_label)}")
    for name in ['excellent', 'good', 'average', 'bad', 'terrible']:
        count = sum(1 for s in report.samples_to_label if s.get('group') == name)
        print(f"   {name:10}: {count} samples")

    print("\n" + "=" * 60)


# ========================
# Usage Example
# ========================

if __name__ == "__main__":
    print("=== Sample Ranker Demo ===\n")

    # Ví dụ 1: Full ranking
    print("--- 1. Full Ranking ---")
    try:
        report = quick_rank(
            dataset_path=r"D:\Dataset\Face_project_datset",
            thresholds_path="baseline_thresholds.json",
            output_path="samples_to_label.json",
            samples_per_group=10
        )
        print_ranking_summary(report)
    except Exception as e:
        print(f"Error: {e}")

    # Ví dụ 2: Quick test (100 frames)
    print("\n--- 2. Quick Test (100 frames) ---")
    try:
        report_test = quick_rank(
            dataset_path=r"D:\Dataset\Face_project_datset",
            output_path="samples_to_label_test.json",
            samples_per_group=5,
            limit=100
        )
        print_ranking_summary(report_test)
    except Exception as e:
        print(f"Error: {e}")

    print("\n✅ Sample Ranker ready!")