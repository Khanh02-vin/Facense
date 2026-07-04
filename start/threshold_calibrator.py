"""
Threshold Calibrator - Calibrate Thresholds từ Dataset Statistics

Dùng statistics đã thu thập từ dataset_scanner.py để:
1. Calibrate baseline thresholds cho Frame Extraction
2. Calibrate baseline thresholds cho Clip/Motion Extraction
3. So sánh các phương pháp (Median+IQR, Percentile, etc.)
4. Đề xuất thresholds tối ưu

Output: baseline_thresholds.json

References:
- idea.md: Phần "Corpus-Based Calibration"
- dataset_scanner.py: Nguồn statistics
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ========================
# Data Classes
# ========================

@dataclass
class ThresholdConfig:
    """Cấu hình cho một threshold."""
    name: str
    value: float
    method: str  # 'median_iqr', 'percentile', 'fixed'
    description: str

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'method': self.method,
            'description': self.description,
        }


@dataclass
class CalibrationResult:
    """Kết quả calibrate cho một metric."""
    metric_name: str
    statistics: Dict  # Từ dataset_statistics.json

    # Thresholds từ các methods
    median_iqr_threshold: float = 0.0
    percentile_1_threshold: float = 0.0
    percentile_5_threshold: float = 0.0
    percentile_10_threshold: float = 0.0
    percentile_20_threshold: float = 0.0
    percentile_25_threshold: float = 0.0
    percentile_75_threshold: float = 0.0
    percentile_80_threshold: float = 0.0
    percentile_90_threshold: float = 0.0
    percentile_95_threshold: float = 0.0
    percentile_99_threshold: float = 0.0

    # Recommended
    recommended_threshold: float = 0.0
    recommended_method: str = ""

    def to_dict(self) -> Dict:
        return {
            'metric_name': self.metric_name,
            'statistics': self.statistics,
            'median_iqr_threshold': self.median_iqr_threshold,
            'percentile_1_threshold': self.percentile_1_threshold,
            'percentile_5_threshold': self.percentile_5_threshold,
            'percentile_10_threshold': self.percentile_10_threshold,
            'percentile_20_threshold': self.percentile_20_threshold,
            'percentile_25_threshold': self.percentile_25_threshold,
            'percentile_75_threshold': self.percentile_75_threshold,
            'percentile_80_threshold': self.percentile_80_threshold,
            'percentile_90_threshold': self.percentile_90_threshold,
            'percentile_95_threshold': self.percentile_95_threshold,
            'percentile_99_threshold': self.percentile_99_threshold,
            'recommended_threshold': self.recommended_threshold,
            'recommended_method': self.recommended_method,
        }


@dataclass
class CalibrationReport:
    """Report đầy đủ cho tất cả thresholds."""
    calibration_date: str = field(default_factory=lambda: datetime.now().isoformat())
    statistics_file: str = ""
    total_videos: int = 0
    total_frames: int = 0

    # Per-metric calibrations
    blur_calibration: Dict = field(default_factory=dict)
    brightness_min_calibration: Dict = field(default_factory=dict)
    brightness_max_calibration: Dict = field(default_factory=dict)
    face_size_calibration: Dict = field(default_factory=dict)
    motion_start_calibration: Dict = field(default_factory=dict)
    motion_stop_calibration: Dict = field(default_factory=dict)

    # All recommended thresholds
    recommended_thresholds: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'calibration_date': self.calibration_date,
            'statistics_file': self.statistics_file,
            'total_videos': self.total_videos,
            'total_frames': self.total_frames,
            'blur_calibration': self.blur_calibration,
            'brightness_min_calibration': self.brightness_min_calibration,
            'brightness_max_calibration': self.brightness_max_calibration,
            'face_size_calibration': self.face_size_calibration,
            'motion_start_calibration': self.motion_start_calibration,
            'motion_stop_calibration': self.motion_stop_calibration,
            'recommended_thresholds': self.recommended_thresholds,
        }


# ========================
# Calibrators
# ========================

class ThresholdCalibrator:
    """
    Calibrate thresholds từ dataset statistics.

    Các phương pháp:
    1. Median + IQR: threshold = median + k * IQR
    2. Percentile: threshold = percentile(values, p)
    3. Median + MAD: threshold = median + k * 1.4826 * MAD

    Usage:
        calibrator = ThresholdCalibrator()
        calibrator.load_statistics("dataset_statistics.json")
        report = calibrator.calibrate()
        calibrator.save("baseline_thresholds.json")
    """

    def __init__(self, k: float = 1.5):
        """
        Args:
            k: Multiplier cho IQR/MAD (k=1.5 normal, k=3.0 strict)
        """
        self.k = k
        self.statistics: Optional[Dict] = None
        self.report = CalibrationReport()

    def load_statistics(self, stats_path: str) -> bool:
        """
        Load statistics từ dataset_scanner.py output.

        Args:
            stats_path: Path đến dataset_statistics.json

        Returns:
            True nếu load thành công
        """
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")

        with open(stats_path, 'r', encoding='utf-8') as f:
            self.statistics = json.load(f)

        self.report.statistics_file = str(stats_path)
        self.report.total_videos = self.statistics.get('total_videos', 0)
        self.report.total_frames = self.statistics.get('total_frames', 0)

        return True

    def calibrate_blur(self) -> CalibrationResult:
        """
        Calibrate blur threshold.

        Frame được coi là "sharp" nếu blur_score > threshold
        (Laplacian variance cao = sharp)
        """
        blur_stats = self.statistics.get('blur_stats', {})

        values = blur_stats.get('values', [])
        if not values:
            values = self._estimate_blur_distribution(blur_stats)

        median = blur_stats.get('median', 100)
        q1 = blur_stats.get('q1', 50)
        q3 = blur_stats.get('q3', 150)
        iqr = q3 - q1

        result = CalibrationResult(
            metric_name='blur',
            statistics=blur_stats,
            median_iqr_threshold=median + self.k * iqr,
            percentile_90_threshold=np.percentile(values, 90) if values else 0,
            percentile_95_threshold=np.percentile(values, 95) if values else 0,
            percentile_99_threshold=np.percentile(values, 99) if values else 0,
        )

        # Recommended: Median + IQR (bền vững)
        result.recommended_threshold = result.median_iqr_threshold
        result.recommended_method = 'median_iqr'

        return result

    def calibrate_brightness(self) -> Tuple[CalibrationResult, CalibrationResult]:
        """
        Calibrate brightness thresholds.

        Brightness score = mean pixel value (0-255)
        - bright_min: Frame too dark nếu < threshold
        - bright_max: Frame too bright nếu > threshold

        Returns:
            (bright_min_calibration, bright_max_calibration)
        """
        brightness_stats = self.statistics.get('brightness_stats', {})

        values = brightness_stats.get('values', [])
        if not values:
            values = self._estimate_brightness_distribution(brightness_stats)

        median = brightness_stats.get('median', 128)
        q1 = brightness_stats.get('q1', 100)
        q3 = brightness_stats.get('q3', 160)
        iqr = q3 - q1

        # Brightness min (too dark) = Q1 - k * IQR
        bright_min_result = CalibrationResult(
            metric_name='brightness_min',
            statistics=brightness_stats,
            median_iqr_threshold=max(0, q1 - self.k * iqr),
            percentile_90_threshold=np.percentile(values, 90) if values else 255,
            percentile_95_threshold=np.percentile(values, 95) if values else 255,
            percentile_99_threshold=np.percentile(values, 99) if values else 255,
        )
        bright_min_result.recommended_threshold = bright_min_result.median_iqr_threshold
        bright_min_result.recommended_method = 'median_iqr'

        # Brightness max (too bright) = Q3 + k * IQR
        bright_max_result = CalibrationResult(
            metric_name='brightness_max',
            statistics=brightness_stats,
            median_iqr_threshold=min(255, q3 + self.k * iqr),
            percentile_10_threshold=np.percentile(values, 10) if values else 0,
            percentile_5_threshold=np.percentile(values, 5) if values else 0,
            percentile_1_threshold=np.percentile(values, 1) if values else 0,
        )
        bright_max_result.recommended_threshold = bright_max_result.median_iqr_threshold
        bright_max_result.recommended_method = 'median_iqr'

        return bright_min_result, bright_max_result

    def calibrate_face_size(self) -> CalibrationResult:
        """
        Calibrate face size threshold.

        Face được coi là "đủ lớn" nếu face_size_ratio > threshold
        """
        face_stats = self.statistics.get('face_size_stats', {})

        values = face_stats.get('values', [])
        if not values:
            values = self._estimate_face_distribution(face_stats)

        median = face_stats.get('median', 0.15)
        q1 = face_stats.get('q1', 0.08)
        q3 = face_stats.get('q3', 0.25)
        iqr = q3 - q1

        result = CalibrationResult(
            metric_name='face_size',
            statistics=face_stats,
            median_iqr_threshold=max(0, q1 - self.k * iqr),  # Min threshold
            percentile_10_threshold=np.percentile(values, 10) if values else 0,
            percentile_5_threshold=np.percentile(values, 5) if values else 0,
            percentile_1_threshold=np.percentile(values, 1) if values else 0,
        )

        # Recommended: P10 (chỉ lấy top 90%)
        result.recommended_threshold = result.percentile_10_threshold
        result.recommended_method = 'percentile_10'

        return result

    def calibrate_motion(self) -> Tuple[CalibrationResult, CalibrationResult]:
        """
        Calibrate motion thresholds.

        - motion_start: Bắt đầu motion clip (frame diff > threshold)
        - motion_stop: Dừng motion clip (frame diff < threshold)

        Returns:
            (motion_start_calibration, motion_stop_calibration)
        """
        motion_stats = self.statistics.get('motion_energy_stats', {})

        values = motion_stats.get('values', [])
        if not values:
            values = self._estimate_motion_distribution(motion_stats)

        median = motion_stats.get('median', 3.0)
        q1 = motion_stats.get('q1', 1.5)
        q3 = motion_stats.get('q3', 5.8)
        iqr = q3 - q1

        # Motion start = Q3 (top 25% motion)
        motion_start_result = CalibrationResult(
            metric_name='motion_start',
            statistics=motion_stats,
            median_iqr_threshold=q3,  # Top 25%
            percentile_75_threshold=np.percentile(values, 75) if values else 0,
            percentile_80_threshold=np.percentile(values, 80) if values else 0,
            percentile_90_threshold=np.percentile(values, 90) if values else 0,
        )
        motion_start_result.recommended_threshold = motion_start_result.median_iqr_threshold
        motion_start_result.recommended_method = 'q3'

        # Motion stop = Q1 (bottom 25% motion)
        motion_stop_result = CalibrationResult(
            metric_name='motion_stop',
            statistics=motion_stats,
            median_iqr_threshold=q1,  # Bottom 25%
            percentile_25_threshold=np.percentile(values, 25) if values else 0,
            percentile_20_threshold=np.percentile(values, 20) if values else 0,
            percentile_10_threshold=np.percentile(values, 10) if values else 0,
        )
        motion_stop_result.recommended_threshold = motion_stop_result.median_iqr_threshold
        motion_stop_result.recommended_method = 'q1'

        return motion_start_result, motion_stop_result

    def calibrate(self) -> CalibrationReport:
        """
        Calibrate tất cả thresholds.

        Returns:
            CalibrationReport với tất cả thresholds đã calibrate
        """
        if not self.statistics:
            raise ValueError("Chưa load statistics. Gọi load_statistics() trước.")

        # Calibrate từng metric
        blur_result = self.calibrate_blur()
        bright_min_result, bright_max_result = self.calibrate_brightness()
        face_result = self.calibrate_face_size()
        motion_start_result, motion_stop_result = self.calibrate_motion()

        # Lưu vào report
        self.report.blur_calibration = blur_result.to_dict()
        self.report.brightness_min_calibration = bright_min_result.to_dict()
        self.report.brightness_max_calibration = bright_max_result.to_dict()
        self.report.face_size_calibration = face_result.to_dict()
        self.report.motion_start_calibration = motion_start_result.to_dict()
        self.report.motion_stop_calibration = motion_stop_result.to_dict()

        # Tổng hợp recommended thresholds
        self.report.recommended_thresholds = {
            'blur_threshold': blur_result.recommended_threshold,
            'brightness_min': bright_min_result.recommended_threshold,
            'brightness_max': bright_max_result.recommended_threshold,
            'face_size_min': face_result.recommended_threshold,
            'motion_start_threshold': motion_start_result.recommended_threshold,
            'motion_stop_threshold': motion_stop_result.recommended_threshold,
            'calibration_method': 'median_iqr',
            'k': self.k,
        }

        return self.report

    def save(self, output_path: str):
        """Lưu calibration results."""
        with open(output_path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)

    # Helper methods để estimate distribution nếu không có raw values
    def _estimate_blur_distribution(self, stats: Dict) -> List[float]:
        """Estimate blur values từ summary statistics."""
        # Giả lập distribution dựa trên median và IQR
        median = stats.get('median', 100)
        q1 = stats.get('q1', 50)
        q3 = stats.get('q3', 150)
        # Tạo synthetic values
        values = np.random.normal(median, (q3 - q1) / 1.35, 1000)
        values = np.clip(values, 0, None)
        return values.tolist()

    def _estimate_brightness_distribution(self, stats: Dict) -> List[float]:
        """Estimate brightness values."""
        median = stats.get('median', 128)
        q1 = stats.get('q1', 100)
        q3 = stats.get('q3', 160)
        values = np.random.normal(median, (q3 - q1) / 1.35, 1000)
        values = np.clip(values, 0, 255)
        return values.tolist()

    def _estimate_face_distribution(self, stats: Dict) -> List[float]:
        """Estimate face size values."""
        median = stats.get('median', 0.15)
        q1 = stats.get('q1', 0.08)
        q3 = stats.get('q3', 0.25)
        values = np.random.normal(median, (q3 - q1) / 1.35, 1000)
        values = np.clip(values, 0, 1)
        return values.tolist()

    def _estimate_motion_distribution(self, stats: Dict) -> List[float]:
        """Estimate motion energy values."""
        median = stats.get('median', 3.0)
        q1 = stats.get('q1', 1.5)
        q3 = stats.get('q3', 5.8)
        values = np.random.exponential(median, 1000)
        return values.tolist()


# ========================
# Utility Functions
# ========================

def compare_methods(stats: Dict, metric_name: str) -> Dict:
    """
    So sánh các phương pháp calibrate cho một metric.

    Returns:
        Dict với so sánh các methods
    """
    values = stats.get('values', [])
    median = stats.get('median', 0)
    q1 = stats.get('q1', 0)
    q3 = stats.get('q3', 0)
    iqr = q3 - q1

    return {
        'metric': metric_name,
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'methods': {
            'median_iqr_k15': median + 1.5 * iqr,
            'median_iqr_k30': median + 3.0 * iqr,
            'percentile_90': np.percentile(values, 90) if values else 0,
            'percentile_95': np.percentile(values, 95) if values else 0,
            'q3': q3,
            'q1': q1,
        }
    }


def print_calibration_report(report: CalibrationReport):
    """In calibration report ra console."""
    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION REPORT")
    print("=" * 60)

    print(f"\n📅 Date: {report.calibration_date}")
    print(f"📁 Source: {report.statistics_file}")
    print(f"📊 Videos: {report.total_videos}, Frames: {report.total_frames}")

    print(f"\n🎯 Recommended Thresholds:")

    for key, value in report.recommended_thresholds.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print(f"\n📋 Detailed Calibrations:")

    # Blur
    blur = report.blur_calibration
    if blur:
        print(f"\n   🔍 BLUR (Laplacian variance):")
        print(f"      Median + IQR: {blur.get('median_iqr_threshold', 0):.2f}")
        print(f"      P90: {blur.get('percentile_90_threshold', 0):.2f}")
        print(f"      P95: {blur.get('percentile_95_threshold', 0):.2f}")
        print(f"      ✅ Recommended: {blur.get('recommended_threshold', 0):.2f} ({blur.get('recommended_method', '')})")

    # Brightness
    bright_min = report.brightness_min_calibration
    bright_max = report.brightness_max_calibration
    if bright_min and bright_max:
        print(f"\n   💡 BRIGHTNESS (0-255):")
        print(f"      Min - Median + IQR: {bright_min.get('median_iqr_threshold', 0):.2f}")
        print(f"      Max - Median + IQR: {bright_max.get('median_iqr_threshold', 0):.2f}")
        print(f"      ✅ Recommended: [{bright_min.get('recommended_threshold', 0):.2f}, {bright_max.get('recommended_threshold', 0):.2f}]")

    # Face
    face = report.face_size_calibration
    if face:
        print(f"\n   👤 FACE SIZE (ratio):")
        print(f"      P10: {face.get('percentile_10_threshold', 0):.4f}")
        print(f"      P5: {face.get('percentile_5_threshold', 0):.4f}")
        print(f"      ✅ Recommended: {face.get('recommended_threshold', 0):.4f} ({face.get('recommended_method', '')})")

    # Motion
    motion_start = report.motion_start_calibration
    motion_stop = report.motion_stop_calibration
    if motion_start and motion_stop:
        print(f"\n   ⚡ MOTION (frame diff):")
        print(f"      Start (Q3): {motion_start.get('median_iqr_threshold', 0):.4f}")
        print(f"      Stop (Q1): {motion_stop.get('median_iqr_threshold', 0):.4f}")
        print(f"      ✅ Recommended: [{motion_stop.get('recommended_threshold', 0):.4f}, {motion_start.get('recommended_threshold', 0):.4f}]")

    print("\n" + "=" * 60)


# ========================
# Usage Example
# ========================

if __name__ == "__main__":
    print("=== Threshold Calibrator Demo ===\n")

    # Ví dụ 1: Calibrate từ statistics file
    print("--- 1. Full Calibration ---")
    stats_file = "data/dataset_base/dataset_statistics.json"
    output_file = "data/dataset_base/baseline_thresholds.json"

    try:
        calibrator = ThresholdCalibrator(k=1.5)
        calibrator.load_statistics(stats_file)
        report = calibrator.calibrate()

        # In report
        print_calibration_report(report)

        # Save
        calibrator.save(output_file)
        print(f"\n✅ Saved to: {output_file}")

    except FileNotFoundError:
        print(f"⚠️  Statistics file not found: {stats_file}")
        print("   Run dataset_scanner.py first to generate statistics.")

    # Ví dụ 2: Test với sample statistics
    print("\n--- 2. Test Calibration ---")
    sample_stats = {
        'blur_stats': {
            'median': 150.5,
            'q1': 80.0,
            'q3': 220.0,
        },
        'brightness_stats': {
            'median': 128.0,
            'q1': 100.0,
            'q3': 160.0,
        },
        'face_size_stats': {
            'median': 0.15,
            'q1': 0.08,
            'q3': 0.25,
        },
        'motion_energy_stats': {
            'median': 3.2,
            'q1': 1.5,
            'q3': 5.8,
        }
    }

    # Mock calibration với sample stats
    print("Sample statistics:")
    for key, value in sample_stats.items():
        print(f"   {key}: median={value['median']}, q1={value['q1']}, q3={value['q3']}")

    iqr = 220.0 - 80.0
    print(f"\nSample blur threshold (median + 1.5*IQR): {150.5 + 1.5 * iqr:.2f}")

    print("\n✅ Threshold Calibrator ready!")