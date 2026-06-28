"""
Auto Label với Quality Metrics nâng cao
Dựa trên motion segmentation + frame quality assessment
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Config (từ code của bạn)
MOTION_START_THRESH = 5.0
MOTION_STOP_THRESH = 1.0
MIN_FRAMES_PER_CLIP = 10
BUFFER_FRAMES = 10
BLUR_THRESH = 70.0
BRIGHT_MIN = 20.0
BRIGHT_MAX = 220.0
FACE_CONF_THRESH = 0.6
ACCEPTANCE_RATIO = 0.25

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=FACE_CONF_THRESH)

DATASET_DIR = Path("D:/Dataset/Face_project_datset")
OUTPUT_DIR = Path("D:/Dataset/Face_project_datset/Clipped_Results")
FEATURES_FILE = Path("D:/Project/Face_project/data/dataset_processed/features.json")
OUTPUT_FILE = Path("D:/Project/Face_project/data/auto_labels_v2.json")


def calculate_motion_energy(prev_gray, curr_gray):
    if prev_gray is None:
        return 0.0
    return np.mean(cv2.absdiff(curr_gray, prev_gray))


def assess_frame_quality(frame, gray_frame):
    h, w = frame.shape[:2]
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_32F).var()
    is_blurry = laplacian_var < BLUR_THRESH

    brightness = np.mean(gray_frame)
    is_bad_lighting = brightness < BRIGHT_MIN or brightness > BRIGHT_MAX

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    has_face = False

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            if (bbox.width * w * bbox.height * h) / (w * h) > 0.02:
                has_face = True
                break

    return {
        "is_blurry": is_blurry,
        "is_bad_lighting": is_bad_lighting,
        "has_face": has_face,
        "blur_score": float(laplacian_var),
        "brightness": float(brightness),
        "face_size_ratio": float(bbox.width * bbox.height) if has_face else 0.0
    }


def analyze_video_quality(video_path: str) -> Dict:
    """
    Phân tích toàn bộ video, trả về metrics chất lượng.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_gray = None
    is_in_motion = False
    no_motion_counter = 0

    # Thống kê
    motion_energies = []
    good_frame_count = 0
    total_quality_frames = 0
    blur_scores = []
    brightness_scores = []
    face_scores = []

    # Clip segments
    clip_qualities = []
    current_clip_frames = 0
    current_clip_good = 0
    current_clip_energy_sum = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_energy = calculate_motion_energy(prev_gray, gray_frame)
        prev_gray = gray_frame.copy()

        # Quality assessment
        quality = assess_frame_quality(frame, gray_frame)

        motion_energies.append(motion_energy)
        blur_scores.append(quality["blur_score"])
        brightness_scores.append(quality["brightness"])
        if quality["has_face"]:
            face_scores.append(quality["face_size_ratio"])

        total_quality_frames += 1
        if not quality["is_blurry"] and not quality["is_bad_lighting"] and quality["has_face"]:
            good_frame_count += 1

        # Motion state tracking
        if not is_in_motion:
            if motion_energy > MOTION_START_THRESH:
                is_in_motion = True
                current_clip_frames = 0
                current_clip_good = 0
                current_clip_energy_sum = 0
        else:
            current_clip_frames += 1
            current_clip_energy_sum += motion_energy

            if not quality["is_blurry"] and not quality["is_bad_lighting"] and quality["has_face"]:
                current_clip_good += 1

            if motion_energy < MOTION_STOP_THRESH:
                no_motion_counter += 1
            else:
                no_motion_counter = 0

            if no_motion_counter >= BUFFER_FRAMES:
                is_in_motion = False
                # Evaluate clip quality
                if current_clip_frames >= MIN_FRAMES_PER_CLIP:
                    acceptance = current_clip_good / current_clip_frames
                    if acceptance >= ACCEPTANCE_RATIO:
                        clip_qualities.append({
                            "frames": current_clip_frames,
                            "good_ratio": acceptance,
                            "avg_energy": current_clip_energy_sum / current_clip_frames
                        })

    cap.release()

    if not motion_energies:
        return None

    # Tổng hợp metrics
    metrics = {
        "total_frames": total_frames,
        "good_frame_ratio": good_frame_count / total_quality_frames if total_quality_frames > 0 else 0,
        "avg_motion_energy": float(np.mean(motion_energies)),
        "max_motion_energy": float(np.max(motion_energies)),
        "avg_blur": float(np.mean(blur_scores)),
        "avg_brightness": float(np.mean(brightness_scores)),
        "face_detection_rate": len(face_scores) / total_quality_frames if total_quality_frames > 0 else 0,
        "avg_face_size": float(np.mean(face_scores)) if face_scores else 0,
        "num_qualifying_clips": len(clip_qualities),
        "clip_qualities": clip_qualities
    }

    return metrics


def calculate_preference_score(metrics: Dict) -> Tuple[float, float]:
    """
    Tính preference score (1-5) và confidence (0-1) từ quality metrics.
    """
    score = 0.0
    factors = []

    # 1. Good frame ratio (quan trọng nhất) - weight 30%
    good_ratio = metrics["good_frame_ratio"]
    score += good_ratio * 2.0  # 0-2 điểm
    factors.append(("Good frame ratio", good_ratio))

    # 2. Face detection rate - weight 25%
    face_rate = metrics["face_detection_rate"]
    score += face_rate * 1.5  # 0-1.5 điểm
    factors.append(("Face detection", face_rate))

    # 3. Average blur (sharpness) - weight 15%
    blur = metrics["avg_blur"]
    blur_score = min(1.0, blur / 150.0)  # Normalize: 150 = very sharp
    score += blur_score * 1.0  # 0-1 điểm
    factors.append(("Sharpness", blur_score))

    # 4. Brightness (optimal range) - weight 10%
    bright = metrics["avg_brightness"]
    if BRIGHT_MIN <= bright <= BRIGHT_MAX:
        bright_score = 1.0
    else:
        bright_score = max(0, 1.0 - abs(bright - 128) / 128)
    score += bright_score * 0.5  # 0-0.5 điểm
    factors.append(("Lighting", bright_score))

    # 5. Number of qualifying clips (engaging content) - weight 10%
    num_clips = min(1.0, metrics["num_qualifying_clips"] / 5.0)
    score += num_clips * 0.5  # 0-0.5 điểm
    factors.append(("Engaging clips", num_clips))

    # 6. Average face size (closer is better) - weight 10%
    face_size = min(1.0, metrics["avg_face_size"] * 5)  # Normalize
    score += face_size * 0.5  # 0-0.5 điểm
    factors.append(("Face proximity", face_size))

    # Normalize score to 1-5
    normalized_score = max(1.0, min(5.0, score))

    # Calculate confidence based on data quality
    confidence = 0.5
    if metrics["total_frames"] > 50:
        confidence += 0.1
    if metrics["face_detection_rate"] > 0.5:
        confidence += 0.2
    if good_ratio > 0.3:
        confidence += 0.2

    return normalized_score, min(1.0, confidence)


def auto_label_all_videos():
    """Auto-label tất cả videos trong dataset."""

    # Load existing features
    if FEATURES_FILE.exists():
        with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
            features = json.load(f)
        existing_data = {f['clip_name']: f for f in features}
    else:
        existing_data = {}

    print("="*60)
    print("AUTO LABEL V2 - Quality-Based Preferences")
    print("="*60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Total clips to analyze: {len(existing_data)}")

    # Get unique identities
    identities = sorted(set(f.get('identity', 'unknown') for f in existing_data.values()))
    print(f"Identities: {len(identities)}")

    results = []
    identity_scores = {}

    # Analyze per identity (group videos)
    for identity in identities:
        identity_folder = DATASET_DIR / identity
        if not identity_folder.exists():
            continue

        # Get all videos for this identity
        videos = list(identity_folder.glob("*.mp4"))

        identity_metrics = []
        for video in videos:
            metrics = analyze_video_quality(str(video))
            if metrics:
                identity_metrics.append(metrics)

        if not identity_metrics:
            continue

        # Average metrics across videos
        avg_metrics = {
            "total_frames": np.mean([m["total_frames"] for m in identity_metrics]),
            "good_frame_ratio": np.mean([m["good_frame_ratio"] for m in identity_metrics]),
            "avg_motion_energy": np.mean([m["avg_motion_energy"] for m in identity_metrics]),
            "avg_blur": np.mean([m["avg_blur"] for m in identity_metrics]),
            "avg_brightness": np.mean([m["avg_brightness"] for m in identity_metrics]),
            "face_detection_rate": np.mean([m["face_detection_rate"] for m in identity_metrics]),
            "avg_face_size": np.mean([m["avg_face_size"] for m in identity_metrics]),
            "num_qualifying_clips": np.mean([m["num_qualifying_clips"] for m in identity_metrics]),
            "clip_qualities": []
        }

        score, confidence = calculate_preference_score(avg_metrics)
        identity_scores[identity] = {
            "score": score,
            "confidence": confidence,
            "metrics": avg_metrics
        }

        # Map to clips
        for clip_name, feat in existing_data.items():
            if feat.get('identity') == identity:
                results.append({
                    "clip_id": clip_name,
                    "predicted_rating": score,
                    "rounded_rating": round(score),
                    "confidence": confidence,
                    "identity": identity,
                    "auto_labeled": True,
                    "method": "quality_metrics"
                })

    # Sort by confidence
    results.sort(key=lambda x: -x['confidence'])

    # Statistics
    ratings = [r['rounded_rating'] for r in results]
    print(f"\n[RESULTS]")
    print(f"  Total labeled: {len(results)} clips")
    print(f"  Unique identities: {len(identity_scores)}")
    print(f"\n  Rating distribution:")
    for r in range(1, 6):
        count = ratings.count(r)
        pct = count / len(ratings) * 100 if ratings else 0
        bar = "█" * int(pct / 5)
        print(f"    {r}: {bar} {count} ({pct:.1f}%)")

    # Top/Bottom identities
    sorted_identities = sorted(identity_scores.items(), key=lambda x: -x[1]['score'])
    print(f"\n  TOP 5 identities:")
    for i, (name, data) in enumerate(sorted_identities[:5], 1):
        print(f"    {i}. {name}: {data['score']:.2f} (conf: {data['confidence']:.2f})")

    print(f"\n  BOTTOM 5 identities:")
    for i, (name, data) in enumerate(sorted_identities[-5:], 1):
        print(f"    {i}. {name}: {data['score']:.2f} (conf: {data['confidence']:.2f})")

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(results),
            "identity_scores": identity_scores,
            "predictions": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    face_detector.close()
    return results


if __name__ == "__main__":
    auto_label_all_videos()