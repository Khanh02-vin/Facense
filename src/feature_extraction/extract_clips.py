"""
Extract Video Clips from Dataset
Chạy sau pipeline để tạo clips cho labeling UI
"""

import cv2
import json
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = Path("D:/Dataset/Face_project_datset")
OUTPUT_DIR = Path("D:/Project/Face_project/data/dataset_processed")
CLIPS_DIR = OUTPUT_DIR / "clips"
FEATURES_FILE = OUTPUT_DIR / "features.json"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def get_existing_clip_names():
    """Get names of clips already extracted."""
    existing = set()
    if CLIPS_DIR.exists():
        for f in CLIPS_DIR.glob("*.mp4"):
            existing.add(f.stem)
    return existing


def extract_clip(video_path: str, clip_name: str, max_frames: int = 30) -> bool:
    """Extract frames and save as video clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_indices = list(range(0, total_frames, max(total_frames // max_frames, 1)))[:max_frames]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        return False

    clip_path = CLIPS_DIR / f"{clip_name}.mp4"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(clip_path), fourcc, min(fps, 10), (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True


def main():
    print("="*60)
    print("EXTRACT CLIPS FOR LABELING UI")
    print("="*60)

    # Load features to get video list
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found. Run pipeline first.")
        return

    with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
        features = json.load(f)

    print(f"Found {len(features)} processed items")

    # Get existing clips
    existing = get_existing_clip_names()
    print(f"Already extracted: {len(existing)} clips")

    # Find videos to process
    videos = []
    for item in features:
        identity = item.get('identity', 'unknown')
        video_file = item.get('video_file', '')
        clip_name = item.get('clip_name', '')

        if not video_file or not clip_name:
            continue
        if clip_name in existing:
            continue

        video_path = DATASET_DIR / identity / video_file
        if video_path.exists():
            videos.append({
                'path': str(video_path),
                'clip_name': clip_name
            })

    print(f"Need to extract: {len(videos)} clips")
    print("-"*60)

    if not videos:
        print("All clips already extracted!")
        return

    # Extract
    import time
    start = time.time()
    saved = 0
    failed = 0

    for i, v in enumerate(videos):
        ok = extract_clip(v['path'], v['clip_name'])
        if ok:
            saved += 1
        else:
            failed += 1

        pct = (i+1)/len(videos)*100
        elapsed = time.time() - start
        eta = (elapsed/(i+1))*(len(videos)-i-1) if i > 0 else 0

        bar_len = 25
        filled = int(bar_len*(i+1)/len(videos))
        bar = "█"*filled + "░"*(bar_len-filled)

        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}% | {i+1}/{len(videos)} | "
            f"Saved: {saved} | Failed: {failed} | ETA: {int(eta/60)}m      "
        )
        sys.stdout.flush()

        # Save checkpoint every 100
        if (i+1) % 100 == 0:
            sys.stdout.write(f"\n  [CHECKPOINT] {saved} clips saved")

    print(f"\n\n{'='*60}")
    print(f"DONE: {saved} clips saved, {failed} failed")
    print(f"Clips directory: {CLIPS_DIR}")
    print(f"\nNext: python src/user_labeling_ui.py")


if __name__ == "__main__":
    main()
