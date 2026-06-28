"""
Extract frames from videos for annotation UI.
Creates image files that can be displayed in HTML.
"""

import os
import json
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames_from_video(video_path, n_frames=3):
    """Extract frames from video at evenly spaced intervals."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = list(range(0, total_frames, max(1, total_frames // n_frames)))[:n_frames]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def main():
    dataset_dir = Path("D:/Dataset/Face_project_datset")
    output_dir = Path("./data/annotations/frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotation pairs
    with open("./data/annotations/annotation_pairs.json", 'r', encoding='utf-8') as f:
        pairs_data = json.load(f)

    # Get all needed identities
    needed_identities = set()
    for pair in pairs_data["pairs"]:
        needed_identities.add(pair["identity_A"])
        needed_identities.add(pair["identity_B"])

    print(f"Extracting frames for {len(needed_identities)} identities...")

    # Extract frames for each identity
    for identity in tqdm(sorted(needed_identities)):
        identity_dir = dataset_dir / identity
        if not identity_dir.exists():
            continue

        video_files = list(identity_dir.glob("*.mp4"))
        if not video_files:
            continue

        video_file = video_files[0]
        frames = extract_frames_from_video(str(video_file), n_frames=3)

        # Save frames
        for i, frame in enumerate(frames):
            output_path = output_dir / f"{identity}_{i}.jpg"
            cv2.imwrite(str(output_path), frame)

    # Count extracted frames
    frame_files = list(output_dir.glob("*.jpg"))
    print(f"\nExtracted {len(frame_files)} frames to {output_dir}")

    # Update annotation_pairs.json with frame paths
    for pair in pairs_data["pairs"]:
        identity_a = pair["identity_A"]
        identity_b = pair["identity_B"]
        pair["frame_A"] = f"frames/{identity_a}_0.jpg"
        pair["frame_B"] = f"frames/{identity_b}_0.jpg"

    with open("./data/annotations/annotation_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(pairs_data, f, indent=2, ensure_ascii=False)

    print("Updated annotation_pairs.json with frame paths")


if __name__ == "__main__":
    main()
