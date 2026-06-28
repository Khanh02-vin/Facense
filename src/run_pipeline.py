"""
Run Complete Pipeline on Dataset

Process all videos from D:\Dataset\Face_project_datset

Workflow:
1. Scan videos
2. Extract clips
3. Extract features
4. Generate pairs for labeling
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import sys
import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor_layer1 import FeatureExtractorLayer1
from feature_extractor_layer2 import FeatureExtractorLayer2


DATASET_DIR = Path("D:/Dataset/Face_project_datset")
OUTPUT_DIR = Path("D:/Project/Face_project/data/dataset_processed")
CLIPS_DIR = OUTPUT_DIR / "clips"
FEATURES_FILE = OUTPUT_DIR / "features.json"


class DatasetPipeline:
    """
    Process entire dataset:
    1. Scan videos
    2. Extract clips
    3. Extract features
    4. Save for labeling
    """

    def __init__(self):
        self.extractor_l1 = FeatureExtractorLayer1()
        self.extractor_l2 = FeatureExtractorLayer2()

        # Create output directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)

        print("="*60)
        print("DATASET PIPELINE")
        print("="*60)
        print(f"Dataset: {DATASET_DIR}")
        print(f"Output: {OUTPUT_DIR}")

    def scan_videos(self) -> List[Dict]:
        """Scan dataset for videos."""
        print("\n[1] Scanning videos...")

        videos = []
        for identity_dir in sorted(DATASET_DIR.iterdir()):
            if not identity_dir.is_dir():
                continue

            identity_name = identity_dir.name

            for video_file in sorted(identity_dir.glob("*.mp4")):
                videos.append({
                    'path': str(video_file),
                    'identity': identity_name,
                    'filename': video_file.name
                })

        print(f"  Found {len(videos)} videos")
        return videos

    def _save_clip_video(
        self,
        frames: List[np.ndarray],
        clip_path: str,
        fps: float = 5.0
    ) -> bool:
        """Save frames as a video clip file."""
        if not frames:
            return False
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()
            return True
        except Exception as e:
            print(f"\n  WARNING: Could not save clip {clip_path}: {e}")
            return False

    def extract_clip_from_video(
        self,
        video_path: str,
        clip_name: str,
        max_frames: int = 30,
        save_clip: bool = True
    ) -> Tuple[List[np.ndarray], Dict]:
        """Extract a clip from video (sample frames) and optionally save as file."""
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        # Save clip video file if requested
        clip_saved = False
        if save_clip and frames:
            clip_path = CLIPS_DIR / f"{clip_name}.mp4"
            clip_saved = self._save_clip_video(frames, clip_path, fps=min(fps, 10))

        return frames, {'total_frames': total_frames, 'fps': fps, 'clip_saved': clip_saved}

    def process_video(self, video_info: Dict) -> Dict:
        """Process single video and extract features."""
        video_path = video_info['path']
        identity = video_info['identity']
        filename = video_info['filename']

        clip_name = f"{identity}_{filename.replace('.mp4', '')}"

        try:
            # Extract clip (sample frames) and save to file
            frames, video_meta = self.extract_clip_from_video(
                video_path, clip_name, max_frames=30, save_clip=True
            )

            if len(frames) < 5:
                return None

            # Layer 1 features
            l1_feats = self.extractor_l1.extract_from_frames(frames, clip_name)
            l1_dict = l1_feats.to_dict() if l1_feats else {}

            # Layer 2 features
            l2_feats_list = self.extractor_l2.extract_from_frames(frames)
            if l2_feats_list:
                # Aggregate Layer 2 features (mean)
                l2_vecs = np.array([f.to_vector() for f in l2_feats_list])
                l2_agg = {
                    'smile': float(np.mean(l2_vecs[:, 0])),
                    'mouth_open': float(np.mean(l2_vecs[:, 1])),
                    'eye_contact': float(np.mean(l2_vecs[:, 2])),
                    'pupil_left': float(np.mean(l2_vecs[:, 3])),
                    'pupil_right': float(np.mean(l2_vecs[:, 4])),
                    'head_yaw': float(np.mean(l2_vecs[:, 5])),
                    'head_pitch': float(np.mean(l2_vecs[:, 6])),
                    'head_roll': float(np.mean(l2_vecs[:, 7])),
                    'face_symmetry': float(np.mean(l2_vecs[:, 8])),
                    'face_clarity': float(np.mean(l2_vecs[:, 9])),
                }
            else:
                l2_agg = {k: 0.0 for k in [
                    'smile', 'mouth_open', 'eye_contact', 'pupil_left', 'pupil_right',
                    'head_yaw', 'head_pitch', 'head_roll', 'face_symmetry', 'face_clarity'
                ]}

            # Combine features
            combined = {**l1_dict, **l2_agg}
            combined['identity'] = identity
            combined['video_file'] = filename
            combined['clip_name'] = clip_name
            combined['n_frames'] = len(frames)

            return combined

        except Exception as e:
            print(f"  ERROR: {clip_name}: {e}")
            return None

    def run(self, max_videos: int = None, resume: bool = True):
        """Run pipeline on all videos."""
        # Scan
        videos = self.scan_videos()

        if max_videos:
            videos = videos[:max_videos]

        # Check for existing features to resume
        processed: Set[str] = set()
        all_results: List[Dict] = []
        if resume and FEATURES_FILE.exists():
            try:
                with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                for item in all_results:
                    processed.add(item.get('clip_name', ''))
                print(f"\n[RESUME] Found {len(processed)} existing results - will skip")
            except Exception as e:
                print(f"[WARN] Could not load existing features: {e}")
                all_results = []

        # Filter out already processed
        videos_to_process = [
            v for v in videos
            if f"{v['identity']}_{v['filename'].replace('.mp4', '')}" not in processed
        ]
        total = len(videos_to_process)
        print(f"\n[2] Processing {total} new videos...")
        print("-" * 60)

        if total == 0:
            print("  All videos already processed!")
            self.print_statistics(all_results)
            return all_results

        start_time = time.time()
        new_results = []

        for i, video_info in enumerate(videos_to_process):
            t0 = time.time()
            result = self.process_video(video_info)
            dt = time.time() - t0

            if result:
                new_results.append(result)

            # Progress bar
            pct = (i + 1) / total * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0

            bar_len = 30
            filled = int(bar_len * (i + 1) / total)
            bar = "█" * filled + "░" * (bar_len - filled)

            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}% | {i+1}/{total} | "
                f"Speed: {1/dt:.1f}/s | ETA: {int(eta/60)}m {int(eta)%60}s      "
            )
            sys.stdout.flush()

            # Save every 50 videos
            if (i + 1) % 50 == 0:
                with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_results + new_results, f, indent=2, ensure_ascii=False)
                sys.stdout.write(f"\n  [SAVE] checkpoint saved at {i+1} videos")

        print()  # newline after progress bar

        # Final save
        results = all_results + new_results
        print(f"\n[3] Saving {len(results)} total results...")
        with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  Processed: {len(new_results)} new videos")
        print(f"  Failed: {total - len(new_results)}")

        # Statistics
        self.print_statistics(results)

        return results

    def print_statistics(self, results: List[Dict]):
        """Print feature statistics."""
        if not results:
            return

        print(f"\n[5] Feature Statistics:")

        # Aggregate by identity
        identity_scores = {}
        for r in results:
            ident = r.get('identity', 'unknown')
            score = r.get('attractiveness_score', r.get('blur_score', 0))

            if ident not in identity_scores:
                identity_scores[ident] = []
            identity_scores[ident].append(score)

        # Top identities by mean score
        identity_means = {
            k: np.mean(v) for k, v in identity_scores.items()
        }

        print(f"\n  Top 10 identities by attractiveness:")
        for ident, score in sorted(identity_means.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ident}: {score:.3f}")

        print(f"\n  Total identities: {len(identity_scores)}")

        # Feature ranges
        print(f"\n  Feature ranges:")
        keys = ['motion_energy', 'blur_score', 'brightness', 'smile', 'eye_contact']
        for key in keys:
            values = [r.get(key, 0) for r in results if r.get(key, 0) > 0]
            if values:
                print(f"    {key}: {np.min(values):.2f} - {np.max(values):.2f}")


def generate_labeling_pairs(results: List[Dict], n_pairs: int = 50) -> List[Dict]:
    """Generate pairs for user labeling."""
    import random
    random.seed(42)

    if len(results) < 2:
        return []

    pairs = []
    identities = list(set(r.get('identity', 'unknown') for r in results))

    for _ in range(n_pairs):
        # Random pair from different identities
        if len(identities) >= 2:
            id_a, id_b = random.sample(identities, 2)

            clips_a = [r for r in results if r.get('identity') == id_a]
            clips_b = [r for r in results if r.get('identity') == id_b]

            if clips_a and clips_b:
                clip_a = random.choice(clips_a)
                clip_b = random.choice(clips_b)

                pairs.append({
                    'video_a': clip_a.get('clip_name', clip_a.get('video_file', 'a')),
                    'video_b': clip_b.get('clip_name', clip_b.get('video_file', 'b')),
                    'identity_a': id_a,
                    'identity_b': id_b,
                    'position': random.choice(['A', 'B'])
                })

    return pairs


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Pipeline')
    parser.add_argument('--max', type=int, default=None,
                       help='Max videos to process')
    parser.add_argument('--pairs', type=int, default=50,
                       help='Number of pairs to generate')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip processing, just generate pairs')

    args = parser.parse_args()

    if args.skip_processing and FEATURES_FILE.exists():
        print("Loading existing features...")
        with open(FEATURES_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        # Run pipeline
        pipeline = DatasetPipeline()
        results = pipeline.run(max_videos=args.max)

    if results and args.pairs > 0:
        print(f"\n[6] Generating {args.pairs} pairs for labeling...")
        pairs = generate_labeling_pairs(results, n_pairs=args.pairs)

        pairs_file = OUTPUT_DIR / "labeling_pairs.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(pairs)} pairs to {pairs_file}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Run labeling: python src/user_labeling_ui.py")
    print(f"  2. Train model: python src/train_preference_model.py")
    print(f"  3. Validate: python src/validate_preference_model.py")


if __name__ == "__main__":
    main()
