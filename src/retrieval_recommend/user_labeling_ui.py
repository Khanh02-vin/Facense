"""
User Labeling UI - Clip Preference Collection

Cho user xem clips và label preference (1-5)

Usage:
1. Extract clips từ videos
2. User xem và rate mỗi clip
3. Save labels cho ML training
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class ClipLabel:
    """Label cho một clip."""
    clip_id: str
    rating: int  # 1-5
    confidence: int  # 1-3
    kept: bool  # True = keep, False = skip
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class UserLabelingUI:
    """
    UI cho user label clips.

    Controls:
    - 1-5: Rate clip
    - SPACE: Next clip
    - P: Previous clip
    - Q: Quit and save
    - S: Skip (no preference)
    - R: Replay clip
    """

    WINDOW_NAME = "Clip Preference Labeling"

    RATING_LABELS = {
        5: "⭐⭐⭐⭐⭐ Rất thích",
        4: "⭐⭐⭐⭐ Thích",
        3: "⭐⭐⭐ Bình thường",
        2: "⭐⭐ Chỉ muốn skip",
        1: "⭐ Rất ghét",
    }

    def __init__(self, clips_dir: str, output_path: str = "./data/user_labels.json"):
        self.clips_dir = Path(clips_dir)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.labels: List[ClipLabel] = []
        self.current_idx = 0
        self.current_rating = None

        # Find clips
        self.clips = self._find_clips()
        print(f"\n{'='*60}")
        print("USER LABELING UI")
        print(f"{'='*60}")
        print(f"Found {len(self.clips)} clips")
        print(f"Clips directory: {self.clips_dir}")
        print(f"\nControls:")
        print("  1-5: Rate clip (1=Rất ghét, 5=Rất thích)")
        print("  SPACE: Next clip")
        print("  P: Previous clip")
        print("  S: Skip (no preference)")
        print("  R: Replay clip")
        print("  Q: Quit and save")
        print(f"\nStart labeling...")

    def _find_clips(self) -> List[dict]:
        """Find all video files in clips directory."""
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        clips = []

        if not self.clips_dir.exists():
            print(f"⚠️ Directory not found: {self.clips_dir}")
            return clips

        for ext in extensions:
            for f in self.clips_dir.glob(f'*{ext}'):
                clips.append({
                    'path': str(f),
                    'id': f.stem,
                    'rated': False
                })

        return sorted(clips, key=lambda x: x['id'])

    def _load_existing_labels(self):
        """Load existing labels if file exists."""
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing = {l['clip_id']: l for l in data.get('labels', [])}
                for clip in self.clips:
                    if clip['id'] in existing:
                        clip['rated'] = True
                self.labels = list(existing.values())
                print(f"Loaded {len(self.labels)} existing labels")

    def run(self):
        """Main labeling loop."""
        if len(self.clips) == 0:
            print("❌ No clips found!")
            return

        self._load_existing_labels()

        # Create window
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)

        # Load first clip
        self._load_and_display(self.current_idx)

        # Main loop
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == ord('Q'):
                # Quit
                self._save()
                break

            elif key >= ord('1') and key <= ord('5'):
                # Rate clip
                rating = key - ord('0')
                self._rate_clip(rating)

            elif key == ord('s') or key == ord('S'):
                # Skip
                self._skip_clip()

            elif key == ord(' '):
                # Next
                if self.current_idx < len(self.clips) - 1:
                    self.current_idx += 1
                    self._load_and_display(self.current_idx)

            elif key == ord('p') or key == ord('P'):
                # Previous
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self._load_and_display(self.current_idx)

            elif key == ord('r') or key == ord('R'):
                # Replay
                self._load_and_display(self.current_idx)

        cv2.destroyAllWindows()

    def _load_and_display(self, idx: int):
        """Load and display clip."""
        clip = self.clips[idx]

        # Create info overlay
        info = self._create_overlay(clip, idx)
        self._display_frame(info)

        # Load video
        self.cap = cv2.VideoCapture(clip['path'])
        self.current_clip = clip

    def _create_overlay(self, clip: dict, idx: int) -> np.ndarray:
        """Create overlay frame with info."""
        # Black frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Title
        cv2.putText(frame, "CLIP PREFERENCE LABELING",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Clip info
        y = 150
        cv2.putText(frame, f"Clip: {clip['id']}",
                    (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        y += 50
        cv2.putText(frame, f"Progress: {idx + 1} / {len(self.clips)}",
                    (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

        if clip.get('rated'):
            y += 50
            cv2.putText(frame, "✓ Already rated",
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Instructions
        y = 400
        cv2.putText(frame, "Rate this clip:",
                    (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        y += 60
        instructions = [
            "1 = Rất ghét (Definitely skip)",
            "2 = Không thích (Skip)",
            "3 = Bình thường (Maybe)",
            "4 = Thích (Keep)",
            "5 = Rất thích (Definitely keep)",
        ]

        for i, text in enumerate(instructions, 1):
            color = (150, 150, 150)
            if self.current_rating == i:
                color = (0, 255, 0)
            cv2.putText(frame, f"  [{i}] {text}",
                        (50, y + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        # Controls
        y = 720 - 100
        cv2.putText(frame, "SPACE=Next | P=Previous | S=Skip | R=Replay | Q=Quit",
                    (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        return frame

    def _display_frame(self, frame: np.ndarray):
        """Display frame in window."""
        cv2.imshow(self.WINDOW_NAME, frame)

    def _rate_clip(self, rating: int):
        """Rate current clip."""
        clip = self.clips[self.current_idx]

        # Ask confidence
        confidence = self._ask_confidence()

        # Create label
        label = ClipLabel(
            clip_id=clip['id'],
            rating=rating,
            confidence=confidence,
            kept=rating >= 3,  # Keep if rating >= 3
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

        # Update or add
        self.labels = [l for l in self.labels if l.clip_id != clip['id']]
        self.labels.append(label)
        clip['rated'] = True

        print(f"  ✓ Rated: {clip['id']} → {rating} (kept={label.kept})")

        # Show confirmation
        confirmation = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(confirmation, f"Rated: {rating}",
                    (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        rating_text = self.RATING_LABELS.get(rating, "")
        cv2.putText(confirmation, rating_text,
                    (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow(self.WINDOW_NAME, confirmation)
        cv2.waitKey(500)

        # Auto-advance to next
        if self.current_idx < len(self.clips) - 1:
            self.current_idx += 1
            self._load_and_display(self.current_idx)

    def _skip_clip(self):
        """Skip current clip."""
        clip = self.clips[self.current_idx]

        label = ClipLabel(
            clip_id=clip['id'],
            rating=0,  # Skipped
            confidence=0,
            kept=False,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

        self.labels = [l for l in self.labels if l.clip_id != clip['id']]
        self.labels.append(label)
        clip['rated'] = True

        print(f"  ○ Skipped: {clip['id']}")

        # Auto-advance
        if self.current_idx < len(self.clips) - 1:
            self.current_idx += 1
            self._load_and_display(self.current_idx)

    def _ask_confidence(self) -> int:
        """Ask user how confident they are."""
        # Simple: 1=Not sure, 2=Somewhat, 3=Very sure
        return 2  # Default to medium

    def _save(self):
        """Save labels to file."""
        data = {
            'user_id': 'user_001',
            'total_clips': len(self.clips),
            'rated_clips': sum(1 for c in self.clips if c.get('rated')),
            'labels': [l.to_dict() for l in self.labels]
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved {len(self.labels)} labels to: {self.output_path}")
        self._print_stats()

    def _print_stats(self):
        """Print labeling statistics."""
        if not self.labels:
            return

        ratings = [l.rating for l in self.labels if l.rating > 0]
        if ratings:
            from collections import Counter
            dist = Counter(ratings)
            print(f"\n📊 Statistics:")
            print(f"   Rated: {len(ratings)}")
            print(f"   Skipped: {sum(1 for l in self.labels if l.rating == 0)}")
            print(f"   Distribution:")
            for r in range(1, 6):
                count = dist.get(r, 0)
                bar = "█" * count
                print(f"     {r}: {bar} ({count})")


# ============================================================
# CONSOLE VERSION (No GUI)
# ============================================================

class ConsoleLabeling:
    """
    Console version - không cần GUI.
    User xem clip bằng file explorer, paste path vào console.
    """

    def __init__(self, output_path: str = "./data/user_labels_console.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.labels: List[ClipLabel] = []

    def add_label(self, clip_id: str, rating: int):
        """Add a label."""
        from datetime import datetime

        label = ClipLabel(
            clip_id=clip_id,
            rating=rating,
            confidence=2,
            kept=rating >= 3,
            timestamp=datetime.now().isoformat()
        )
        self.labels.append(label)
        self._save()
        return label

    def add_batch(self, labels: List[dict]):
        """Add multiple labels at once."""
        from datetime import datetime

        for item in labels:
            label = ClipLabel(
                clip_id=item['clip_id'],
                rating=item['rating'],
                confidence=item.get('confidence', 2),
                kept=item['rating'] >= 3,
                timestamp=datetime.now().isoformat()
            )
            self.labels.append(label)

        self._save()

    def _save(self):
        """Save to file."""
        data = {
            'user_id': 'user_001',
            'total_labels': len(self.labels),
            'labels': [l.to_dict() for l in self.labels]
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def interactive_mode(self):
        """Interactive console mode."""
        print("\n" + "="*60)
        print("CONSOLE LABELING MODE")
        print("="*60)
        print("\nEnter labels one by one.")
        print("Format: clip_id,rating (e.g., video_001,5)")
        print("Type 'q' to quit and save")
        print()

        while True:
            try:
                user_input = input("> ").strip()

                if user_input.lower() == 'q':
                    break

                parts = user_input.split(',')
                if len(parts) != 2:
                    print("Format: clip_id,rating")
                    continue

                clip_id = parts[0].strip()
                try:
                    rating = int(parts[1])
                    if rating < 1 or rating > 5:
                        print("Rating must be 1-5")
                        continue
                except ValueError:
                    print("Rating must be 1-5")
                    continue

                label = self.add_label(clip_id, rating)
                print(f"  ✓ {clip_id} → {rating}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print(f"\n💾 Saved {len(self.labels)} labels")


# ============================================================
# BATCH LABELING FROM CSV
# ============================================================

def load_labels_from_csv(csv_path: str) -> List[dict]:
    """Load labels from CSV file."""
    import csv

    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                'clip_id': row['clip_id'],
                'rating': int(row['rating']),
                'confidence': int(row.get('confidence', 2))
            })
    return labels


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='User Labeling UI')
    parser.add_argument('--clips-dir', default='./data/dataset_processed/clips',
                        help='Directory containing video clips')
    parser.add_argument('--output', default='./data/user_labels.json',
                        help='Output path for labels')
    parser.add_argument('--console', action='store_true',
                        help='Use console mode instead of GUI')
    parser.add_argument('--batch', type=str,
                        help='Batch load labels from CSV file')

    args = parser.parse_args()

    if args.batch:
        # Batch mode
        labels = load_labels_from_csv(args.batch)
        collector = ConsoleLabeling(args.output)
        collector.add_batch(labels)
        print(f"Loaded {len(labels)} labels from {args.batch}")

    elif args.console:
        # Console mode
        collector = ConsoleLabeling(args.output)
        collector.interactive_mode()

    else:
        # GUI mode
        ui = UserLabelingUI(args.clips_dir, args.output)
        ui.run()


if __name__ == "__main__":
    main()
