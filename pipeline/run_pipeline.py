"""
Run Video Preference Pipeline

Usage:
    python run_pipeline.py --process-videos
    python run_pipeline.py --train
    python run_pipeline.py --predict --video-a "video1.mp4" --video-b "video2.mp4"
"""

import argparse
import json
from pathlib import Path

from pipeline.video_pipeline import PersonalAttractionPipeline, PipelineConfig


def load_annotations_to_pairs():
    """Convert annotations to preference pairs."""
    annotations_path = "./data/annotations/annotations_result.json"

    if not Path(annotations_path).exists():
        print(f"Annotations not found: {annotations_path}")
        return []

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    pairs = []
    for ann in annotations:
        if ann['choice'] == 'A':
            pairs.append((ann['identity_A'], ann['identity_B'], 1))
        elif ann['choice'] == 'B':
            pairs.append((ann['identity_B'], ann['identity_A'], 1))
        # Skip 'equal' and 'skip'

    print(f"Loaded {len(pairs)} preference pairs from annotations")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Video Preference Pipeline")
    parser.add_argument("--process-videos", action="store_true", help="Process videos")
    parser.add_argument("--train", action="store_true", help="Train preference model")
    parser.add_argument("--predict", action="store_true", help="Predict preference")
    parser.add_argument("--video-a", type=str, help="Video A path")
    parser.add_argument("--video-b", type=str, help="Video B path")
    parser.add_argument("--backbone", type=str, default="siglip", choices=["clip", "siglip"])
    parser.add_argument("--temporal", type=str, default="transformer", choices=["transformer", "attention", "weighted"])

    args = parser.parse_args()

    # Config
    config = PipelineConfig(
        backbone=args.backbone,
        temporal_method=args.temporal,
        n_frames_per_video=16,
        epochs=50
    )

    # Initialize pipeline
    pipeline = PersonalAttractionPipeline(config)

    if args.process_videos:
        print("="*50)
        print("PROCESSING VIDEOS")
        print("="*50)

        # Process videos from dataset
        video_dir = "./data/processed/frames"
        if Path(video_dir).exists():
            embeddings = pipeline.process_video_directory(video_dir)
            print(f"\nProcessed {len(embeddings)} videos")
        else:
            print(f"Video directory not found: {video_dir}")
            print("Creating sample embeddings from existing data...")

            # Use existing embeddings as fallback
            import numpy as np
            embeddings_file = "./data/processed/embeddings_siglip_multiframe.npy"
            identity_file = "./data/processed/image_to_identity_multiframe.json"

            if Path(embeddings_file).exists():
                embeddings = np.load(embeddings_file)
                with open(identity_file, 'r', encoding='utf-8') as f:
                    identity_map = json.load(f)

                print(f"Loaded {len(embeddings)} pre-computed embeddings")

    if args.train:
        print("="*50)
        print("TRAINING PREFERENCE MODEL")
        print("="*50)

        # Load preference pairs from annotations
        pairs = load_annotations_to_pairs()

        if len(pairs) == 0:
            print("No preference pairs available")
            return

        # Split into train/val
        import random
        random.seed(42)
        random.shuffle(pairs)
        split = int(0.8 * len(pairs))
        train_pairs = pairs[:split]
        val_pairs = pairs[split:]

        print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        # Initialize (use pre-computed embeddings)
        pipeline.initialize_models()

        # Train
        metrics = pipeline.train_preference_model(
            train_pairs,
            val_pairs,
            save_path="./models/preference_model.pt"
        )

        print("\nFinal metrics:")
        print(f"  Train loss: {metrics['train_loss'][-1]:.4f}")
        print(f"  Train acc:  {metrics['train_acc'][-1]:.2%}")
        if 'val_acc' in metrics and metrics['val_acc']:
            print(f"  Val acc:    {metrics['val_acc'][-1]:.2%}")

    if args.predict:
        print("="*50)
        print("PREDICTION")
        print("="*50)

        if args.video_a and args.video_b:
            score, explanation = pipeline.predict_preference(args.video_a, args.video_b)
            print(f"\n{explanation}")
        else:
            print("Please provide --video-a and --video-b")


if __name__ == "__main__":
    main()
