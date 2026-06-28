"""
Video Frame Extraction and Embedding Pipeline

Extracts frames from videos and generates appearance embeddings.
Usage:
    python -m evaluation.extract_and_embed --dataset_dir "D:/Dataset/Face_project_datset" --output_dir "./data/embeddings"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import numpy as np
from PIL import Image
import cv2

# Fix encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


@dataclass
class VideoMetadata:
    """Metadata for a processed video."""
    video_path: str
    identity: str
    n_frames_extracted: int
    frame_paths: list[str]
    embeddings_path: str
    status: str  # 'success', 'failed', 'partial'


@dataclass
class IdentityGroup:
    """Group of images belonging to one identity."""
    identity_id: str
    image_ids: list[str]
    frame_paths: list[str]
    embedding_paths: list[str]


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    identity: str,
    max_frames: int = 10,
    fps: float = 1.0
) -> list[str]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        identity: Identity/celebrity name
        max_frames: Maximum frames to extract
        fps: Frames per second to extract

    Returns:
        List of extracted frame paths
    """
    frame_paths = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / fps))

        frame_count = 0
        saved_count = 0

        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Save frame
                frame_filename = f"{identity}_{saved_count:03d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(frame_path, quality=85)

                frame_paths.append(frame_path)
                saved_count += 1

            frame_count += 1

        cap.release()

    except Exception as e:
        print(f"[ERROR] Failed to extract from {video_path}: {e}")
        return []

    return frame_paths


def process_identity_folder(
    identity_folder: str,
    output_base: str,
    identity_name: str,
    max_frames_per_video: int = 5,
    fps: float = 1.0
) -> IdentityGroup:
    """Process all videos for one identity.

    Args:
        identity_folder: Path to folder containing videos
        output_base: Base output directory
        identity_name: Name of the identity
        max_frames_per_video: Max frames per video
        fps: FPS for frame extraction

    Returns:
        IdentityGroup with processed data
    """
    output_dir = os.path.join(output_base, "frames", identity_name)
    os.makedirs(output_dir, exist_ok=True)

    all_frames = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    video_files = [
        f for f in os.listdir(identity_folder)
        if Path(f).suffix.lower() in video_extensions
    ]

    for video_file in video_files:
        video_path = os.path.join(identity_folder, video_file)

        frames = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            identity=identity_name,
            max_frames=max_frames_per_video,
            fps=fps
        )

        all_frames.extend(frames)

    return IdentityGroup(
        identity_id=identity_name,
        image_ids=[os.path.basename(f).replace('.jpg', '') for f in all_frames],
        frame_paths=all_frames,
        embedding_paths=[]
    )


def generate_embeddings_batch(
    frame_paths: list[str],
    model_name: Literal["siglip", "dinov2", "clip"] = "dinov2",
    batch_size: int = 8
) -> dict[str, np.ndarray]:
    """Generate embeddings for frames using specified model.

    Args:
        frame_paths: List of frame image paths
        model_name: Model to use ('dinov2', 'siglip', 'clip')
        batch_size: Batch size for processing

    Returns:
        Dict mapping image_id -> embedding array
    """
    embeddings = {}

    try:
        import torch
        from torchvision import transforms
    except ImportError:
        print("[ERROR] torch/torchvision required for embedding generation")
        return embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    if model_name == "dinov2":
        print("[INFO] Loading DINOv2...")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.to(device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print(f"[INFO] Model {model_name} requires transformers - using DINOv2 fallback")
        print("[INFO] Loading DINOv2...")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.to(device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Process in batches
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
                continue

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            batch_embeddings = model(batch)

        # Normalize
        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)

        # Store
        for j, path in enumerate(batch_paths):
            image_id = Path(path).stem
            embeddings[image_id] = batch_embeddings[j].cpu().numpy()

        print(f"[INFO] Processed {min(i + batch_size, len(frame_paths))}/{len(frame_paths)} frames")

    return embeddings


def build_identity_groups(
    dataset_dir: str,
    identity_groups: dict[str, IdentityGroup]
) -> dict[str, list[str]]:
    """Build mapping from image_id to identity for identity control.

    Returns:
        Dict mapping image_id -> identity_id
    """
    mapping = {}
    for identity_id, group in identity_groups.items():
        for image_id in group.image_ids:
            mapping[image_id] = identity_id
    return mapping


def save_embeddings_and_metadata(
    embeddings: dict[str, np.ndarray],
    identity_mapping: dict[str, str],
    output_dir: str
):
    """Save embeddings and metadata to files.

    Args:
        embeddings: Dict of image_id -> embedding
        identity_mapping: Dict of image_id -> identity_id
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as numpy
    image_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[iid] for iid in image_ids])

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embedding_matrix)

    # Save mapping
    mapping_path = os.path.join(output_dir, "image_to_embedding_idx.json")
    image_to_idx = {iid: idx for idx, iid in enumerate(image_ids)}
    with open(mapping_path, 'w') as f:
        json.dump(image_to_idx, f, indent=2)

    # Save identity mapping
    identity_path = os.path.join(output_dir, "image_to_identity.json")
    with open(identity_path, 'w', encoding='utf-8') as f:
        json.dump(identity_mapping, f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata = {
        "n_images": len(image_ids),
        "embedding_dim": embedding_matrix.shape[1] if len(embedding_matrix) > 0 else 0,
        "n_identities": len(set(identity_mapping.values())),
        "model": "dinov2_vitb14"
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved embeddings to {embeddings_path}")
    print(f"[INFO] Saved metadata: {metadata}")


def run_pipeline(
    dataset_dir: str,
    output_dir: str,
    max_frames_per_video: int = 5,
    fps: float = 1.0,
    model_name: str = "dinov2",
    batch_size: int = 8
):
    """Run full extraction and embedding pipeline.

    Args:
        dataset_dir: Path to dataset directory
        output_dir: Path to output directory
        max_frames_per_video: Max frames per video
        fps: FPS for extraction
        model_name: Embedding model name
        batch_size: Batch size for embedding
    """
    print("=" * 60)
    print("Video Frame Extraction and Embedding Pipeline")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print()

    dataset_path = Path(dataset_dir)
    identity_folders = [d for d in dataset_path.iterdir() if d.is_dir()]

    print(f"[INFO] Found {len(identity_folders)} identities")
    print()

    identity_groups = {}
    all_frame_paths = []
    total_frames = 0

    # Process each identity
    for i, identity_folder in enumerate(identity_folders):
        identity_name = identity_folder.name
        print(f"[{i+1}/{len(identity_folders)}] Processing: {identity_name}")

        group = process_identity_folder(
            identity_folder=str(identity_folder),
            output_base=output_dir,
            identity_name=identity_name,
            max_frames_per_video=max_frames_per_video,
            fps=fps
        )

        identity_groups[identity_name] = group
        all_frame_paths.extend(group.frame_paths)
        total_frames += len(group.frame_paths)

        print(f"  -> Extracted {len(group.frame_paths)} frames")

    print()
    print(f"[INFO] Total: {len(identity_groups)} identities, {total_frames} frames")
    print()

    # Generate embeddings
    if all_frame_paths:
        print("[INFO] Generating embeddings...")
        embeddings = generate_embeddings_batch(
            frame_paths=all_frame_paths,
            model_name=model_name,
            batch_size=batch_size
        )
        print()
    else:
        print("[WARN] No frames extracted!")
        return

    # Build identity mapping
    identity_mapping = build_identity_groups(dataset_dir, identity_groups)

    # Save
    print("[INFO] Saving embeddings and metadata...")
    save_embeddings_and_metadata(
        embeddings=embeddings,
        identity_mapping=identity_mapping,
        output_dir=output_dir
    )

    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return {
        "n_identities": len(identity_groups),
        "n_frames": total_frames,
        "n_embeddings": len(embeddings)
    }


def main():
    parser = argparse.ArgumentParser(description="Extract frames and generate embeddings")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="D:/Dataset/Face_project_datset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Path to output directory"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=5,
        help="Max frames per video"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2",
        choices=["dinov2", "siglip", "clip"],
        help="Embedding model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding"
    )

    args = parser.parse_args()

    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_frames_per_video=args.max_frames,
        fps=args.fps,
        model_name=args.model,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
