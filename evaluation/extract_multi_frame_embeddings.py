"""
Extract embeddings for multiple frames per video.
This allows same-identity pair generation.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModel


def extract_frames_from_video(video_path, n_frames=5):
    """Extract frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def load_frames_and_extract(dataset_dir, extractor, n_frames=3):
    """Load frames from videos and extract embeddings."""
    dataset_dir = Path(dataset_dir)

    all_data = []  # (frame_id, identity, embedding)

    identity_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    print(f"Processing {len(identity_dirs)} identities...")

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        video_files = list(identity_dir.glob("*.mp4"))[:1]  # One video

        for video_file in video_files:
            frames = extract_frames_from_video(str(video_file), n_frames=n_frames)

            for i, frame in enumerate(frames):
                frame_id = f"{identity}_{i}"
                emb = extractor.extract(frame)
                all_data.append({
                    'frame_id': frame_id,
                    'identity': identity,
                    'embedding': emb
                })

    return all_data


def main():
    print("=" * 60)
    print("Extracting Multi-Frame SigLIP Embeddings")
    print("=" * 60)

    dataset_dir = "D:/Dataset/Face_project_datset"
    output_dir = "./data/processed"
    n_frames = 3

    # Load SigLIP
    print("\n[1] Loading SigLIP model...")
    from transformers import AutoProcessor, AutoModel

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    class Extractor:
        def __init__(self, processor, model):
            self.processor = processor
            self.model = model

        @torch.no_grad()
        def extract(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            outputs = self.model.get_image_features(**inputs)
            return outputs[0].numpy()

    extractor = Extractor(processor, model)

    # Extract
    print(f"\n[2] Extracting {n_frames} frames per identity...")
    all_data = load_frames_and_extract(dataset_dir, extractor, n_frames=n_frames)

    print(f"\n    Total frames: {len(all_data)}")

    # Stack embeddings
    embeddings = np.array([d['embedding'] for d in all_data])
    print(f"    Embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_file = os.path.join(output_dir, "embeddings_siglip_multiframe.npy")
    np.save(embeddings_file, embeddings)
    print(f"    Saved to: {embeddings_file}")

    # Save metadata
    metadata = {
        'n_frames_per_video': n_frames,
        'n_identities': len(set(d['identity'] for d in all_data)),
        'n_total': len(all_data),
        'embedding_dim': embeddings.shape[1]
    }

    meta_file = os.path.join(output_dir, "metadata_siglip_multiframe.json")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Metadata saved to: {meta_file}")

    # Save frame-to-identity mapping
    frame_map = {i: d['frame_id'] for i, d in enumerate(all_data)}
    identity_map = {i: d['identity'] for i, d in enumerate(all_data)}

    with open(os.path.join(output_dir, "frame_to_identity.json"), 'w', encoding='utf-8') as f:
        json.dump(frame_map, f, indent=2)

    with open(os.path.join(output_dir, "image_to_identity_multiframe.json"), 'w', encoding='utf-8') as f:
        json.dump(identity_map, f, indent=2)

    # Count same-identity pairs
    same_identity_count = sum(
        1 for i in range(len(all_data))
        for j in range(i + 1, len(all_data))
        if all_data[i]['identity'] == all_data[j]['identity']
    )

    print(f"\n[3] Same-identity pairs available: {same_identity_count}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
