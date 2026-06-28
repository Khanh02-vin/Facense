"""
Deep Embedding Extraction Script

Extracts embeddings using CLIP, SigLIP, DINOv2.
Run inside Docker container.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
import timm


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


def load_video_dataset(dataset_dir, max_per_identity=1):
    """Load frames from video dataset."""
    dataset_dir = Path(dataset_dir)
    frames = []
    identities = []

    identity_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        video_files = list(identity_dir.glob("*.mp4"))[:max_per_identity]

        for video_file in video_files:
            extracted_frames = extract_frames_from_video(str(video_file), n_frames=1)
            for frame in extracted_frames:
                frames.append(frame)
                identities.append(identity)

    return frames, identities


# CLIP Model
class CLIPEmbedding:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image):
        """Extract CLIP embedding from image."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs[0].numpy()


# SigLIP Model
class SigLIPEmbedding:
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        print(f"Loading SigLIP model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image):
        """Extract SigLIP embedding from image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs[0].numpy()


# DINOv2 Model
class DINOv2Embedding:
    def __init__(self, model_name="vit_base_patch14_dinov2"):
        print(f"Loading DINOv2 model: {model_name}")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.data_config = timm.data.resolve_model_data_config(self.model)

    @torch.no_grad()
    def extract(self, image):
        """Extract DINOv2 embedding from image."""
        from PIL import Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        transform = timm.data.create_transform(**self.data_config, is_training=False)
        input_tensor = transform(pil_image).unsqueeze(0)

        outputs = self.model(input_tensor)
        return outputs[0].numpy()


def run_extraction(
    dataset_dir="/data/Face_project_datset",
    output_dir="/workspace/data/processed",
    models=["clip", "siglip", "dinov2"],
    max_per_identity=1
):
    """Run embedding extraction for all models."""
    print("=" * 60)
    print("Deep Embedding Extraction")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load frames
    print("\n[1] Loading video frames...")
    frames, identities = load_video_dataset(dataset_dir, max_per_identity)
    print(f"    Loaded {len(frames)} frames from {len(set(identities))} identities")

    # Extract for each model
    for model_name in models:
        print(f"\n[2] Extracting {model_name.upper()} embeddings...")

        # Initialize model
        if model_name == "clip":
            extractor = CLIPEmbedding()
        elif model_name == "siglip":
            extractor = SigLIPEmbedding()
        elif model_name == "dinov2":
            extractor = DINOv2Embedding()
        else:
            continue

        # Extract embeddings
        embeddings = []
        for i, frame in enumerate(tqdm(frames, desc=f"{model_name}")):
            emb = extractor.extract(frame)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        # Save embeddings
        output_file = os.path.join(output_dir, f"embeddings_{model_name}.npy")
        np.save(output_file, embeddings)
        print(f"    Saved to: {output_file}")
        print(f"    Shape: {embeddings.shape}")

        # Save metadata
        metadata = {
            "model": model_name,
            "n_images": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "n_identities": len(set(identities)),
            "identities": list(set(identities))
        }

        meta_file = os.path.join(output_dir, f"metadata_{model_name}.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save identity mapping
        identity_map = {i: ident for i, ident in enumerate(identities)}
        map_file = os.path.join(output_dir, f"image_to_identity_{model_name}.json")
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(identity_map, f, indent=2, ensure_ascii=False)

        print(f"    Done: {model_name}")

    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract deep embeddings")
    parser.add_argument("--dataset_dir", type=str, default="/data/Face_project_datset")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/processed")
    parser.add_argument("--models", nargs="+", default=["clip", "siglip", "dinov2"])
    parser.add_argument("--max_per_identity", type=int, default=1)

    args = parser.parse_args()

    run_extraction(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        models=args.models,
        max_per_identity=args.max_per_identity
    )
