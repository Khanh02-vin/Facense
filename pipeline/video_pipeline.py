"""
Complete Video Preference Pipeline

Video → Face Detection → Quality Filter → Temporal Aggregation → Preference Model
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.video_preference_model import (
    VideoFrameExtractor,
    CLIPVideoEncoder,
    SigLIPVideoEncoder,
    PreferenceHead,
    RewardModel,
    BradleyTerryLoss,
    PreferenceTrainer,
    PreferencePredictor,
    load_backbone
)


# ============================================================
# PIPELINE CONFIG
# ============================================================

@dataclass
class PipelineConfig:
    """Configuration for the preference pipeline."""

    # Video processing
    n_frames_per_video: int = 16
    quality_threshold: float = 0.3
    frame_size: Tuple[int, int] = (224, 224)

    # Embedding
    backbone: str = "siglip"  # "clip" or "siglip"
    embed_dim: int = 768

    # Temporal aggregation
    temporal_method: str = "transformer"  # "transformer", "attention", "weighted"

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 50

    # Paths
    video_dir: str = "./data/videos"
    processed_dir: str = "./data/processed"
    model_dir: str = "./models"


class PersonalAttractionPipeline:
    """
    Complete pipeline for personal attraction modeling.

    Usage:
        pipeline = PersonalAttractionPipeline(config)
        pipeline.process_videos()
        pipeline.train_preference_model(preference_pairs)
        predictions = pipeline.predict("video_a.mp4", "video_b.mp4")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Components
        self.frame_extractor = VideoFrameExtractor(
            n_frames=config.n_frames_per_video,
            quality_threshold=config.quality_threshold,
            target_size=config.frame_size
        )

        self.backbone = None
        self.video_encoder = None
        self.preference_model = None
        self.trainer = None

        # Cached data
        self.video_embeddings = {}
        self.video_metadata = {}

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def initialize_models(self):
        """Initialize backbone and models."""
        print(f"Initializing with {self.config.backbone} backbone...")

        # Load backbone
        self.backbone, _ = load_backbone(self.config.backbone)
        self.backbone.eval()
        self.backbone.to(self.device)

        # Create video encoder
        if self.config.backbone == "clip":
            self.video_encoder = CLIPVideoEncoder(
                embed_dim=self.config.embed_dim,
                temporal_method=self.config.temporal_method
            )
        else:
            self.video_encoder = SigLIPVideoEncoder(
                embed_dim=self.config.embed_dim,
                temporal_method=self.config.temporal_method
            )

        self.video_encoder.set_backbone(self.backbone)
        self.video_encoder.eval()
        self.video_encoder.to(self.device)

        # Preference model
        projected_dim = self.config.embed_dim // 4
        self.preference_model = PreferenceHead(embed_dim=projected_dim)
        self.preference_model.to(self.device)

        print(f"  Device: {self.device}")
        print(f"  Temporal method: {self.config.temporal_method}")
        print("Initialized!")

    # --------------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------------

    def process_videos(self, video_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Process videos and extract video-level embeddings.

        Args:
            video_paths: List of video file paths

        Returns:
            Dict mapping video_id -> video embedding
        """
        self.initialize_models()

        embeddings = {}

        for video_path in video_paths:
            video_id = Path(video_path).stem
            print(f"Processing: {video_id}")

            # Extract frames
            frames, qualities = self.frame_extractor.extract(video_path)

            if len(frames) == 0:
                print(f"  Warning: No valid frames extracted")
                continue

            # To tensor
            frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

            qual_tensor = torch.tensor([q.score() for q in qualities]).unsqueeze(0).to(self.device)

            # Extract embedding
            with torch.no_grad():
                embedding = self.video_encoder(frames_tensor, qual_tensor)
                embeddings[video_id] = embedding.cpu().numpy()[0]

            print(f"  Extracted: {embedding.shape}")

        self.video_embeddings = embeddings
        return embeddings

    def process_video_directory(self, video_dir: str) -> Dict[str, np.ndarray]:
        """Process all videos in a directory."""
        video_paths = list(Path(video_dir).glob("*.mp4"))
        return self.process_videos(video_paths)

    # --------------------------------------------------------
    # EMBEDDING OPERATIONS
    # --------------------------------------------------------

    def compute_video_similarity(self, video_a: str, video_b: str) -> float:
        """Compute cosine similarity between two videos."""
        emb_a = self.video_embeddings.get(video_a)
        emb_b = self.video_embeddings.get(video_b)

        if emb_a is None or emb_b is None:
            return 0.0

        # Cosine similarity
        sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        return float(sim)

    def find_similar_videos(self, video_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar videos to given video."""
        target_emb = self.video_embeddings.get(video_id)
        if target_emb is None:
            return []

        similarities = []
        for vid, emb in self.video_embeddings.items():
            if vid != video_id:
                sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb) + 1e-8)
                similarities.append((vid, float(sim)))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    # --------------------------------------------------------
    # PREFERENCE TRAINING
    # --------------------------------------------------------

    def train_preference_model(
        self,
        preference_pairs: List[Tuple[str, str, int]],
        val_pairs: Optional[List[Tuple[str, str, int]]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train preference model on pairwise preferences.

        Args:
            preference_pairs: List of (video_a_id, video_b_id, preference)
                            preference: 1 = prefer A, 0 = prefer B
            val_pairs: Optional validation pairs
            save_path: Path to save trained model

        Returns:
            Training metrics
        """
        if self.video_encoder is None:
            self.initialize_models()

        # Filter valid pairs (both videos must be processed)
        valid_pairs = []
        for video_a, video_b, pref in preference_pairs:
            if video_a in self.video_embeddings and video_b in self.video_embeddings:
                valid_pairs.append((video_a, video_b, pref))

        print(f"Training on {len(valid_pairs)} valid pairs")

        if len(valid_pairs) < 10:
            print("Warning: Very few training pairs, results may be unreliable")

        # Create simple dataset (using pre-computed embeddings)
        train_data = PreferencePairDataset(valid_pairs, self.video_embeddings)

        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = None
        if val_pairs:
            valid_val = [(a, b, p) for a, b, p in val_pairs
                        if a in self.video_embeddings and b in self.video_embeddings]
            if valid_val:
                val_loader = DataLoader(
                    PreferencePairDataset(valid_val, self.video_embeddings),
                    batch_size=self.config.batch_size
                )

        # Initialize trainer
        self.trainer = PreferenceTrainer(
            self.preference_model,
            self.video_encoder,
            lr=self.config.lr,
            device=self.device
        )

        # Train
        best_val_acc = 0
        metrics = {"train_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for batch in train_loader:
                loss = self.trainer.train_step(batch)
                epoch_loss += loss

            # Evaluate
            train_metrics = self.trainer.evaluate(train_loader)
            metrics["train_loss"].append(train_metrics["loss"])
            metrics["train_acc"].append(train_metrics["accuracy"])

            if val_loader:
                val_metrics = self.trainer.evaluate(val_loader)
                metrics["val_acc"].append(val_metrics["accuracy"])
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2%}")

        # Save model
        if save_path:
            self.save_model(save_path)

        return metrics

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict_preference(self, video_a: str, video_b: str) -> Tuple[float, str]:
        """
        Predict preference between two videos.

        Returns:
            (score, explanation)
            score > 0.5 means prefer A, score < 0.5 means prefer B
        """
        if self.trainer is None:
            return 0.5, "Model not trained"

        emb_a = self.video_embeddings.get(video_a)
        emb_b = self.video_embeddings.get(video_b)

        if emb_a is None or emb_b is None:
            return 0.5, "Video not in database"

        # To tensors
        emb_a_t = torch.from_numpy(emb_a).float().unsqueeze(0).to(self.device)
        emb_b_t = torch.from_numpy(emb_b).float().unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            score = torch.sigmoid(self.preference_model(emb_a_t, emb_b_t)).item()

        # Explanation
        if score > 0.6:
            explanation = f"Strongly prefer A (score={score:.2f})"
        elif score > 0.4:
            explanation = f"Slightly prefer A (score={score:.2f})"
        else:
            explanation = f"Prefer B (score={score:.2f})"

        return score, explanation

    def rank_videos(self, target_video: str, candidate_videos: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidate videos by preference relative to target.

        Returns:
            List of (video_id, preference_score) sorted by preference
        """
        rankings = []
        for video in candidate_videos:
            score, _ = self.predict_preference(target_video, video)
            rankings.append((video, score))

        rankings.sort(key=lambda x: -x[1])
        return rankings

    # --------------------------------------------------------
    # PREFERENCE FROM SIMILARITY
    # --------------------------------------------------------

    def preference_from_similarity(
        self,
        video_embeddings: Dict[str, np.ndarray],
        preference_pairs: List[Tuple[str, str, int]]
    ) -> float:
        """
        Compute how well embedding similarity matches preferences.

        Returns:
            Accuracy of using cosine similarity for preference prediction
        """
        correct = 0
        total = 0

        for video_a, video_b, pref in preference_pairs:
            if video_a not in video_embeddings or video_b not in video_embeddings:
                continue

            emb_a = video_embeddings[video_a]
            emb_b = video_embeddings[video_b]

            # Cosine similarity
            sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)

            # Predict: if sim(A,B) > threshold, prefer A
            # For equal similarity, predict randomly
            if sim > 0.5 and pref == 1:
                correct += 1
            elif sim < 0.5 and pref == 0:
                correct += 1
            elif abs(sim - 0.5) < 0.1:
                correct += 0.5  # Partial credit for close calls

            total += 1

        if total == 0:
            return 0.5

        return correct / total

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    def save_model(self, path: str):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "preference_model": self.preference_model.state_dict(),
            "video_encoder": self.video_encoder.state_dict(),
            "config": self.config.__dict__,
            "video_embeddings": self.video_embeddings,
            "video_metadata": self.video_metadata
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.config = PipelineConfig(**checkpoint["config"])
        self.video_embeddings = checkpoint["video_embeddings"]
        self.video_metadata = checkpoint["video_metadata"]

        self.initialize_models()
        self.preference_model.load_state_dict(checkpoint["preference_model"])
        self.video_encoder.load_state_dict(checkpoint["video_encoder"])

        print(f"Model loaded from {path}")


# ============================================================
# SIMPLE DATASET FOR PREFERENCE PAIRS
# ============================================================

class PreferencePairDataset(Dataset):
    """Dataset from pre-computed embeddings."""

    def __init__(self, pairs: List[Tuple[str, str, int]], embeddings: Dict[str, np.ndarray]):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_a, video_b, pref = self.pairs[idx]

        emb_a = torch.from_numpy(self.embeddings[video_a]).float()
        emb_b = torch.from_numpy(self.embeddings[video_b]).float()

        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "preference": torch.tensor(pref, dtype=torch.float32)
        }


def preference_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function."""
    return {
        "emb_a": torch.stack([b["emb_a"] for b in batch]),
        "emb_b": torch.stack([b["emb_b"] for b in batch]),
        "preference": torch.stack([b["preference"] for b in batch])
    }


# ============================================================
# USAGE EXAMPLE
# ============================================================

def main():
    """Example usage."""
    # Config
    config = PipelineConfig(
        backbone="siglip",
        temporal_method="transformer",
        n_frames_per_video=16,
        epochs=50
    )

    # Initialize pipeline
    pipeline = PersonalAttractionPipeline(config)

    # Process videos
    embeddings = pipeline.process_video_directory("./data/videos")
    print(f"Processed {len(embeddings)} videos")

    # Define preference pairs (from annotations)
    preference_pairs = [
        ("video_a", "video_b", 1),  # prefer A
        ("video_c", "video_d", 0),  # prefer B
        # ... from annotations_result.json
    ]

    # Train
    metrics = pipeline.train_preference_model(preference_pairs)
    print(f"Final accuracy: {metrics['train_acc'][-1]:.2%}")

    # Predict
    score, explanation = pipeline.predict_preference("new_video_a", "new_video_b")
    print(explanation)

    # Save
    pipeline.save_model("./models/preference_model.pt")


if __name__ == "__main__":
    main()
