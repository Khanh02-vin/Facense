"""
Personal Attraction Model - Complete Architecture

Video → Face Detection → Quality Filter → Temporal Aggregation → Preference Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
import cv2


# ============================================================
# PART 1: VIDEO PROCESSING PIPELINE
# ============================================================

@dataclass
class FrameQuality:
    """Quality metrics for a single frame."""
    blur_score: float      # Laplacian variance
    face_size_ratio: float  # face bbox / image size
    face_clarity: float    # detection confidence
    brightness: float      # average brightness
    occlusion_score: float # estimated occlusion

    def score(self) -> float:
        """Combined quality score (0-1)."""
        return (
            0.3 * self.blur_score +
            0.3 * self.face_size_ratio +
            0.2 * self.face_clarity +
            0.1 * self.brightness +
            0.1 * self.occlusion_score
        )


class VideoFrameExtractor:
    """Extract high-quality face frames from video."""

    def __init__(
        self,
        n_frames: int = 16,
        quality_threshold: float = 0.3,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.n_frames = n_frames
        self.quality_threshold = quality_threshold
        self.target_size = target_size

    def extract(self, video_path: str) -> Tuple[List[np.ndarray], List[FrameQuality]]:
        """Extract quality-filtered frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return [], []

        # Sample candidate frames
        candidate_indices = np.linspace(0, total_frames - 1, self.n_frames * 2, dtype=int)
        candidates = []

        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                quality = self._assess_quality(frame)
                if quality.score() >= self.quality_threshold:
                    candidates.append((quality.score(), frame, quality))

        cap.release()

        # Sort by quality and take top n_frames
        candidates.sort(key=lambda x: -x[0])
        frames = [cv2.resize(c[1], self.target_size) for c in candidates[:self.n_frames]]
        qualities = [c[2] for c in candidates[:self.n_frames]]

        return frames, qualities

    def _assess_quality(self, frame: np.ndarray) -> FrameQuality:
        """Assess frame quality without face detection (simplified)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur score (Laplacian variance)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(blur / 1000.0, 1.0)  # Normalize

        # Brightness
        brightness = np.mean(gray) / 255.0

        # Simple heuristic: face in center is more likely good
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_score = np.mean(center_region) / 255.0

        return FrameQuality(
            blur_score=blur_score,
            face_size_ratio=0.7,  # Placeholder
            face_clarity=center_score,
            brightness=brightness,
            occlusion_score=0.9  # Placeholder
        )


# ============================================================
# PART 2: TEMPORAL AGGREGATION MODELS
# ============================================================

class TemporalAttention(nn.Module):
    """Attention-based temporal aggregation."""

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequence of embeddings
        Returns:
            (B, D) aggregated representation
        """
        # Self-attention
        attn_out, weights = self.attention(x, x, x)

        # Residual + norm
        out = self.norm(attn_out + x)

        # Weighted average with attention weights
        # weights: (B, T, T) after attention
        weights = weights.mean(dim=1)  # (B, T, T) -> (B, T)
        weights = F.softmax(weights, dim=1)

        # Aggregate
        aggregated = (out * weights.unsqueeze(-1)).sum(dim=1)
        return aggregated


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal aggregation."""

    def __init__(self, embed_dim: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequence
        Returns:
            (B, D) video representation
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer layers
        x = self.transformer(x)

        # Global average pooling + residual
        pooled = x.mean(dim=1)
        return self.norm(pooled + x[:, 0])


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class WeightedPooling(nn.Module):
    """Simple weighted pooling with quality scores."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.weights = nn.Linear(embed_dim + 1, 1)  # embed + quality score

    def forward(self, embeddings: torch.Tensor, quality_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, T, D)
            quality_scores: (B, T) quality scores
        Returns:
            (B, D) aggregated
        """
        # Normalize quality scores
        q = quality_scores.unsqueeze(-1)
        q = (q - q.mean(dim=1, keepdim=True)) / (q.std(dim=1, keepdim=True) + 1e-8)
        q = torch.sigmoid(q)

        # Weighted sum
        weights = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        return (embeddings * weights).sum(dim=1)


# ============================================================
# PART 3: VIDEO ENCODER
# ============================================================

class VideoEncoder(nn.Module):
    """Complete video encoder with temporal aggregation."""

    def __init__(
        self,
        backbone: str = "clip",  # "clip", "siglip", "arcface"
        embed_dim: int = 768,
        temporal_method: str = "transformer",  # "attention", "transformer", "weighted"
        n_temporal_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.temporal_method = temporal_method

        # Backbone (loaded separately - see BackboneLoader)
        self.backbone = None

        # Temporal aggregation
        if temporal_method == "transformer":
            self.temporal = TemporalTransformer(embed_dim, n_temporal_layers, n_heads)
        elif temporal_method == "attention":
            self.temporal = TemporalAttention(embed_dim, n_heads)
        elif temporal_method == "weighted":
            self.temporal = WeightedPooling(embed_dim)
        else:
            raise ValueError(f"Unknown temporal method: {temporal_method}")

        # Projection to preference space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
        )

    def set_backbone(self, backbone: nn.Module):
        """Set the image backbone (CLIP/SigLIP/etc)."""
        self.backbone = backbone

    def forward(
        self,
        frames: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            quality_scores: (B, T) optional quality scores
        Returns:
            (B, D) video representation
        """
        B, T, C, H, W = frames.shape

        # Extract frame embeddings
        frames_flat = frames.view(B * T, C, H, W)
        with torch.no_grad():
            frame_embeddings = self._extract_features(frames_flat)

        frame_embeddings = frame_embeddings.view(B, T, -1)

        # Temporal aggregation
        if self.temporal_method == "weighted" and quality_scores is not None:
            video_repr = self.temporal(frame_embeddings, quality_scores)
        else:
            video_repr = self.temporal(frame_embeddings)

        # Project
        return self.projection(video_repr)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using backbone. Override in subclass."""
        raise NotImplementedError


class CLIPVideoEncoder(VideoEncoder):
    """Video encoder using CLIP backbone."""

    def __init__(self, embed_dim: int = 768, temporal_method: str = "transformer"):
        super().__init__("clip", embed_dim, temporal_method)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features."""
        if self.backbone is None:
            raise ValueError("Backbone not set. Call set_backbone() first.")
        return self.backbone.get_image_features(images)


class SigLIPVideoEncoder(VideoEncoder):
    """Video encoder using SigLIP backbone."""

    def __init__(self, embed_dim: int = 768, temporal_method: str = "transformer"):
        super().__init__("siglip", embed_dim, temporal_method)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SigLIP features."""
        if self.backbone is None:
            raise ValueError("Backbone not set. Call set_backbone() first.")
        return self.backbone.get_image_features(images)


# ============================================================
# PART 4: PREFERENCE MODEL
# ============================================================

class PreferenceHead(nn.Module):
    """Predict preference from video representations."""

    def __init__(
        self,
        embed_dim: int = 192,  # projection output
        hidden_dim: int = 128,
        n_classes: int = 1  # 1 for regression, 2+ for classification
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, video_a: torch.Tensor, video_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_a: (B, D) video A representation
            video_b: (B, D) video B representation
        Returns:
            (B,) preference logits (positive = prefer A)
        """
        combined = torch.cat([video_a, video_b], dim=1)
        return self.net(combined).squeeze(-1)


class RewardModel(nn.Module):
    """Reward model that scores individual videos."""

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, video_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_repr: (B, D) video representation
        Returns:
            (B,) reward scores
        """
        return self.net(video_repr).squeeze(-1)


class BradleyTerryLoss(nn.Module):
    """Bradley-Terry loss for preference learning."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        reward_a: torch.Tensor,
        reward_b: torch.Tensor,
        preference: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            reward_a: (B,) reward for video A
            reward_b: (B,) reward for video B
            preference: (B,) 1 if prefer A, 0 if prefer B
        Returns:
            scalar loss
        """
        # P(B > A) = sigmoid(reward_B - reward_A)
        logits = reward_b - reward_a

        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(logits, preference)
        return loss


# ============================================================
# PART 5: DATASET
# ============================================================

class VideoPreferenceDataset(Dataset):
    """Dataset for video preference learning."""

    def __init__(
        self,
        video_pairs: List[Tuple[str, str, int]],  # (video_a, video_b, preference)
        frame_extractor: VideoFrameExtractor,
        transform: Optional[callable] = None
    ):
        """
        Args:
            video_pairs: List of (video_a_path, video_b_path, preference)
                        preference: 1 = prefer A, 0 = prefer B
            frame_extractor: VideoFrameExtractor instance
            transform: Optional transform for frames
        """
        self.video_pairs = video_pairs
        self.frame_extractor = frame_extractor
        self.transform = transform

        # Cache extracted frames
        self.cache = {}

    def __len__(self) -> int:
        return len(self.video_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_a, video_b, pref = self.video_pairs[idx]

        # Extract frames
        frames_a, qual_a = self._get_frames(video_a)
        frames_b, qual_b = self._get_frames(video_b)

        # To tensors
        frames_a = torch.from_numpy(np.stack(frames_a)).float() / 255.0
        frames_b = torch.from_numpy(np.stack(frames_b)).float() / 255.0

        qual_a = torch.tensor([q.score() for q in qual_a])
        qual_b = torch.tensor([q.score() for q in qual_b])

        return {
            'frames_a': frames_a,
            'frames_b': frames_b,
            'quality_a': qual_a,
            'quality_b': qual_b,
            'preference': torch.tensor(pref, dtype=torch.float32)
        }

    def _get_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[FrameQuality]]:
        if video_path not in self.cache:
            frames, qualities = self.frame_extractor.extract(video_path)
            self.cache[video_path] = (frames, qualities)
        return self.cache[video_path]


@dataclass
class Batch:
    """Collated batch."""
    frames_a: torch.Tensor      # (B, T, 3, H, W)
    frames_b: torch.Tensor      # (B, T, 3, H, W)
    quality_a: torch.Tensor     # (B, T)
    quality_b: torch.Tensor     # (B, T)
    preference: torch.Tensor    # (B,)


def collate_fn(batch: List[Dict]) -> Batch:
    """Custom collate function."""
    return Batch(
        frames_a=torch.stack([b['frames_a'] for b in batch]),
        frames_b=torch.stack([b['frames_b'] for b in batch]),
        quality_a=torch.stack([b['quality_a'] for b in batch]),
        quality_b=torch.stack([b['quality_b'] for b in batch]),
        preference=torch.stack([b['preference'] for b in batch])
    )


# ============================================================
# PART 6: TRAINING
# ============================================================

class PreferenceTrainer:
    """Trainer for preference model."""

    def __init__(
        self,
        model: PreferenceHead,
        video_encoder: VideoEncoder,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.video_encoder = video_encoder
        self.device = device

        self.model.to(device)
        self.video_encoder.to(device)

        # Loss
        self.criterion = BradleyTerryLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.video_encoder.parameters()),
            lr=lr,
            weight_decay=0.01
        )

    def train_step(self, batch: Batch) -> float:
        """Single training step."""
        frames_a = batch.frames_a.to(self.device)
        frames_b = batch.frames_b.to(self.device)
        qual_a = batch.quality_a.to(self.device)
        qual_b = batch.quality_b.to(self.device)
        pref = batch.preference.to(self.device)

        # Extract video representations
        repr_a = self.video_encoder(frames_a, qual_a)
        repr_b = self.video_encoder(frames_b, qual_b)

        # Predict preference
        logits = self.model(repr_a, repr_b)

        # Loss
        loss = self.criterion(
            repr_a.sum(dim=1),  # Use raw reward
            repr_b.sum(dim=1),
            pref
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                frames_a = batch.frames_a.to(self.device)
                frames_b = batch.frames_b.to(self.device)
                qual_a = batch.quality_a.to(self.device)
                qual_b = batch.quality_b.to(self.device)
                pref = batch.preference.to(self.device)

                repr_a = self.video_encoder(frames_a, qual_a)
                repr_b = self.video_encoder(frames_b, qual_b)

                loss = self.criterion(
                    repr_a.sum(dim=1),
                    repr_b.sum(dim=1),
                    pref
                )
                total_loss += loss.item()

                # Accuracy
                logits = self.model(repr_a, repr_b)
                preds = (logits > 0).float()
                correct += (preds == pref).sum().item()
                total += pref.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }


# ============================================================
# PART 7: INFERENCE
# ============================================================

class PreferencePredictor:
    """Predict preferences for new videos."""

    def __init__(
        self,
        video_encoder: VideoEncoder,
        model: PreferenceHead,
        frame_extractor: VideoFrameExtractor
    ):
        self.video_encoder = video_encoder
        self.model = model
        self.frame_extractor = frame_extractor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.video_encoder.to(self.device).eval()
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_preference(
        self,
        video_a: str,
        video_b: str
    ) -> Tuple[float, str]:
        """
        Predict which video is preferred.

        Returns:
            (score, explanation)
        """
        # Extract frames
        frames_a, qual_a = self.frame_extractor.extract(video_a)
        frames_b, qual_b = self.frame_extractor.extract(video_b)

        if len(frames_a) == 0 or len(frames_b) == 0:
            return 0.5, "Could not extract frames"

        # To tensors
        frames_a = torch.from_numpy(np.stack(frames_a)).float().unsqueeze(0).to(self.device) / 255.0
        frames_b = torch.from_numpy(np.stack(frames_b)).float().unsqueeze(0).to(self.device) / 255.0
        qual_a = torch.tensor([q.score() for q in qual_a]).unsqueeze(0).to(self.device)
        qual_b = torch.tensor([q.score() for q in qual_b]).unsqueeze(0).to(self.device)

        # Encode
        repr_a = self.video_encoder(frames_a, qual_a)
        repr_b = self.video_encoder(frames_b, qual_b)

        # Predict
        score = torch.sigmoid(self.model(repr_a, repr_b)).item()

        if score > 0.6:
            return score, f"Strongly prefer A (score={score:.2f})"
        elif score > 0.4:
            return score, f"Slightly prefer A (score={score:.2f})"
        else:
            return score, f"Prefer B (score={score:.2f})"


# ============================================================
# PART 8: UTILITIES
# ============================================================

def load_backbone(backbone_name: str = "clip"):
    """Load pretrained backbone."""
    if backbone_name == "clip":
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    elif backbone_name == "siglip":
        from transformers import AutoProcessor, AutoModel
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        return model, processor
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# USAGE EXAMPLE
# ============================================================

"""
Example usage:

# 1. Load backbone
backbone, processor = load_backbone("clip")

# 2. Create video encoder
video_encoder = CLIPVideoEncoder(embed_dim=768, temporal_method="transformer")
video_encoder.set_backbone(backbone)

# 3. Create preference model
preference_model = PreferenceHead(embed_dim=192)

# 4. Create trainer
trainer = PreferenceTrainer(preference_model, video_encoder)

# 5. Create dataset
video_pairs = [
    ("video_a.mp4", "video_b.mp4", 1),  # prefer A
    ("video_c.mp4", "video_d.mp4", 0),  # prefer B
    # ...
]
frame_extractor = VideoFrameExtractor(n_frames=16, quality_threshold=0.3)
dataset = VideoPreferenceDataset(video_pairs, frame_extractor)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# 6. Train
for epoch in range(10):
    for batch in dataloader:
        loss = trainer.train_step(batch)
        print(f"Loss: {loss:.4f}")

# 7. Predict
predictor = PreferencePredictor(video_encoder, preference_model, frame_extractor)
score, explanation = predictor.predict_preference("test_a.mp4", "test_b.mp4")
print(f"{explanation}")
"""
