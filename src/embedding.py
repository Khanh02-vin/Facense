"""
Embedding Module - Appearance Embedding Extraction

Supports:
- SigLIP (google/siglip-base-patch16-224)
- DINOv2 (facebook/dinov2-base)
- CLIP (openai/clip-vit-base-patch32)
"""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result of embedding extraction."""
    embedding: np.ndarray
    model: str
    image_id: str
    timestamp: float


class AppearanceEmbedding:
    """Extract appearance embeddings from images."""

    def __init__(
        self,
        model_name: Literal["siglip", "dinov2", "clip"] = "siglip",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.model = None
        self.processor = None

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self):
        """Load model and processor."""
        if self.model is not None:
            return

        if self.model_name == "siglip":
            self._load_siglip()
        elif self.model_name == "dinov2":
            self._load_dinov2()
        elif self.model_name == "clip":
            self._load_clip()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _load_siglip(self):
        """Load SigLIP model."""
        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("transformers required: pip install transformers torch")

    def _load_dinov2(self):
        """Load DINOv2 model."""
        try:
            import torch
            from torchvision import transforms
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.model.to(self.device)
            self.model.eval()
            self.processor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            raise ImportError("torch and torchvision required")

    def _load_clip(self):
        """Load CLIP model."""
        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("transformers required: pip install transformers torch")

    def extract(self, image, image_id: str = "unknown") -> EmbeddingResult:
        """Extract embedding from a single image.

        Args:
            image: PIL Image or numpy array
            image_id: Identifier for the image

        Returns:
            EmbeddingResult with embedding vector
        """
        self.load()

        if self.model_name == "dinov2":
            return self._extract_dinov2(image, image_id)
        else:
            return self._extract_transformer(image, image_id)

    def _extract_transformer(self, image, image_id: str) -> EmbeddingResult:
        """Extract using transformer models (SigLIP, CLIP)."""
        import torch

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.cpu().numpy().flatten()

            if self.normalize:
                embedding = embedding / np.linalg.norm(embedding)

        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            image_id=image_id,
            timestamp=0.0
        )

    def _extract_dinov2(self, image, image_id: str) -> EmbeddingResult:
        """Extract using DINOv2."""
        import torch

        with torch.no_grad():
            if not isinstance(image, torch.Tensor):
                image_tensor = self.processor(image).unsqueeze(0)
            else:
                image_tensor = image.unsqueeze(0)

            image_tensor = image_tensor.to(self.device)
            embedding = self.model(image_tensor).cpu().numpy().flatten()

            if self.normalize:
                embedding = embedding / np.linalg.norm(embedding)

        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            image_id=image_id,
            timestamp=0.0
        )

    def extract_batch(self, images, image_ids=None) -> list[EmbeddingResult]:
        """Extract embeddings from a batch of images."""
        if image_ids is None:
            image_ids = [f"img_{i}" for i in range(len(images))]

        return [self.extract(img, img_id) for img, img_id in zip(images, image_ids)]


class EmbeddingStability:
    """Test embedding stability under augmentation."""

    def __init__(self, embedding_model: AppearanceEmbedding):
        self.model = embedding_model

    def compute_stability(
        self,
        image,
        n_augmentations: int = 5
    ) -> dict:
        """Compute cosine similarity between original and augmented embeddings.

        Args:
            image: PIL Image
            n_augmentations: Number of augmentations to test

        Returns:
            dict with stability metrics
        """
        import torch
        import torchvision.transforms as T

        self.model.load()

        # Define augmentation pipeline
        augmentations = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.RandomRotation(degrees=5),
        ])

        # Get original embedding
        original = self.model.extract(image, "original")

        similarities = []
        for i in range(n_augmentations):
            aug_image = augmentations(image)
            aug_result = self.model.extract(aug_image, f"aug_{i}")
            sim = np.dot(original.embedding, aug_result.embedding)
            similarities.append(sim)

        return {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "similarities": similarities
        }
