"""
CLIP encoder wrapper using the open_clip library.

Provides image and text embeddings with a unified interface shared by
ALIGN and SigLIP encoders.
"""

import logging
from typing import List

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """CLIP encoder for computing image and text embeddings.

    Uses open_clip for model loading which supports a wide variety of
    ViT checkpoints including OpenAI ViT-L/14.

    Args:
        model_name: open_clip model name, e.g. "ViT-L-14".
        pretrained: Pre-trained weights tag, e.g. "openai".
        device: Target device string, e.g. "cuda" or "cpu".
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(
            "Loading CLIP model '%s' (pretrained='%s') on device '%s'.",
            model_name,
            pretrained,
            self.device,
        )

        # create_model_and_transforms returns (model, preprocess_train, preprocess_val)
        model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(model_name)

        logger.info("CLIP model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL images into L2-normalised embeddings.

        Args:
            images: List of PIL Images.

        Returns:
            Tensor of shape (N, D), L2-normalised.
        """
        if not images:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(tensors)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into L2-normalised embeddings.

        Args:
            texts: List of text strings.

        Returns:
            Tensor of shape (N, D), L2-normalised.
        """
        if not texts:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity matrix between image and text embeddings.

        Args:
            image_embeds: Tensor of shape (N, D), assumed L2-normalised.
            text_embeds: Tensor of shape (M, D), assumed L2-normalised.

        Returns:
            Similarity matrix of shape (N, M).
        """
        # Both inputs are already L2-normalised; dot product = cosine similarity.
        return image_embeds @ text_embeds.T

    def encode_image_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Encode a potentially large list of images in mini-batches.

        Args:
            images: List of PIL Images.
            batch_size: Number of images per batch.

        Returns:
            Tensor of shape (N, D), L2-normalised.
        """
        all_embeds: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            embeds = self.encode_image(batch)
            all_embeds.append(embeds)

        if not all_embeds:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        return torch.cat(all_embeds, dim=0)
