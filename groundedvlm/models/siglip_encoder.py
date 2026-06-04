"""
SigLIP encoder wrapper using the HuggingFace transformers library.

SigLIP uses a sigmoid loss during training rather than softmax, so
compute_similarity returns raw dot products (not softmax-normalised).
The interface mirrors CLIPEncoder and ALIGNEncoder for interchangeability.
"""

import logging
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

logger = logging.getLogger(__name__)


class SigLIPEncoder:
    """SigLIP image-text encoder.

    Note: SigLIP was trained with a sigmoid loss, so similarities are best
    interpreted as logits rather than probabilities.  compute_similarity()
    therefore returns raw (unnormalised) dot products.

    Args:
        model_id: HuggingFace model identifier, e.g.
                  "google/siglip-large-patch16-384".
        device: Target device string.
    """

    def __init__(
        self,
        model_id: str = "google/siglip-large-patch16-384",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(
            "Loading SigLIP model '%s' on device '%s'.",
            model_id,
            self.device,
        )

        self.processor = SiglipProcessor.from_pretrained(model_id)
        self.model = SiglipModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        logger.info("SigLIP model loaded successfully.")

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

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)

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

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw dot-product similarity (SigLIP does NOT use softmax).

        Args:
            image_embeds: Tensor of shape (N, D), L2-normalised.
            text_embeds: Tensor of shape (M, D), L2-normalised.

        Returns:
            Similarity matrix of shape (N, M) containing raw dot products.
        """
        # SigLIP uses sigmoid loss; return dot products as logits.
        return image_embeds @ text_embeds.T

    def encode_image_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode a large list of images in mini-batches.

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
