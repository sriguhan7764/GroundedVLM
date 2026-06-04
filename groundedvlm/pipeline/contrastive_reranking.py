"""
Contrastive re-ranking stage using CLIP/ALIGN/SigLIP encoders.

Given a set of candidate detections, crops each region from the image,
encodes them with a vision-language encoder, and re-scores them against
the text query using cosine similarity.  Final scores combine the original
detector confidence with the encoder similarity via a weighted blend.
"""

import logging
from typing import List

import numpy as np
from PIL import Image

from groundedvlm.models.grounding_dino import Detection

logger = logging.getLogger(__name__)


class ContrastiveReranker:
    """Re-rank detections by comparing cropped regions to the query text.

    Args:
        encoder: Any encoder with ``encode_image`` and ``encode_text``
                 returning L2-normalised tensors.
        temperature: Softmax temperature applied to similarities (unused in
                     scoring but kept for API parity).
        alpha: Weight for blending detector score (alpha) with rerank score
               (1 - alpha).  alpha=1.0 means pure detector score; alpha=0.0
               means pure rerank score.
    """

    def __init__(self, encoder, temperature: float = 0.07, alpha: float = 0.5) -> None:
        self.encoder = encoder
        self.temperature = temperature
        self.alpha = alpha
        logger.debug(
            "ContrastiveReranker initialised (alpha=%.2f, temperature=%.3f).",
            alpha,
            temperature,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        image: Image.Image,
        detection: Detection,
        query: str,
    ) -> Detection:
        """Re-score and reorder detections using encoder similarity.

        Args:
            image: The original PIL Image from which detections were obtained.
            detection: Detection dataclass with boxes/scores/labels.
            query: Text query to rank against.

        Returns:
            New Detection with boxes/scores/labels reordered by combined score
            (highest combined score first).
        """
        if len(detection) == 0:
            logger.debug("No detections to re-rank.")
            return detection

        crops = self._crop_candidates(image, detection.boxes)
        rerank_scores = self._compute_region_text_similarity(crops, query)
        combined = self._combine_scores(detection.scores, rerank_scores, self.alpha)

        # Sort by combined score descending
        order = np.argsort(-combined)
        reordered_boxes = detection.boxes[order]
        reordered_scores = combined[order]
        reordered_labels = [detection.labels[i] for i in order]

        return Detection(
            boxes=reordered_boxes,
            scores=reordered_scores,
            labels=reordered_labels,
            image_size=detection.image_size,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _crop_candidates(
        self,
        image: Image.Image,
        boxes: np.ndarray,
    ) -> List[Image.Image]:
        """Crop image regions corresponding to each detection box.

        Args:
            image: Original PIL Image.
            boxes: Array of shape (N, 4) in xyxy pixel format.

        Returns:
            List of N cropped PIL Images.  Degenerate crops are replaced with
            a 1x1 black image to avoid downstream errors.
        """
        crops: List[Image.Image] = []
        img_w, img_h = image.size
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            # Clamp to image boundaries
            x1 = max(0.0, min(x1, img_w))
            y1 = max(0.0, min(y1, img_h))
            x2 = max(0.0, min(x2, img_w))
            y2 = max(0.0, min(y2, img_h))

            if x2 <= x1 or y2 <= y1:
                # Degenerate box — create a tiny black placeholder
                crops.append(Image.new("RGB", (1, 1), color=(0, 0, 0)))
            else:
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
        return crops

    def _compute_region_text_similarity(
        self,
        crops: List[Image.Image],
        query: str,
    ) -> np.ndarray:
        """Encode crops and query, return per-crop cosine similarity to query.

        Args:
            crops: List of cropped PIL Images.
            query: Text string.

        Returns:
            NumPy array of shape (N,) with similarity scores in [-1, 1].
        """
        image_embeds = self.encoder.encode_image(crops)       # (N, D)
        text_embeds = self.encoder.encode_text([query])        # (1, D)

        # Similarity matrix (N, 1) → squeeze to (N,)
        sim_matrix = self.encoder.compute_similarity(image_embeds, text_embeds)
        similarities = sim_matrix[:, 0].cpu().numpy()
        return similarities.astype(np.float32)

    def _combine_scores(
        self,
        detection_scores: np.ndarray,
        rerank_scores: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Blend detector confidence with encoder similarity scores.

        Both score arrays are independently min-max normalised to [0, 1]
        before blending so that their scales are comparable.

        Args:
            detection_scores: Array of shape (N,) from the detector.
            rerank_scores: Array of shape (N,) from the encoder.
            alpha: Blend weight for detector scores.

        Returns:
            Combined score array of shape (N,).
        """
        def _minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-8:
                return np.ones_like(arr, dtype=np.float32)
            return ((arr - lo) / (hi - lo)).astype(np.float32)

        norm_det = _minmax(detection_scores)
        norm_rerank = _minmax(rerank_scores)
        combined = alpha * norm_det + (1.0 - alpha) * norm_rerank
        return combined
