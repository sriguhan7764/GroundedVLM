"""
Visual grounding pipeline: detector + optional contrastive re-ranker.

Supports Grounding DINO and Florence-2 as detector backends and
CLIP / ALIGN / SigLIP as optional re-ranking encoders.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from groundedvlm.models.grounding_dino import Detection, GroundingDINODetector
from groundedvlm.models.florence2 import Florence2Detector
from groundedvlm.models.clip_encoder import CLIPEncoder
from groundedvlm.models.align_encoder import ALIGNEncoder
from groundedvlm.models.siglip_encoder import SigLIPEncoder
from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker

logger = logging.getLogger(__name__)


class VisualGroundingPipeline:
    """End-to-end visual grounding pipeline.

    Args:
        detector_type: One of "dino" or "florence2".
        detector_config: Configuration dict for the chosen detector.
        encoder_config: Optional configuration dict for the re-ranking encoder.
                        Must include an "encoder_type" key ("clip", "align",
                        "siglip") and optionally "temperature" and "alpha".
                        If None, re-ranking is disabled.
    """

    def __init__(
        self,
        detector_type: str,
        detector_config: Dict[str, Any],
        encoder_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.detector_type = detector_type.lower()
        self.detector = self._load_detector(self.detector_type, detector_config)

        self.reranker: Optional[ContrastiveReranker] = None
        if encoder_config is not None:
            encoder = self._load_encoder(encoder_config)
            temperature = float(encoder_config.get("temperature", 0.07))
            alpha = float(encoder_config.get("alpha", 0.5))
            self.reranker = ContrastiveReranker(
                encoder=encoder,
                temperature=temperature,
                alpha=alpha,
            )
            logger.info("Contrastive re-ranker enabled.")
        else:
            logger.info("No encoder config provided; re-ranking disabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground(self, image: Image.Image, query: str) -> Detection:
        """Detect and optionally re-rank objects matching the query.

        Args:
            image: PIL Image (RGB).
            query: Text description of objects to find.

        Returns:
            Detection dataclass sorted by (combined) score descending.
        """
        if self.detector_type == "dino":
            detection = self.detector.detect(image, query)
        else:  # florence2
            detection = self.detector.caption_to_phrase_grounding(image, query)

        if self.reranker is not None and len(detection) > 0:
            detection = self.reranker.rerank(image, detection, query)

        return detection

    def ground_batch(
        self,
        image_query_pairs: List[Tuple[Image.Image, str]],
    ) -> List[Detection]:
        """Ground a list of (image, query) pairs.

        Args:
            image_query_pairs: List of (PIL Image, query string) tuples.

        Returns:
            List of Detection objects, one per pair.
        """
        results: List[Detection] = []
        for image, query in image_query_pairs:
            detection = self.ground(image, query)
            results.append(detection)
        return results

    # ------------------------------------------------------------------
    # Private factory methods
    # ------------------------------------------------------------------

    def _load_detector(
        self,
        detector_type: str,
        config: Dict[str, Any],
    ):
        """Instantiate the appropriate detector.

        Args:
            detector_type: "dino" or "florence2".
            config: Detector configuration dict.

        Returns:
            Detector instance.
        """
        if detector_type == "dino":
            logger.info("Loading Grounding DINO detector.")
            return GroundingDINODetector(config)
        elif detector_type == "florence2":
            logger.info("Loading Florence-2 detector.")
            return Florence2Detector(config)
        else:
            raise ValueError(
                f"Unknown detector_type '{detector_type}'. "
                "Choose 'dino' or 'florence2'."
            )

    def _load_encoder(self, config: Dict[str, Any]):
        """Instantiate the appropriate encoder for re-ranking.

        Args:
            config: Encoder configuration dict.  Must include "encoder_type".

        Returns:
            Encoder instance (CLIPEncoder, ALIGNEncoder, or SigLIPEncoder).
        """
        encoder_type = config.get("encoder_type", "clip").lower()
        device = config.get("device", "cuda")

        if encoder_type == "clip":
            model_name = config.get("model_name", "ViT-L-14")
            pretrained = config.get("pretrained", "openai")
            logger.info("Loading CLIP encoder ('%s', '%s').", model_name, pretrained)
            return CLIPEncoder(model_name=model_name, pretrained=pretrained, device=device)

        elif encoder_type == "align":
            model_id = config.get("model_id", "kakaobrain/align-base")
            logger.info("Loading ALIGN encoder ('%s').", model_id)
            return ALIGNEncoder(model_id=model_id, device=device)

        elif encoder_type == "siglip":
            model_id = config.get("model_id", "google/siglip-large-patch16-384")
            logger.info("Loading SigLIP encoder ('%s').", model_id)
            return SigLIPEncoder(model_id=model_id, device=device)

        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                "Choose 'clip', 'align', or 'siglip'."
            )
