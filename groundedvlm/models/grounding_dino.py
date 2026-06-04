"""
Grounding DINO zero-shot object detector wrapper.

Wraps the HuggingFace Grounding DINO model for open-vocabulary detection.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.ops as tv_ops
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Container for object detection results.

    Attributes:
        boxes: Bounding boxes of shape (N, 4) in xyxy pixel format.
        scores: Confidence scores of shape (N,).
        labels: List of N label strings.
        image_size: (H, W) of the source image.
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: List[str]
    image_size: Tuple[int, int]  # (H, W)

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32)
        self.scores = np.asarray(self.scores, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.scores)


class GroundingDINODetector:
    """Zero-shot object detector based on Grounding DINO.

    Args:
        config: Dictionary with keys:
            - model_id (str)
            - box_threshold (float)
            - text_threshold (float)
            - device (str)
            - nms_threshold (float)
            - max_detections (int)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.model_id: str = config.get("model_id", "IDEA-Research/grounding-dino-tiny")
        self.box_threshold: float = float(config.get("box_threshold", 0.35))
        self.text_threshold: float = float(config.get("text_threshold", 0.25))
        self.nms_threshold: float = float(config.get("nms_threshold", 0.5))
        self.max_detections: int = int(config.get("max_detections", 300))

        # Resolve device — fall back to CPU when CUDA is unavailable
        requested_device = config.get("device", "cuda")
        self.device = torch.device(
            requested_device if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading Grounding DINO model '%s' on device '%s'.",
            self.model_id,
            self.device,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Grounding DINO model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image, text_prompt: str) -> Detection:
        """Run open-vocabulary detection on a single image.

        Args:
            image: PIL Image (RGB).
            text_prompt: Dot-separated category string, e.g. "cat. dog. car."

        Returns:
            Detection dataclass with boxes, scores, labels, image_size.
        """
        image_size = (image.height, image.width)

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        detection = self._postprocess(outputs, image_size, text_prompt)
        return detection

    def detect_batch(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
    ) -> List[Detection]:
        """Run detection on a list of image-text pairs.

        Args:
            images: List of PIL Images.
            text_prompts: List of text prompt strings (one per image).

        Returns:
            List of Detection objects, one per image.
        """
        if len(images) != len(text_prompts):
            raise ValueError(
                f"images ({len(images)}) and text_prompts ({len(text_prompts)}) "
                "must have the same length."
            )

        results: List[Detection] = []
        for image, text_prompt in zip(images, text_prompts):
            detection = self.detect(image, text_prompt)
            results.append(detection)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        outputs: Any,
        image_size: Tuple[int, int],
        text_prompt: str,
    ) -> Detection:
        """Convert raw model outputs to a Detection object.

        Applies score thresholding and NMS.

        Args:
            outputs: Raw model output dict.
            image_size: (H, W) of the input image.
            text_prompt: Original text prompt (used for label parsing).

        Returns:
            Detection dataclass.
        """
        h, w = image_size
        target_sizes = torch.tensor([[h, w]], device=self.device)

        # post_process_grounded_object_detection returns a list of dicts
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        result = results[0]
        boxes: torch.Tensor = result["boxes"]       # (N, 4) xyxy, pixel
        scores: torch.Tensor = result["scores"]     # (N,)
        labels: List[str] = result["labels"]        # List[str]

        if boxes.numel() == 0:
            return Detection(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=[],
                image_size=image_size,
            )

        # NMS
        keep = tv_ops.nms(boxes, scores, iou_threshold=self.nms_threshold)
        # Limit to max_detections
        keep = keep[: self.max_detections]

        boxes_np = boxes[keep].cpu().numpy()
        scores_np = scores[keep].cpu().numpy()
        labels_filtered = [labels[i] for i in keep.tolist()]

        return Detection(
            boxes=boxes_np,
            scores=scores_np,
            labels=labels_filtered,
            image_size=image_size,
        )
