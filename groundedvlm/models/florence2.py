"""
Florence-2 object detection and phrase grounding wrapper.

Uses the Microsoft Florence-2 model for open-vocabulary detection and
caption-to-phrase grounding tasks.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from groundedvlm.models.grounding_dino import Detection

logger = logging.getLogger(__name__)

# Florence-2 bounding box normalisation range
FLORENCE_NORM = 1000.0


class Florence2Detector:
    """Open-vocabulary detector using Florence-2.

    Args:
        config: Dict with keys:
            - model_id (str)
            - task_prompt (str)          default "<OD>"
            - phrase_grounding_prompt (str)
            - device (str)
            - max_new_tokens (int)
            - num_beams (int)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.model_id: str = config.get("model_id", "microsoft/Florence-2-large")
        self.task_prompt: str = config.get("task_prompt", "<OD>")
        self.phrase_grounding_prompt: str = config.get(
            "phrase_grounding_prompt", "<CAPTION_TO_PHRASE_GROUNDING>"
        )
        self.max_new_tokens: int = int(config.get("max_new_tokens", 1024))
        self.num_beams: int = int(config.get("num_beams", 3))

        requested_device = config.get("device", "cuda")
        self.device = torch.device(
            requested_device if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading Florence-2 model '%s' on device '%s'.",
            self.model_id,
            self.device,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("Florence-2 model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Image.Image,
        task_prompt: Optional[str] = None,
    ) -> Detection:
        """Run object detection using the <OD> task.

        Args:
            image: PIL Image (RGB).
            task_prompt: Override default task prompt.

        Returns:
            Detection dataclass.
        """
        prompt = task_prompt or self.task_prompt
        response = self._run_inference(image, prompt, text_input=None)
        image_size = (image.height, image.width)
        return self._parse_od_response(response, image_size)

    def caption_to_phrase_grounding(
        self,
        image: Image.Image,
        caption: str,
    ) -> Detection:
        """Run phrase grounding from a free-form caption.

        Args:
            image: PIL Image (RGB).
            caption: A descriptive caption; detected phrases will be localised.

        Returns:
            Detection dataclass.
        """
        response = self._run_inference(
            image,
            self.phrase_grounding_prompt,
            text_input=caption,
        )
        image_size = (image.height, image.width)
        return self._parse_od_response(response, image_size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: Optional[str] = None,
    ) -> str:
        """Execute Florence-2 generation and decode the response.

        Args:
            image: PIL Image (RGB).
            task_prompt: Task token, e.g. "<OD>".
            text_input: Additional text appended after the task token (optional).

        Returns:
            Decoded text response string.
        """
        if text_input is not None:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )

        generated_text: str = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Use the processor to post-process if available
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed

    def _parse_od_response(
        self,
        response: Any,
        image_size: Tuple[int, int],
    ) -> Detection:
        """Parse a Florence-2 object detection or phrase grounding response.

        Florence-2 boxes are normalised to [0, 1000] in (x1, y1, x2, y2) order.
        This method converts them to pixel coordinates.

        Args:
            response: Either a dict returned by post_process_generation or a raw
                      string. Expected dict keys: 'bboxes', 'labels'.
            image_size: (H, W) of the source image.

        Returns:
            Detection dataclass with pixel-coordinate boxes.
        """
        h, w = image_size

        # post_process_generation returns a dict like:
        # {task_key: {"bboxes": [...], "labels": [...]}}
        if isinstance(response, dict):
            # Find the nested payload regardless of task key
            payload: Optional[Dict[str, Any]] = None
            for key, val in response.items():
                if isinstance(val, dict) and "bboxes" in val:
                    payload = val
                    break
            if payload is None:
                logger.warning("Florence-2 response missing 'bboxes' key: %s", response)
                return Detection(
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    scores=np.zeros((0,), dtype=np.float32),
                    labels=[],
                    image_size=image_size,
                )
        else:
            # Fallback: no structured parse available
            logger.warning(
                "Florence-2 response is not a dict; returning empty detection."
            )
            return Detection(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=[],
                image_size=image_size,
            )

        raw_boxes: List[List[float]] = payload.get("bboxes", [])
        labels: List[str] = payload.get("labels", [])

        if not raw_boxes:
            return Detection(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=[],
                image_size=image_size,
            )

        boxes_pixel = []
        for box in raw_boxes:
            # box is [x1, y1, x2, y2] in [0, 1000] range
            x1 = box[0] * w / FLORENCE_NORM
            y1 = box[1] * h / FLORENCE_NORM
            x2 = box[2] * w / FLORENCE_NORM
            y2 = box[3] * h / FLORENCE_NORM
            boxes_pixel.append([x1, y1, x2, y2])

        boxes_np = np.array(boxes_pixel, dtype=np.float32)
        # Florence-2 doesn't emit per-box confidence; assign score of 1.0
        scores_np = np.ones(len(boxes_np), dtype=np.float32)

        return Detection(
            boxes=boxes_np,
            scores=scores_np,
            labels=labels,
            image_size=image_size,
        )
