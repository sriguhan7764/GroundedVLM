"""
COCO grounding evaluator using pycocotools.

Runs a VisualGroundingPipeline over COCO val2017 images and computes
standard detection metrics (mAP@50, mAP@75, mAP@[.5:.95]).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)


class COCOGroundingEvaluator:
    """Evaluate a visual grounding pipeline on COCO val2017.

    Args:
        ann_file: Path to COCO instances annotation JSON.
    """

    def __init__(self, ann_file: str) -> None:
        logger.info("Loading COCO annotations from '%s'.", ann_file)
        self.coco_gt = COCO(ann_file)
        self.ann_file = ann_file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pipeline,
        image_dir: str,
        categories: Optional[List[str]] = None,
        max_images: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run detection over COCO images and return AP metrics.

        Args:
            pipeline: VisualGroundingPipeline instance.
            image_dir: Directory containing COCO val2017 images.
            categories: List of category name strings to evaluate on.
                        If None, evaluates all 80 COCO categories.
            max_images: Limit number of images evaluated (for quick testing).

        Returns:
            Dict with keys mAP, mAP_50, mAP_75, mAP_small, mAP_medium,
            mAP_large and their float values.
        """
        # Determine category IDs to evaluate
        if categories is not None:
            cat_ids = self.coco_gt.getCatIds(catNms=categories)
        else:
            cat_ids = self.coco_gt.getCatIds()

        cat_info = self.coco_gt.loadCats(cat_ids)
        cat_names = [c["name"] for c in cat_info]
        text_prompt = ". ".join(cat_names) + "."

        img_ids = self.coco_gt.getImgIds(catIds=cat_ids)
        if max_images is not None:
            img_ids = img_ids[:max_images]

        logger.info(
            "Evaluating %d images with %d categories.", len(img_ids), len(cat_ids)
        )

        all_predictions: List[Dict[str, Any]] = []

        for img_id in tqdm(img_ids, desc="Detecting"):
            img_info = self.coco_gt.loadImgs([img_id])[0]
            img_path = os.path.join(image_dir, img_info["file_name"])

            if not os.path.isfile(img_path):
                logger.warning("Image not found: %s – skipping.", img_path)
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                detections = self._run_detection(pipeline, image, text_prompt)
                preds = self._format_predictions(detections, img_id, cat_names, cat_ids)
                all_predictions.extend(preds)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error on image %d: %s", img_id, exc)
                continue

        if not all_predictions:
            logger.warning("No predictions generated; returning zero metrics.")
            return {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "mAP_small": 0.0,
                "mAP_medium": 0.0,
                "mAP_large": 0.0,
            }

        coco_dt = self.coco_gt.loadRes(all_predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = img_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats  # 12-element array
        metrics = {
            "mAP": float(stats[0]),
            "mAP_50": float(stats[1]),
            "mAP_75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
        }
        logger.info("Evaluation complete: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        pipeline,
        image: Image.Image,
        text_prompt: str,
    ) -> Any:
        """Run the pipeline on a single image.

        Args:
            pipeline: VisualGroundingPipeline.
            image: PIL Image.
            text_prompt: Dot-separated category list.

        Returns:
            Detection dataclass.
        """
        return pipeline.ground(image, text_prompt)

    def _format_predictions(
        self,
        detection: Any,
        image_id: int,
        cat_names: List[str],
        cat_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """Convert a Detection to COCO-format prediction dicts.

        Args:
            detection: Detection dataclass.
            image_id: COCO image ID.
            cat_names: List of category names matching cat_ids order.
            cat_ids: List of COCO category IDs.

        Returns:
            List of dicts each with keys: image_id, category_id, bbox, score.
            bbox is in xywh format as required by COCO.
        """
        name_to_id: Dict[str, int] = {
            name: cid for name, cid in zip(cat_names, cat_ids)
        }

        preds: List[Dict[str, Any]] = []
        for box, score, label in zip(
            detection.boxes, detection.scores, detection.labels
        ):
            # Match label to a COCO category ID
            # Labels from detectors may contain extra whitespace or casing
            label_clean = label.strip().lower()
            cat_id: Optional[int] = None
            for name, cid in name_to_id.items():
                if name.lower() == label_clean:
                    cat_id = cid
                    break
            if cat_id is None:
                # Try partial match
                for name, cid in name_to_id.items():
                    if label_clean in name.lower() or name.lower() in label_clean:
                        cat_id = cid
                        break
            if cat_id is None:
                # Fall back to first category
                cat_id = cat_ids[0] if cat_ids else 1

            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            preds.append(
                {
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )
        return preds
