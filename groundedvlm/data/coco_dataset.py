"""
PyTorch Dataset wrapper for COCO detection with grounding support.

Loads COCO val2017 images and annotations, optionally filtering by category.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class COCOGroundingDataset(Dataset):
    """COCO dataset returning images, annotations, and category name lists.

    Each item consists of:
        - image (PIL Image or transformed tensor)
        - annotations dict with keys:
              boxes: (N, 4) float32 array in xyxy pixel format
              category_ids: list of int COCO category IDs
              category_names: list of str category names
              image_id: int
        - text_prompt: dot-separated category string for the detector

    Args:
        root: Root directory containing COCO data.
        ann_file: Path to COCO annotation JSON.
        transform: Optional callable applied to the PIL Image.
        categories: Optional list of category name strings to include.
                    If None, all 80 categories are included.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        self.root = root
        self.transform = transform

        logger.info("Loading COCO annotations from '%s'.", ann_file)
        self.coco = COCO(ann_file)

        self.cat_ids: List[int] = self._load_categories(categories)
        self.cat_info: Dict[int, Dict[str, Any]] = {
            c["id"]: c for c in self.coco.loadCats(self.cat_ids)
        }
        self.category_names: List[str] = [
            self.cat_info[cid]["name"] for cid in self.cat_ids
        ]

        # Collect image IDs that have at least one annotation in the chosen cats
        self.img_ids: List[int] = list(
            set(
                img_id
                for cat_id in self.cat_ids
                for img_id in self.coco.getImgIds(catIds=[cat_id])
            )
        )
        self.img_ids.sort()

        logger.info(
            "COCOGroundingDataset: %d images, %d categories.",
            len(self.img_ids),
            len(self.cat_ids),
        )

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(
        self, index: int
    ) -> Tuple[Any, Dict[str, Any], str]:
        """Return (image, annotations_dict, text_prompt) for the given index."""
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        cat_ids_out: List[int] = []
        cat_names_out: List[str] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            cid = ann["category_id"]
            cat_ids_out.append(cid)
            cat_names_out.append(self.cat_info.get(cid, {}).get("name", "unknown"))

        annotations = {
            "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
            "category_ids": cat_ids_out,
            "category_names": cat_names_out,
            "image_id": img_id,
        }

        text_prompt = ". ".join(self.category_names) + "."
        return image, annotations, text_prompt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_categories(self, categories: Optional[List[str]]) -> List[int]:
        """Return COCO category IDs for the requested category names.

        Args:
            categories: List of category name strings, or None for all.

        Returns:
            Sorted list of COCO category IDs.
        """
        if categories is None:
            cat_ids = self.coco.getCatIds()
        else:
            cat_ids = self.coco.getCatIds(catNms=categories)
            if len(cat_ids) == 0:
                raise ValueError(
                    f"No COCO categories found matching: {categories}"
                )
        return sorted(cat_ids)
