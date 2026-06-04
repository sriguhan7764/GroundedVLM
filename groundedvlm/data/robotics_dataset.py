"""
Robotics manipulation query dataset for cross-modal retrieval benchmarking.

If the data directory does not exist, synthetic data is generated on-the-fly:
random RGB images paired with robotics manipulation queries, with pre-defined
positive and hard-negative annotations.
"""

import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Canonical query templates for synthetic generation
_ROBOTICS_QUERIES: List[str] = [
    "pick up the red cube",
    "grasp the screwdriver",
    "place the blue block on the shelf",
    "hand me the yellow wrench",
    "push the green button",
    "lift the cylindrical container",
    "open the drawer on the left",
    "pour liquid from the bottle",
    "stack the small box on top",
    "grab the metal bracket",
    "press the emergency stop button",
    "rotate the valve clockwise",
    "insert the peg into the hole",
    "slide the door to the right",
    "unscrew the bolt with the tool",
    "place the cup upright on the table",
    "pick up the orange sphere",
    "drop the item in the bin",
    "align the gears carefully",
    "fetch the white clipboard",
]

# Hard-negative pairs: queries that are semantically close but distinct
_HARD_NEGATIVES: Dict[str, List[str]] = {
    "pick up the red cube": ["pick up the blue cube", "grab the red sphere"],
    "grasp the screwdriver": ["grasp the wrench", "pick up the screwdriver"],
    "place the blue block on the shelf": ["place the red block on the shelf", "put the blue block on the table"],
    "hand me the yellow wrench": ["hand me the blue wrench", "pass me the yellow hammer"],
    "push the green button": ["press the green button", "push the red button"],
    "lift the cylindrical container": ["lift the rectangular container", "pick up the cylindrical object"],
    "open the drawer on the left": ["open the drawer on the right", "close the drawer on the left"],
    "pour liquid from the bottle": ["pour water from the jug", "fill the cup from the bottle"],
    "stack the small box on top": ["place the small box beside", "stack the large box on top"],
    "grab the metal bracket": ["grab the plastic bracket", "pick up the metal bar"],
}


class RoboticsQueryDataset(Dataset):
    """Dataset of robotics manipulation image-query pairs for retrieval.

    If ``root`` does not exist or contains no data, synthetic data is generated
    automatically using random RGB images and the predefined query bank.

    Each item returns:
        - image: PIL Image (real or synthetic)
        - query_text: str
        - relevant_indices: list of int gallery indices relevant to this query

    The dataset also exposes:
        - ``self.images``:           flat list of all gallery PIL Images
        - ``self.queries``:          flat list of all query strings
        - ``self.relevant_indices``: list of relevant-index lists per query

    Args:
        root: Path to dataset directory. May be non-existent for synthetic data.
        split: "train" or "val".
        transform: Optional transform applied to images.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.transform = transform

        self.images: List[Image.Image] = []
        self.queries: List[str] = []
        self.relevant_indices: List[List[int]] = []
        self._items: List[Tuple[Image.Image, str, List[int]]] = []

        if os.path.isdir(root) and len(os.listdir(root)) > 0:
            self._load_from_disk()
        else:
            logger.info(
                "Root '%s' not found or empty — generating synthetic data.", root
            )
            self._generate_synthetic_data()

        logger.info(
            "RoboticsQueryDataset (%s): %d items.", split, len(self._items)
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, index: int
    ) -> Tuple[Any, str, List[int]]:
        image, query_text, relevant = self._items[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, query_text, relevant

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_synthetic_data(self) -> None:
        """Create synthetic image-query pairs with random RGB images.

        Generates one image per query.  Relevant indices point back to the
        same query's image.  Hard-negative queries share the same image
        so the encoder must distinguish based on semantics.
        """
        random.seed(42)
        np.random.seed(42)

        num_queries = len(_ROBOTICS_QUERIES)
        if self.split == "train":
            query_subset = _ROBOTICS_QUERIES
        else:
            query_subset = _ROBOTICS_QUERIES[:max(4, num_queries // 4)]

        # Build gallery: one base image per query (224x224 random RGB)
        gallery_images: List[Image.Image] = []
        for i, _ in enumerate(query_subset):
            # Each image has a dominant colour derived from its index
            r = int((i * 37) % 256)
            g = int((i * 71) % 256)
            b = int((i * 113) % 256)
            arr = np.full((224, 224, 3), [r, g, b], dtype=np.uint8)
            # Add some random noise so images are not identical
            noise = np.random.randint(0, 30, arr.shape, dtype=np.uint8)
            arr = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            gallery_images.append(Image.fromarray(arr))

        self.images = gallery_images

        # Build query list and relevant-index annotations
        self.queries = list(query_subset)
        self.relevant_indices = [[i] for i in range(len(query_subset))]

        # Add hard-negative queries
        for i, query in enumerate(query_subset):
            negatives = _HARD_NEGATIVES.get(query, [])
            for neg in negatives:
                self.queries.append(neg)
                # Relevant image is same as positive (index i)
                self.relevant_indices.append([i])

        # Build item list: (image, query, relevant_indices)
        self._items = [
            (self.images[idx[0]], q, idx)
            for q, idx in zip(self.queries, self.relevant_indices)
        ]

    def _load_from_disk(self) -> None:
        """Load images and queries from disk.

        Expects the directory structure:
            root/
                images/  *.jpg / *.png
                queries.txt   (one query per line)
                relevant.txt  (space-separated indices per line, matching queries.txt)
        """
        images_dir = os.path.join(self.root, "images")
        queries_file = os.path.join(self.root, "queries.txt")
        relevant_file = os.path.join(self.root, "relevant.txt")

        if not os.path.isdir(images_dir):
            logger.warning("images/ directory not found; falling back to synthetic.")
            self._generate_synthetic_data()
            return

        image_paths = sorted(
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.images = [Image.open(p).convert("RGB") for p in image_paths]

        if os.path.isfile(queries_file) and os.path.isfile(relevant_file):
            with open(queries_file, "r") as qf:
                self.queries = [line.strip() for line in qf if line.strip()]
            with open(relevant_file, "r") as rf:
                self.relevant_indices = [
                    [int(x) for x in line.strip().split()]
                    for line in rf
                    if line.strip()
                ]
        else:
            logger.warning(
                "queries.txt or relevant.txt missing; generating synthetic queries."
            )
            self.queries = [f"query_{i}" for i in range(len(self.images))]
            self.relevant_indices = [[i] for i in range(len(self.images))]

        self._items = [
            (self.images[idx[0]], q, idx)
            for q, idx in zip(self.queries, self.relevant_indices)
        ]
