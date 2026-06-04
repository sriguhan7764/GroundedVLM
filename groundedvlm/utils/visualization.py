"""
Visualization utilities for detection and retrieval results.

Provides colour-coded bounding box overlays and retrieval result grids.
"""

import logging
import math
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from groundedvlm.models.grounding_dino import Detection

logger = logging.getLogger(__name__)


def _confidence_color(score: float) -> Tuple[int, int, int]:
    """Map confidence score to an RGB colour.

    Args:
        score: Value in [0, 1].

    Returns:
        (R, G, B) tuple where:
            high  (>= 0.7) → green  (0, 200, 0)
            medium (0.4–0.7) → yellow (220, 180, 0)
            low   (< 0.4) → red    (220, 50, 50)
    """
    if score >= 0.7:
        return (0, 200, 0)
    elif score >= 0.4:
        return (220, 180, 0)
    else:
        return (220, 50, 50)


class Visualizer:
    """Draw detection and retrieval results onto images."""

    def draw_detections(
        self,
        image: Image.Image,
        detection: Detection,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """Overlay bounding boxes and labels on a PIL Image.

        Boxes are colour-coded by confidence: green (high), yellow (medium),
        red (low).  Each box has a filled label banner showing the category
        name and confidence score.

        Args:
            image: Source PIL Image (RGB).
            detection: Detection dataclass with boxes, scores, labels.
            save_path: If provided, saves the result to this path.

        Returns:
            Annotated PIL Image.
        """
        out_image = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", out_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            # Attempt to load a reasonable font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for box, score, label in zip(detection.boxes, detection.scores, detection.labels):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            r, g, b = _confidence_color(float(score))
            color_fill = (r, g, b, 180)
            color_outline = (r, g, b, 255)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color_outline, width=2)

            # Draw label banner
            label_text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            banner_y1 = max(0, y1 - text_h - 4)
            banner_y2 = max(text_h + 4, y1)
            draw.rectangle([x1, banner_y1, x1 + text_w + 4, banner_y2], fill=color_fill)
            draw.text((x1 + 2, banner_y1 + 2), label_text, fill=(255, 255, 255, 255), font=font)

        annotated = Image.alpha_composite(out_image, overlay).convert("RGB")

        if save_path is not None:
            annotated.save(save_path)
            logger.info("Saved detection visualisation to '%s'.", save_path)

        return annotated

    def draw_retrieval_results(
        self,
        query_image: Image.Image,
        result_images: List[Image.Image],
        scores: List[float],
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """Compose a grid showing the query and top retrieved images.

        The query image appears on the left with a blue border labelled
        "Query". Retrieved results are placed to the right in rank order,
        labelled with their similarity score.

        Args:
            query_image: PIL Image of the query.
            result_images: List of retrieved PIL Images (ranked).
            scores: Similarity score for each retrieved image.
            save_path: If provided, saves the grid to this path.

        Returns:
            Composite PIL Image grid.
        """
        thumb_size = (224, 224)
        border = 4
        padding = 8

        query_thumb = query_image.resize(thumb_size, Image.LANCZOS)
        result_thumbs = [img.resize(thumb_size, Image.LANCZOS) for img in result_images]

        num_results = len(result_thumbs)
        total_cols = 1 + num_results  # query + results
        cell_w = thumb_size[0] + 2 * border + padding
        cell_h = thumb_size[1] + 2 * border + padding + 20  # extra for label
        grid_w = total_cols * cell_w + padding
        grid_h = cell_h + 2 * padding

        grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))
        draw = ImageDraw.Draw(grid)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()

        def _paste_thumb(
            thumb: Image.Image,
            col: int,
            label: str,
            border_color: Tuple[int, int, int],
        ) -> None:
            x_off = padding + col * cell_w
            y_off = padding
            # Draw border
            draw.rectangle(
                [x_off, y_off, x_off + thumb_size[0] + 2 * border, y_off + thumb_size[1] + 2 * border],
                outline=border_color,
                width=border,
            )
            grid.paste(thumb, (x_off + border, y_off + border))
            text_y = y_off + thumb_size[1] + 2 * border + 2
            draw.text((x_off + border, text_y), label, fill=(50, 50, 50), font=font)

        _paste_thumb(query_thumb, col=0, label="Query", border_color=(0, 100, 220))

        for rank, (thumb, score) in enumerate(zip(result_thumbs, scores)):
            label = f"#{rank+1} ({score:.3f})"
            _paste_thumb(thumb, col=1 + rank, label=label, border_color=(80, 80, 80))

        if save_path is not None:
            grid.save(save_path)
            logger.info("Saved retrieval grid to '%s'.", save_path)

        return grid
