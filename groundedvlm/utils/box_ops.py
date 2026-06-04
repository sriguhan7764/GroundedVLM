"""
Bounding box utility functions for format conversion, IoU, NMS, and normalization.
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format.
        boxes2: Tensor of shape (M, 4) in xyxy format.

    Returns:
        iou: Tensor of shape (N, M) containing pairwise IoU values.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Non-Maximum Suppression for xyxy format boxes.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format.
        scores: Tensor of shape (N,) with confidence scores.
        threshold: IoU threshold above which to suppress.

    Returns:
        keep: Tensor of indices (long) of kept boxes.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    sorted_indices = scores.argsort(descending=True)
    keep = []

    while sorted_indices.numel() > 0:
        current = sorted_indices[0].item()
        keep.append(current)

        if sorted_indices.numel() == 1:
            break

        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]

        iou_vals = box_iou(current_box, remaining_boxes).squeeze(0)
        mask = iou_vals <= threshold
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy (x1, y1, x2, y2) to xywh (x, y, w, h) format.

    Args:
        boxes: Tensor of shape (..., 4).

    Returns:
        Tensor of shape (..., 4) in xywh format.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x1, y1, w, h], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xywh (x, y, w, h) to xyxy (x1, y1, x2, y2) format.

    Args:
        boxes: Tensor of shape (..., 4).

    Returns:
        Tensor of shape (..., 4) in xyxy format.
    """
    x, y, w, h = boxes.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    return torch.stack([x, y, x2, y2], dim=-1)


def normalize_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Normalize boxes from pixel coordinates to [0, 1] range.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy pixel format.
        image_size: (height, width) of the image.

    Returns:
        Tensor of shape (N, 4) with coordinates in [0, 1].
    """
    h, w = image_size
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes / scale


def denormalize_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert boxes from [0, 1] normalized coordinates to pixel coordinates.

    Args:
        boxes: Tensor of shape (N, 4) with coordinates in [0, 1].
        image_size: (height, width) of the image.

    Returns:
        Tensor of shape (N, 4) in pixel coordinates (xyxy).
    """
    h, w = image_size
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


def clip_boxes_to_image(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes so they lie within the image boundaries.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy pixel format.
        image_size: (height, width) of the image.

    Returns:
        Tensor of shape (N, 4) clipped to [0, W] x [0, H].
    """
    h, w = image_size
    x1 = boxes[:, 0].clamp(min=0, max=w)
    y1 = boxes[:, 1].clamp(min=0, max=h)
    x2 = boxes[:, 2].clamp(min=0, max=w)
    y2 = boxes[:, 3].clamp(min=0, max=h)
    return torch.stack([x1, y1, x2, y2], dim=-1)
