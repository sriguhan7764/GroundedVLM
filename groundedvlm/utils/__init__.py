"""
Utility functions and classes for GroundedVLM.

Exports:
    Visualizer          – detection and retrieval result visualiser
    box_iou             – pairwise IoU for xyxy boxes
    nms                 – Non-Maximum Suppression
    xyxy_to_xywh        – box format conversion
    xywh_to_xyxy        – box format conversion
    normalize_boxes     – pixel → [0,1] normalisation
    denormalize_boxes   – [0,1] → pixel denormalisation
    clip_boxes_to_image – clamp boxes to image bounds
    compute_recall_at_k – Recall@k metric
    compute_average_precision – AP metric
    compute_map         – mAP over queries
    compute_mrr         – Mean Reciprocal Rank
    compute_f1          – F1 for detection
"""

from groundedvlm.utils.visualization import Visualizer
from groundedvlm.utils.box_ops import (
    box_iou,
    nms,
    xyxy_to_xywh,
    xywh_to_xyxy,
    normalize_boxes,
    denormalize_boxes,
    clip_boxes_to_image,
)
from groundedvlm.utils.metrics import (
    compute_recall_at_k,
    compute_average_precision,
    compute_map,
    compute_mrr,
    compute_f1,
)

__all__ = [
    "Visualizer",
    "box_iou",
    "nms",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "normalize_boxes",
    "denormalize_boxes",
    "clip_boxes_to_image",
    "compute_recall_at_k",
    "compute_average_precision",
    "compute_map",
    "compute_mrr",
    "compute_f1",
]
