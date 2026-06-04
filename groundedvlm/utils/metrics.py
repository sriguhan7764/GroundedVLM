"""
Evaluation metrics: Recall@k, AP, mAP, MRR, F1 for detection and retrieval tasks.
"""

import logging
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_recall_at_k(
    retrieved_indices: List[int],
    relevant_indices: List[int],
    k: int,
) -> float:
    """
    Compute Recall@k: fraction of relevant items found in the top-k results.

    Args:
        retrieved_indices: Ordered list of retrieved item indices.
        relevant_indices: Set of ground-truth relevant item indices.
        k: Cut-off rank.

    Returns:
        Recall@k as a float in [0, 1].
    """
    if not relevant_indices:
        return 0.0
    top_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def compute_average_precision(
    retrieved_indices: List[int],
    relevant_indices: List[int],
) -> float:
    """
    Compute Average Precision (AP) for a single query.

    Args:
        retrieved_indices: Ordered list of retrieved item indices.
        relevant_indices: Ground-truth relevant item indices.

    Returns:
        AP score as a float in [0, 1].
    """
    relevant_set = set(relevant_indices)
    if not relevant_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_set:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_set)


def compute_map(
    all_retrieved: List[List[int]],
    all_relevant: List[List[int]],
) -> float:
    """
    Compute Mean Average Precision (mAP) over multiple queries.

    Args:
        all_retrieved: List of retrieved index lists, one per query.
        all_relevant: List of relevant index lists, one per query.

    Returns:
        mAP score as a float in [0, 1].
    """
    if not all_retrieved:
        return 0.0
    ap_scores = [
        compute_average_precision(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return float(np.mean(ap_scores))


def compute_mrr(
    retrieved_indices_list: List[List[int]],
    relevant_indices_list: List[List[int]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over multiple queries.

    Args:
        retrieved_indices_list: List of retrieved index lists per query.
        relevant_indices_list: List of relevant index lists per query.

    Returns:
        MRR score as a float in [0, 1].
    """
    if not retrieved_indices_list:
        return 0.0

    rr_scores = []
    for retrieved, relevant in zip(retrieved_indices_list, relevant_indices_list):
        relevant_set = set(relevant)
        rr = 0.0
        for rank, idx in enumerate(retrieved, start=1):
            if idx in relevant_set:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return float(np.mean(rr_scores))


def compute_f1(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute F1 score for detection given predicted and ground-truth boxes.

    Matches predicted boxes to ground-truth boxes greedily by descending IoU.
    A prediction is a true positive if it exceeds iou_threshold with an unmatched GT box.

    Args:
        pred_boxes: Tensor of shape (P, 4) in xyxy format.
        gt_boxes: Tensor of shape (G, 4) in xyxy format.
        iou_threshold: IoU threshold for a match.

    Returns:
        F1 score as a float.
    """
    from groundedvlm.utils.box_ops import box_iou

    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 1.0
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0

    iou_matrix = box_iou(pred_boxes, gt_boxes)  # (P, G)
    matched_gt = set()
    tp = 0

    # Greedily assign highest-IoU pairs
    iou_flat = iou_matrix.flatten()
    sorted_flat = iou_flat.argsort(descending=True)
    num_gt = gt_boxes.shape[0]

    for flat_idx in sorted_flat.tolist():
        pred_idx = flat_idx // num_gt
        gt_idx = flat_idx % num_gt
        iou_val = iou_matrix[pred_idx, gt_idx].item()
        if iou_val < iou_threshold:
            break
        if gt_idx not in matched_gt:
            matched_gt.add(gt_idx)
            tp += 1

    precision = tp / pred_boxes.shape[0] if pred_boxes.shape[0] > 0 else 0.0
    recall = tp / gt_boxes.shape[0] if gt_boxes.shape[0] > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)
