"""
Evaluation utilities for GroundedVLM.

Exports:
    COCOGroundingEvaluator – COCO detection evaluator using pycocotools
    RetrievalEvaluator     – cross-modal retrieval evaluator (Recall@k, MRR)
"""

from groundedvlm.evaluation.coco_eval import COCOGroundingEvaluator
from groundedvlm.evaluation.retrieval_eval import RetrievalEvaluator

__all__ = ["COCOGroundingEvaluator", "RetrievalEvaluator"]
