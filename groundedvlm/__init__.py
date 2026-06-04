"""
GroundedVLM: Open-Vocabulary Visual Grounding with Vision-Language Models.

Combines Grounding DINO and Florence-2 for zero-shot object detection and
benchmarks CLIP, ALIGN, and SigLIP encoders for cross-modal retrieval with
a contrastive re-ranking stage.
"""

__version__ = "1.0.0"

from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline
from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker

__all__ = ["VisualGroundingPipeline", "ContrastiveReranker"]
