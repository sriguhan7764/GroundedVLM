"""
Model classes for GroundedVLM.

Exports:
    Detection          – dataclass for detection results
    GroundingDINODetector – Grounding DINO open-vocabulary detector
    Florence2Detector  – Florence-2 open-vocabulary detector
    CLIPEncoder        – CLIP image/text encoder (open_clip)
    ALIGNEncoder       – ALIGN image/text encoder (HuggingFace)
    SigLIPEncoder      – SigLIP image/text encoder (HuggingFace)
"""

from groundedvlm.models.grounding_dino import Detection, GroundingDINODetector
from groundedvlm.models.florence2 import Florence2Detector
from groundedvlm.models.clip_encoder import CLIPEncoder
from groundedvlm.models.align_encoder import ALIGNEncoder
from groundedvlm.models.siglip_encoder import SigLIPEncoder

__all__ = [
    "Detection",
    "GroundingDINODetector",
    "Florence2Detector",
    "CLIPEncoder",
    "ALIGNEncoder",
    "SigLIPEncoder",
]
