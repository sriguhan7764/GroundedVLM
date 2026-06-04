"""
Dataset classes for GroundedVLM.

Exports:
    COCOGroundingDataset  – COCO val2017 dataset with grounding annotations
    RoboticsQueryDataset  – Robotics manipulation retrieval dataset
"""

from groundedvlm.data.coco_dataset import COCOGroundingDataset
from groundedvlm.data.robotics_dataset import RoboticsQueryDataset

__all__ = ["COCOGroundingDataset", "RoboticsQueryDataset"]
