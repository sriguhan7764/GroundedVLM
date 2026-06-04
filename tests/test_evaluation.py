"""
Unit tests for evaluation classes: COCOGroundingEvaluator and RetrievalEvaluator.

pycocotools calls are mocked for COCO tests.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from groundedvlm.utils.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_average_precision,
    compute_map,
)
from groundedvlm.models.grounding_dino import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(n_boxes: int = 3, image_size=(640, 480)):
    boxes = np.array(
        [[float(i * 10), float(i * 10), float(i * 10 + 30), float(i * 10 + 30)]
         for i in range(n_boxes)],
        dtype=np.float32,
    )
    scores = np.linspace(0.9, 0.5, n_boxes, dtype=np.float32)
    labels = [f"cat" if i % 2 == 0 else "dog" for i in range(n_boxes)]
    return Detection(boxes=boxes, scores=scores, labels=labels, image_size=image_size)


# ===========================================================================
# COCOGroundingEvaluator tests
# ===========================================================================


class TestCOCOGroundingEvaluatorFormat(unittest.TestCase):
    """Tests for COCOGroundingEvaluator._format_predictions."""

    def _make_evaluator(self):
        with patch("groundedvlm.evaluation.coco_eval.COCO") as MockCOCO:
            MockCOCO.return_value = MagicMock()
            from groundedvlm.evaluation.coco_eval import COCOGroundingEvaluator
            evaluator = COCOGroundingEvaluator(ann_file="mock/ann.json")
            return evaluator

    def test_format_predictions_structure(self):
        """_format_predictions should return COCO-format dicts."""
        evaluator = self._make_evaluator()
        detection = _make_detection(3)
        cat_names = ["cat", "dog"]
        cat_ids = [1, 2]

        preds = evaluator._format_predictions(detection, image_id=42, cat_names=cat_names, cat_ids=cat_ids)

        # Each prediction should have required COCO keys
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 3)

        for pred in preds:
            self.assertIn("image_id", pred)
            self.assertIn("category_id", pred)
            self.assertIn("bbox", pred)
            self.assertIn("score", pred)
            self.assertEqual(pred["image_id"], 42)
            # bbox is [x, y, w, h]
            bbox = pred["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertGreaterEqual(bbox[2], 0.0)  # w >= 0
            self.assertGreaterEqual(bbox[3], 0.0)  # h >= 0

    def test_format_predictions_category_mapping(self):
        """Category IDs should map correctly from label strings."""
        evaluator = self._make_evaluator()
        # Detection has labels ["cat", "dog", "cat"]
        detection = _make_detection(3)
        cat_names = ["cat", "dog"]
        cat_ids = [17, 18]  # COCO cat=17, dog=18

        preds = evaluator._format_predictions(
            detection, image_id=1, cat_names=cat_names, cat_ids=cat_ids
        )
        # First label is "cat" → category_id=17
        self.assertEqual(preds[0]["category_id"], 17)
        # Second label is "dog" → category_id=18
        self.assertEqual(preds[1]["category_id"], 18)

    def test_format_predictions_xywh_conversion(self):
        """Bounding boxes must be converted from xyxy to xywh."""
        evaluator = self._make_evaluator()
        # Single box: [10, 10, 50, 80] in xyxy → [10, 10, 40, 70] in xywh
        boxes = np.array([[10.0, 10.0, 50.0, 80.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        detection = Detection(
            boxes=boxes, scores=scores, labels=["cat"], image_size=(100, 100)
        )
        preds = evaluator._format_predictions(detection, image_id=5, cat_names=["cat"], cat_ids=[17])
        bbox = preds[0]["bbox"]
        self.assertAlmostEqual(bbox[0], 10.0, places=3)
        self.assertAlmostEqual(bbox[1], 10.0, places=3)
        self.assertAlmostEqual(bbox[2], 40.0, places=3)  # w = 50 - 10
        self.assertAlmostEqual(bbox[3], 70.0, places=3)  # h = 80 - 10


# ===========================================================================
# RetrievalEvaluator tests
# ===========================================================================


class TestRetrievalEvaluator(unittest.TestCase):
    """Tests for RetrievalEvaluator using mocked encoders."""

    def _make_encoder(self, n_images: int = 10, dim: int = 64):
        """Return a mock encoder that produces random normalised embeddings."""
        import torch.nn.functional as F

        encoder = MagicMock()

        def encode_image(imgs):
            n = len(imgs)
            return F.normalize(torch.randn(n, dim), p=2, dim=-1)

        def encode_text(txts):
            n = len(txts)
            return F.normalize(torch.randn(n, dim), p=2, dim=-1)

        encoder.encode_image.side_effect = encode_image
        encoder.encode_text.side_effect = encode_text
        return encoder

    def _make_dataset(self, n_images: int = 10, n_queries: int = 5):
        """Minimal dataset stub compatible with RetrievalEvaluator.evaluate()."""
        from PIL import Image as PILImage
        import numpy as np

        dataset = MagicMock()
        dataset.images = [
            PILImage.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(n_images)
        ]
        dataset.queries = [f"query_{i}" for i in range(n_queries)]
        # Each query has exactly one relevant image (its own index, cyclic)
        dataset.relevant_indices = [[i % n_images] for i in range(n_queries)]
        return dataset

    def test_evaluate_returns_recall_and_mrr(self):
        """evaluate() should return a dict with Recall@k keys and MRR."""
        from groundedvlm.evaluation.retrieval_eval import RetrievalEvaluator

        encoder = self._make_encoder()
        evaluator = RetrievalEvaluator(encoder=encoder, k_values=[1, 5])
        dataset = self._make_dataset(n_images=10, n_queries=5)

        results = evaluator.evaluate(dataset)

        self.assertIn("Recall@1", results)
        self.assertIn("Recall@5", results)
        self.assertIn("MRR", results)
        for val in results.values():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_retrieve_returns_k_indices(self):
        """_retrieve should return exactly k indices."""
        from groundedvlm.evaluation.retrieval_eval import RetrievalEvaluator
        import torch.nn.functional as F

        encoder = self._make_encoder()
        evaluator = RetrievalEvaluator(encoder=encoder, k_values=[3])

        dim = 64
        gallery = F.normalize(torch.randn(20, dim), p=2, dim=-1)
        query = F.normalize(torch.randn(dim), p=2, dim=-1)

        indices = evaluator._retrieve(query, gallery, k=5)
        self.assertEqual(len(indices), 5)
        # All indices should be valid
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 20)


# ===========================================================================
# Metrics function tests
# ===========================================================================


class TestRecallAtK(unittest.TestCase):
    """Tests for compute_recall_at_k from groundedvlm.utils.metrics."""

    def test_perfect_recall(self):
        """Relevant item at rank 1 → Recall@1 = 1.0."""
        retrieved = [5, 1, 3, 7, 2]
        relevant = [5]
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=1), 1.0)

    def test_recall_miss(self):
        """Relevant item not in top-k → Recall@k = 0.0."""
        retrieved = [0, 1, 2, 3, 4]
        relevant = [99]
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=5), 0.0)

    def test_partial_recall(self):
        """Two relevant items; only one in top-k → Recall@k = 0.5."""
        retrieved = [5, 2, 8, 11, 7]
        relevant = [5, 99]
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=5), 0.5)

    def test_empty_relevant(self):
        """Empty relevant set → Recall = 0.0."""
        self.assertAlmostEqual(compute_recall_at_k([1, 2, 3], [], k=3), 0.0)

    def test_recall_at_zero_k(self):
        """k=0 → nothing retrieved → Recall@0 = 0.0."""
        self.assertAlmostEqual(compute_recall_at_k([1, 2, 3], [1], k=0), 0.0)


class TestMRR(unittest.TestCase):
    """Tests for compute_mrr from groundedvlm.utils.metrics."""

    def test_first_rank(self):
        """Relevant at rank 1 → reciprocal rank = 1.0."""
        retrieved = [[10, 2, 3]]
        relevant = [[10]]
        self.assertAlmostEqual(compute_mrr(retrieved, relevant), 1.0)

    def test_second_rank(self):
        """Relevant at rank 2 → reciprocal rank = 0.5."""
        retrieved = [[1, 10, 3]]
        relevant = [[10]]
        self.assertAlmostEqual(compute_mrr(retrieved, relevant), 0.5)

    def test_multiple_queries(self):
        """MRR is averaged over queries."""
        retrieved = [[10, 2], [1, 20]]
        relevant = [[10], [20]]
        # Query 0: rank 1 → rr=1.0; Query 1: rank 2 → rr=0.5; MRR=0.75
        self.assertAlmostEqual(compute_mrr(retrieved, relevant), 0.75)

    def test_no_relevant_found(self):
        """No relevant item in retrieved → MRR = 0.0."""
        retrieved = [[1, 2, 3]]
        relevant = [[99]]
        self.assertAlmostEqual(compute_mrr(retrieved, relevant), 0.0)

    def test_empty_lists(self):
        """Empty input → MRR = 0.0."""
        self.assertAlmostEqual(compute_mrr([], []), 0.0)


class TestAveragePrecision(unittest.TestCase):
    """Tests for compute_average_precision."""

    def test_perfect_ap(self):
        """Perfect retrieval → AP = 1.0."""
        retrieved = [1, 2, 3]
        relevant = [1, 2, 3]
        self.assertAlmostEqual(compute_average_precision(retrieved, relevant), 1.0)

    def test_zero_ap(self):
        """No matches → AP = 0.0."""
        retrieved = [4, 5, 6]
        relevant = [1, 2, 3]
        self.assertAlmostEqual(compute_average_precision(retrieved, relevant), 0.0)

    def test_partial_ap(self):
        """One match at rank 1 out of two relevant → AP > 0."""
        retrieved = [1, 99, 99]
        relevant = [1, 2]
        ap = compute_average_precision(retrieved, relevant)
        self.assertGreater(ap, 0.0)
        self.assertLess(ap, 1.0)


class TestMAP(unittest.TestCase):
    """Tests for compute_map."""

    def test_map_over_queries(self):
        """mAP should be mean of per-query APs."""
        all_retrieved = [[1, 2], [3, 4]]
        all_relevant = [[1], [3]]
        # Query 0: rank-1 hit → AP = 1.0
        # Query 1: rank-1 hit → AP = 1.0
        self.assertAlmostEqual(compute_map(all_retrieved, all_relevant), 1.0)

    def test_map_empty(self):
        self.assertAlmostEqual(compute_map([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
