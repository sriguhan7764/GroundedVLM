"""
Unit tests for pipeline classes: VisualGroundingPipeline and ContrastiveReranker.

HuggingFace and open_clip calls are mocked throughout.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(w: int = 100, h: int = 100) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_detection(n_boxes: int = 3, image_size=(100, 100)):
    """Create a Detection with n_boxes random boxes."""
    from groundedvlm.models.grounding_dino import Detection

    boxes = np.array(
        [[10.0 * i, 10.0 * i, 10.0 * i + 20, 10.0 * i + 20] for i in range(n_boxes)],
        dtype=np.float32,
    )
    scores = np.linspace(0.9, 0.5, n_boxes, dtype=np.float32)
    labels = [f"label_{i}" for i in range(n_boxes)]
    return Detection(boxes=boxes, scores=scores, labels=labels, image_size=image_size)


# ===========================================================================
# VisualGroundingPipeline tests
# ===========================================================================


class TestVisualGroundingPipelineDINO(unittest.TestCase):
    """Tests for VisualGroundingPipeline with Grounding DINO backend."""

    @patch("groundedvlm.pipeline.visual_grounding.GroundingDINODetector")
    def test_ground_returns_detection(self, MockDINO):
        """pipeline.ground() should return a Detection from the detector."""
        from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline
        from groundedvlm.models.grounding_dino import Detection

        expected = _make_detection(2)
        mock_detector = MockDINO.return_value
        mock_detector.detect.return_value = expected

        pipeline = VisualGroundingPipeline(
            detector_type="dino",
            detector_config={"model_id": "mock", "device": "cpu"},
        )

        image = _make_rgb_image()
        result = pipeline.ground(image, "cat. dog.")

        self.assertIsInstance(result, Detection)
        self.assertEqual(len(result), 2)
        mock_detector.detect.assert_called_once_with(image, "cat. dog.")

    @patch("groundedvlm.pipeline.visual_grounding.GroundingDINODetector")
    def test_ground_batch_length(self, MockDINO):
        """ground_batch should return one Detection per (image, query) pair."""
        from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline

        mock_detector = MockDINO.return_value
        mock_detector.detect.return_value = _make_detection(1)

        pipeline = VisualGroundingPipeline(
            detector_type="dino",
            detector_config={"device": "cpu"},
        )

        pairs = [(_make_rgb_image(), f"query {i}") for i in range(5)]
        results = pipeline.ground_batch(pairs)
        self.assertEqual(len(results), 5)

    def test_invalid_detector_type_raises(self):
        """Unknown detector_type should raise ValueError."""
        from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline

        with self.assertRaises(ValueError):
            VisualGroundingPipeline(
                detector_type="unknown_model",
                detector_config={"device": "cpu"},
            )


class TestVisualGroundingPipelineFlorence2(unittest.TestCase):
    """Tests for VisualGroundingPipeline with Florence-2 backend."""

    @patch("groundedvlm.pipeline.visual_grounding.Florence2Detector")
    def test_ground_uses_phrase_grounding(self, MockFlorence):
        """Florence-2 backend should call caption_to_phrase_grounding."""
        from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline

        expected = _make_detection(4)
        mock_detector = MockFlorence.return_value
        mock_detector.caption_to_phrase_grounding.return_value = expected

        pipeline = VisualGroundingPipeline(
            detector_type="florence2",
            detector_config={"device": "cpu"},
        )

        image = _make_rgb_image()
        result = pipeline.ground(image, "pick up the red cube")

        mock_detector.caption_to_phrase_grounding.assert_called_once_with(
            image, "pick up the red cube"
        )
        self.assertEqual(len(result), 4)


# ===========================================================================
# ContrastiveReranker tests
# ===========================================================================


class TestContrastiveReranker(unittest.TestCase):
    """Tests for ContrastiveReranker."""

    def _make_encoder(self, sim_values: list, dim: int = 64):
        """Build a mock encoder that returns pre-specified similarity values."""
        encoder = MagicMock()
        # encode_image returns normalised random embeddings
        encoder.encode_image.side_effect = lambda imgs: F.normalize(
            torch.randn(len(imgs), dim), p=2, dim=-1
        )
        # encode_text returns normalised random embeddings
        encoder.encode_text.side_effect = lambda txts: F.normalize(
            torch.randn(len(txts), dim), p=2, dim=-1
        )
        # compute_similarity returns a preset column vector
        sim_tensor = torch.tensor(sim_values, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        encoder.compute_similarity.return_value = sim_tensor
        return encoder

    def test_rerank_changes_order(self):
        """Reranker should reorder detections when rerank scores differ."""
        from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker

        detection = _make_detection(3, image_size=(100, 100))
        original_labels = list(detection.labels)

        # Detection scores: [0.9, 0.7, 0.5]
        # Rerank similarities: index 2 is best → should move to front
        encoder = self._make_encoder([0.1, 0.5, 0.95], dim=64)
        reranker = ContrastiveReranker(encoder=encoder, temperature=0.07, alpha=0.0)

        image = _make_rgb_image(100, 100)
        reranked = reranker.rerank(image, detection, "label_2")

        # With alpha=0 (pure rerank), index-2 item should be first
        self.assertEqual(reranked.labels[0], original_labels[2])

    def test_rerank_empty_detection_unchanged(self):
        """Reranking an empty Detection should return the same empty Detection."""
        from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker
        from groundedvlm.models.grounding_dino import Detection

        empty = Detection(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=[],
            image_size=(100, 100),
        )
        encoder = MagicMock()
        reranker = ContrastiveReranker(encoder=encoder)
        result = reranker.rerank(_make_rgb_image(), empty, "some query")
        self.assertEqual(len(result), 0)
        encoder.encode_image.assert_not_called()

    def test_rerank_preserves_box_count(self):
        """Reranking should not add or drop detections."""
        from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker

        n = 5
        detection = _make_detection(n)
        encoder = self._make_encoder([0.3, 0.8, 0.1, 0.95, 0.6], dim=32)
        reranker = ContrastiveReranker(encoder=encoder, alpha=0.5)

        reranked = reranker.rerank(_make_rgb_image(), detection, "query")
        self.assertEqual(len(reranked), n)
        self.assertEqual(reranked.boxes.shape, (n, 4))

    def test_rerank_with_alpha_one_preserves_order(self):
        """alpha=1.0 means pure detector score; order should be unchanged."""
        from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker

        # Detector scores: descending 0.9, 0.7, 0.5
        detection = _make_detection(3)
        # Rerank scores that would reverse the order
        encoder = self._make_encoder([0.1, 0.5, 0.99], dim=32)
        reranker = ContrastiveReranker(encoder=encoder, alpha=1.0)

        reranked = reranker.rerank(_make_rgb_image(), detection, "query")
        # With alpha=1.0, only detector scores matter → order unchanged
        self.assertEqual(reranked.labels[0], detection.labels[0])


class TestCropCandidates(unittest.TestCase):
    """Tests for ContrastiveReranker._crop_candidates."""

    def _make_reranker(self):
        from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker
        return ContrastiveReranker(encoder=MagicMock())

    def test_crop_dimensions(self):
        """Crops should have the correct pixel dimensions from xyxy boxes."""
        reranker = self._make_reranker()
        image = _make_rgb_image(200, 200)
        boxes = np.array([[10.0, 20.0, 80.0, 90.0]], dtype=np.float32)
        crops = reranker._crop_candidates(image, boxes)

        self.assertEqual(len(crops), 1)
        crop = crops[0]
        self.assertIsInstance(crop, Image.Image)
        self.assertEqual(crop.width, 70)   # 80 - 10
        self.assertEqual(crop.height, 70)  # 90 - 20

    def test_degenerate_box_returns_placeholder(self):
        """Degenerate (zero-area) boxes should return a 1x1 placeholder image."""
        reranker = self._make_reranker()
        image = _make_rgb_image(100, 100)
        boxes = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)  # zero area
        crops = reranker._crop_candidates(image, boxes)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].size, (1, 1))

    def test_multiple_crops(self):
        """Multiple non-overlapping boxes should yield one crop each."""
        reranker = self._make_reranker()
        image = _make_rgb_image(200, 200)
        boxes = np.array(
            [[0.0, 0.0, 50.0, 50.0],
             [100.0, 100.0, 160.0, 180.0]],
            dtype=np.float32,
        )
        crops = reranker._crop_candidates(image, boxes)
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0].size, (50, 50))
        self.assertEqual(crops[1].size, (60, 80))


if __name__ == "__main__":
    unittest.main()
