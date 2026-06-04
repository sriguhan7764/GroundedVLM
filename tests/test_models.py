"""
Unit tests for model classes: GroundingDINODetector, Florence2Detector,
CLIPEncoder, ALIGNEncoder, SigLIPEncoder.

HuggingFace and open_clip network calls are mocked at their import sites.
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_tensor(*shape, val: float = 0.5) -> torch.Tensor:
    return torch.full(shape, val, dtype=torch.float32)


# ===========================================================================
# GroundingDINODetector tests
# ===========================================================================


class TestGroundingDINODetector(unittest.TestCase):
    """Tests for GroundingDINODetector with mocked HuggingFace calls."""

    def _make_config(self) -> dict:
        return {
            "model_id": "mock/grounding-dino",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "device": "cpu",
            "nms_threshold": 0.5,
            "max_detections": 100,
        }

    def _build_mock_outputs(self, n_boxes: int = 3):
        """Build a fake HuggingFace output object."""
        return MagicMock()

    def _make_processor_return(self, n_boxes: int = 3):
        """Fake post_process_grounded_object_detection return value."""
        boxes = torch.tensor(
            [[10.0, 10.0, 50.0, 50.0],
             [60.0, 60.0, 120.0, 120.0],
             [15.0, 15.0, 55.0, 55.0]],  # overlaps with box 0 → NMS may remove
            dtype=torch.float32,
        )[:n_boxes]
        scores = torch.tensor([0.9, 0.8, 0.75], dtype=torch.float32)[:n_boxes]
        labels = ["cat", "dog", "cat"][:n_boxes]
        return [{"boxes": boxes, "scores": scores, "labels": labels}]

    @patch("groundedvlm.models.grounding_dino.AutoModelForZeroShotObjectDetection")
    @patch("groundedvlm.models.grounding_dino.AutoProcessor")
    def test_detect_output_shape(self, MockProcessor, MockModel):
        """Detection output should be a Detection with correct field types."""
        from groundedvlm.models.grounding_dino import GroundingDINODetector, Detection

        # Arrange
        mock_proc_instance = MockProcessor.from_pretrained.return_value
        mock_proc_instance.return_value = {}  # inputs after processing
        mock_proc_instance.post_process_grounded_object_detection.return_value = (
            self._make_processor_return(2)
        )

        mock_model_instance = MockModel.from_pretrained.return_value
        mock_model_instance.return_value = MagicMock()

        detector = GroundingDINODetector(self._make_config())

        image = _make_rgb_image()
        detection = detector.detect(image, "cat. dog.")

        self.assertIsInstance(detection, Detection)
        self.assertIsInstance(detection.boxes, np.ndarray)
        self.assertIsInstance(detection.scores, np.ndarray)
        self.assertIsInstance(detection.labels, list)
        self.assertEqual(detection.boxes.ndim, 2)
        self.assertEqual(detection.boxes.shape[1], 4)
        self.assertEqual(len(detection.scores), len(detection.labels))
        self.assertEqual(len(detection.scores), detection.boxes.shape[0])
        self.assertEqual(detection.image_size, (image.height, image.width))

    @patch("groundedvlm.models.grounding_dino.AutoModelForZeroShotObjectDetection")
    @patch("groundedvlm.models.grounding_dino.AutoProcessor")
    def test_detect_nms_removes_overlapping_box(self, MockProcessor, MockModel):
        """NMS should suppress the heavily overlapping third box."""
        from groundedvlm.models.grounding_dino import GroundingDINODetector

        # Box 0 and Box 2 overlap heavily (IoU > 0.5)
        boxes_with_overlap = torch.tensor(
            [[10.0, 10.0, 60.0, 60.0],   # high score
             [200.0, 200.0, 250.0, 250.0],  # no overlap
             [12.0, 12.0, 62.0, 62.0]],   # overlaps box 0 — should be suppressed
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        labels = ["cat", "dog", "cat"]

        mock_proc = MockProcessor.from_pretrained.return_value
        mock_proc.return_value = {}
        mock_proc.post_process_grounded_object_detection.return_value = [
            {"boxes": boxes_with_overlap, "scores": scores, "labels": labels}
        ]

        MockModel.from_pretrained.return_value.return_value = MagicMock()

        config = self._make_config()
        config["nms_threshold"] = 0.5
        detector = GroundingDINODetector(config)

        detection = detector.detect(_make_rgb_image(300, 300), "cat. dog.")
        # Only 2 should survive NMS (box 0 suppresses box 2)
        self.assertEqual(len(detection), 2)

    @patch("groundedvlm.models.grounding_dino.AutoModelForZeroShotObjectDetection")
    @patch("groundedvlm.models.grounding_dino.AutoProcessor")
    def test_detect_empty_response(self, MockProcessor, MockModel):
        """Empty detector output should yield a Detection with zero boxes."""
        from groundedvlm.models.grounding_dino import GroundingDINODetector

        mock_proc = MockProcessor.from_pretrained.return_value
        mock_proc.return_value = {}
        mock_proc.post_process_grounded_object_detection.return_value = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": []}
        ]
        MockModel.from_pretrained.return_value.return_value = MagicMock()

        detector = GroundingDINODetector(self._make_config())
        detection = detector.detect(_make_rgb_image(), "nonexistent")
        self.assertEqual(len(detection), 0)
        self.assertEqual(detection.boxes.shape, (0, 4))

    @patch("groundedvlm.models.grounding_dino.AutoModelForZeroShotObjectDetection")
    @patch("groundedvlm.models.grounding_dino.AutoProcessor")
    def test_detect_batch_returns_list(self, MockProcessor, MockModel):
        """detect_batch should return one Detection per image."""
        from groundedvlm.models.grounding_dino import GroundingDINODetector

        mock_proc = MockProcessor.from_pretrained.return_value
        mock_proc.return_value = {}
        mock_proc.post_process_grounded_object_detection.return_value = (
            self._make_processor_return(1)
        )
        MockModel.from_pretrained.return_value.return_value = MagicMock()

        detector = GroundingDINODetector(self._make_config())
        images = [_make_rgb_image() for _ in range(3)]
        prompts = ["cat.", "dog.", "bird."]
        results = detector.detect_batch(images, prompts)
        self.assertEqual(len(results), 3)


# ===========================================================================
# Florence2Detector tests
# ===========================================================================


class TestFlorence2Detector(unittest.TestCase):
    """Tests for Florence2Detector._parse_od_response."""

    def _make_config(self) -> dict:
        return {
            "model_id": "mock/florence-2",
            "task_prompt": "<OD>",
            "phrase_grounding_prompt": "<CAPTION_TO_PHRASE_GROUNDING>",
            "device": "cpu",
            "max_new_tokens": 64,
            "num_beams": 1,
        }

    @patch("groundedvlm.models.florence2.AutoModelForCausalLM")
    @patch("groundedvlm.models.florence2.AutoProcessor")
    def test_parse_od_response_pixel_coords(self, MockProcessor, MockModel):
        """_parse_od_response converts [0,1000] boxes to pixel coordinates."""
        from groundedvlm.models.florence2 import Florence2Detector

        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        detector = Florence2Detector(self._make_config())

        # Simulate a post_process_generation response
        response = {
            "<OD>": {
                "bboxes": [[0.0, 0.0, 500.0, 500.0], [250.0, 100.0, 750.0, 800.0]],
                "labels": ["cube", "box"],
            }
        }
        image_size = (400, 640)  # H, W
        detection = detector._parse_od_response(response, image_size)

        self.assertEqual(len(detection), 2)
        # Box 0: x1=0, y1=0, x2=500/1000*640=320, y2=500/1000*400=200
        np.testing.assert_allclose(detection.boxes[0], [0.0, 0.0, 320.0, 200.0], atol=0.1)
        # Box 1: x1=250/1000*640=160, y1=100/1000*400=40, x2=480, y2=320
        np.testing.assert_allclose(detection.boxes[1], [160.0, 40.0, 480.0, 320.0], atol=0.1)

    @patch("groundedvlm.models.florence2.AutoModelForCausalLM")
    @patch("groundedvlm.models.florence2.AutoProcessor")
    def test_parse_od_response_empty(self, MockProcessor, MockModel):
        """Empty response dict should yield a zero-detection Detection."""
        from groundedvlm.models.florence2 import Florence2Detector

        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        detector = Florence2Detector(self._make_config())
        response = {"<OD>": {"bboxes": [], "labels": []}}
        detection = detector._parse_od_response(response, (100, 100))
        self.assertEqual(len(detection), 0)

    @patch("groundedvlm.models.florence2.AutoModelForCausalLM")
    @patch("groundedvlm.models.florence2.AutoProcessor")
    def test_parse_od_response_scores_are_one(self, MockProcessor, MockModel):
        """Florence-2 confidence scores should all be 1.0 (not emitted by model)."""
        from groundedvlm.models.florence2 import Florence2Detector

        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        detector = Florence2Detector(self._make_config())
        response = {"<OD>": {"bboxes": [[100.0, 100.0, 500.0, 500.0]], "labels": ["tool"]}}
        detection = detector._parse_od_response(response, (200, 200))
        np.testing.assert_array_equal(detection.scores, [1.0])


# ===========================================================================
# CLIPEncoder tests
# ===========================================================================


class TestCLIPEncoder(unittest.TestCase):
    """Tests for CLIPEncoder using mocked open_clip."""

    def _make_encoder(self, dim: int = 512):
        """Build a CLIPEncoder with fully mocked open_clip internals."""
        with patch("groundedvlm.models.clip_encoder.open_clip") as mock_oc:
            # create_model_and_transforms returns (model, preprocess_train, preprocess_val)
            mock_model = MagicMock()
            mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
            mock_oc.create_model_and_transforms.return_value = (
                mock_model,
                mock_preprocess,
                mock_preprocess,
            )
            mock_oc.get_tokenizer.return_value = MagicMock(
                return_value=torch.zeros(1, 77, dtype=torch.long)
            )

            # Stub encode_image and encode_text to return unit vectors
            mock_model.encode_image.side_effect = lambda x: torch.randn(x.shape[0], dim)
            mock_model.encode_text.side_effect = lambda x: torch.randn(x.shape[0], dim)

            from groundedvlm.models.clip_encoder import CLIPEncoder
            encoder = CLIPEncoder(model_name="ViT-B-32", pretrained="openai", device="cpu")
            encoder.model = mock_model
            encoder.preprocess = mock_preprocess
            encoder.tokenizer = mock_oc.get_tokenizer.return_value
            return encoder, dim

    def test_encode_image_shape(self):
        """encode_image should return (N, D) L2-normalised tensor."""
        encoder, dim = self._make_encoder(dim=512)
        images = [_make_rgb_image() for _ in range(4)]

        with patch.object(encoder.model, "encode_image", return_value=torch.randn(4, 512)):
            embeds = encoder.encode_image(images)
        self.assertEqual(embeds.shape, (4, 512))

    def test_encode_text_shape(self):
        """encode_text should return (N, D) L2-normalised tensor."""
        encoder, dim = self._make_encoder(dim=512)
        texts = ["cat", "dog", "bird"]

        with patch.object(encoder.model, "encode_text", return_value=torch.randn(3, 512)):
            embeds = encoder.encode_text(texts)
        self.assertEqual(embeds.shape, (3, 512))

    def test_encode_image_is_normalised(self):
        """Embeddings from encode_image should be unit-normalised (L2 norm ≈ 1)."""
        encoder, dim = self._make_encoder(dim=512)
        raw = torch.randn(2, 512)

        with patch.object(encoder.model, "encode_image", return_value=raw):
            embeds = encoder.encode_image([_make_rgb_image(), _make_rgb_image()])
        norms = embeds.norm(dim=-1)
        for n in norms.tolist():
            self.assertAlmostEqual(n, 1.0, places=5)

    def test_compute_similarity_shape(self):
        """compute_similarity should return (N, M) matrix."""
        encoder, dim = self._make_encoder(dim=256)
        import torch.nn.functional as F
        img_e = F.normalize(torch.randn(5, 256), p=2, dim=-1)
        txt_e = F.normalize(torch.randn(3, 256), p=2, dim=-1)
        sim = encoder.compute_similarity(img_e, txt_e)
        self.assertEqual(sim.shape, (5, 3))

    def test_encode_image_batch_concatenates(self):
        """encode_image_batch should process in batches and concatenate."""
        encoder, dim = self._make_encoder(dim=128)
        images = [_make_rgb_image() for _ in range(10)]
        raw = torch.randn(10, 128)

        call_count = {"n": 0}

        def side_effect(x):
            call_count["n"] += 1
            return torch.randn(x.shape[0], 128)

        with patch.object(encoder.model, "encode_image", side_effect=side_effect):
            embeds = encoder.encode_image_batch(images, batch_size=4)
        # 10 images with batch_size=4 → ceil(10/4)=3 forward passes
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(embeds.shape[0], 10)


# ===========================================================================
# ALIGNEncoder tests
# ===========================================================================


class TestALIGNEncoder(unittest.TestCase):
    """Tests for ALIGNEncoder with mocked HuggingFace calls."""

    @patch("groundedvlm.models.align_encoder.AlignModel")
    @patch("groundedvlm.models.align_encoder.AlignProcessor")
    def test_encode_image_shape(self, MockProc, MockModel):
        """encode_image should return (N, D) normalised tensor."""
        from groundedvlm.models.align_encoder import ALIGNEncoder

        dim = 640
        mock_model = MockModel.from_pretrained.return_value
        mock_model.get_image_features.return_value = torch.randn(2, dim)
        MockProc.from_pretrained.return_value.return_value = {"pixel_values": torch.zeros(2, 3, 224, 224)}

        encoder = ALIGNEncoder(model_id="mock/align", device="cpu")
        images = [_make_rgb_image(), _make_rgb_image()]
        embeds = encoder.encode_image(images)

        self.assertEqual(embeds.shape, (2, dim))
        norms = embeds.norm(dim=-1)
        for n in norms.tolist():
            self.assertAlmostEqual(n, 1.0, places=5)

    @patch("groundedvlm.models.align_encoder.AlignModel")
    @patch("groundedvlm.models.align_encoder.AlignProcessor")
    def test_encode_text_shape(self, MockProc, MockModel):
        """encode_text should return (N, D) normalised tensor."""
        from groundedvlm.models.align_encoder import ALIGNEncoder

        dim = 640
        mock_model = MockModel.from_pretrained.return_value
        mock_model.get_text_features.return_value = torch.randn(3, dim)
        MockProc.from_pretrained.return_value.return_value = {"input_ids": torch.zeros(3, 32, dtype=torch.long)}

        encoder = ALIGNEncoder(model_id="mock/align", device="cpu")
        embeds = encoder.encode_text(["a", "b", "c"])
        self.assertEqual(embeds.shape, (3, dim))

    @patch("groundedvlm.models.align_encoder.AlignModel")
    @patch("groundedvlm.models.align_encoder.AlignProcessor")
    def test_compute_similarity_shape(self, MockProc, MockModel):
        """compute_similarity should return (N, M) matrix."""
        from groundedvlm.models.align_encoder import ALIGNEncoder
        import torch.nn.functional as F

        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()

        encoder = ALIGNEncoder(model_id="mock/align", device="cpu")
        img_e = F.normalize(torch.randn(4, 256), p=2, dim=-1)
        txt_e = F.normalize(torch.randn(2, 256), p=2, dim=-1)
        sim = encoder.compute_similarity(img_e, txt_e)
        self.assertEqual(sim.shape, (4, 2))


# ===========================================================================
# SigLIPEncoder tests
# ===========================================================================


class TestSigLIPEncoder(unittest.TestCase):
    """Tests for SigLIPEncoder with mocked HuggingFace calls."""

    @patch("groundedvlm.models.siglip_encoder.SiglipModel")
    @patch("groundedvlm.models.siglip_encoder.SiglipProcessor")
    def test_encode_image_shape(self, MockProc, MockModel):
        """encode_image should return (N, D) normalised tensor."""
        from groundedvlm.models.siglip_encoder import SigLIPEncoder

        dim = 1024
        mock_model = MockModel.from_pretrained.return_value
        mock_model.get_image_features.return_value = torch.randn(2, dim)
        MockProc.from_pretrained.return_value.return_value = {"pixel_values": torch.zeros(2, 3, 384, 384)}

        encoder = SigLIPEncoder(model_id="mock/siglip", device="cpu")
        embeds = encoder.encode_image([_make_rgb_image(), _make_rgb_image()])
        self.assertEqual(embeds.shape, (2, dim))

    @patch("groundedvlm.models.siglip_encoder.SiglipModel")
    @patch("groundedvlm.models.siglip_encoder.SiglipProcessor")
    def test_encode_text_shape(self, MockProc, MockModel):
        """encode_text should return (N, D) normalised tensor."""
        from groundedvlm.models.siglip_encoder import SigLIPEncoder

        dim = 1024
        mock_model = MockModel.from_pretrained.return_value
        mock_model.get_text_features.return_value = torch.randn(3, dim)
        MockProc.from_pretrained.return_value.return_value = {"input_ids": torch.zeros(3, 64, dtype=torch.long)}

        encoder = SigLIPEncoder(model_id="mock/siglip", device="cpu")
        embeds = encoder.encode_text(["pick up cube", "grasp wrench", "open door"])
        self.assertEqual(embeds.shape, (3, dim))

    @patch("groundedvlm.models.siglip_encoder.SiglipModel")
    @patch("groundedvlm.models.siglip_encoder.SiglipProcessor")
    def test_compute_similarity_returns_dot_products(self, MockProc, MockModel):
        """SigLIP compute_similarity returns raw dot products (no softmax)."""
        from groundedvlm.models.siglip_encoder import SigLIPEncoder
        import torch.nn.functional as F

        MockModel.from_pretrained.return_value = MagicMock()
        MockProc.from_pretrained.return_value = MagicMock()

        encoder = SigLIPEncoder(model_id="mock/siglip", device="cpu")
        img_e = F.normalize(torch.randn(3, 64), p=2, dim=-1)
        txt_e = F.normalize(torch.randn(5, 64), p=2, dim=-1)
        sim = encoder.compute_similarity(img_e, txt_e)
        self.assertEqual(sim.shape, (3, 5))
        # Values should lie in [-1, 1] since inputs are L2 normalised
        self.assertTrue(sim.abs().max().item() <= 1.0 + 1e-5)


if __name__ == "__main__":
    unittest.main()
