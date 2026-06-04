"""
CLI: Run GroundedVLM visual grounding on a single image.

Usage:
    python scripts/run_grounding.py \
        --image path/to/image.jpg \
        --query "red cube . blue block" \
        --detector dino \
        --output output.jpg

    # With contrastive re-ranking:
    python scripts/run_grounding.py \
        --image path/to/image.jpg \
        --query "red cube" \
        --detector dino \
        --rerank \
        --encoder clip \
        --output output.jpg
"""

import argparse
import logging
import os
import sys

import yaml
from PIL import Image

# Ensure the project root is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline
from groundedvlm.utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GroundedVLM visual grounding on a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--query",
        required=True,
        help="Text query / dot-separated categories, e.g. 'cat. dog. car.'",
    )
    parser.add_argument(
        "--detector",
        default="dino",
        choices=["dino", "florence2"],
        help="Detector backend to use.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable contrastive re-ranking after detection.",
    )
    parser.add_argument(
        "--encoder",
        default="clip",
        choices=["clip", "align", "siglip"],
        help="Encoder to use for re-ranking (only used with --rerank).",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/",
        help="Directory containing YAML configuration files.",
    )
    parser.add_argument(
        "--output",
        default="output.jpg",
        help="Path to save the annotated output image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    args = parser.parse_args()

    # --- Load configs ---
    config_dir = args.config_dir
    if args.detector == "dino":
        detector_cfg_path = os.path.join(config_dir, "grounding_dino.yaml")
    else:
        detector_cfg_path = os.path.join(config_dir, "florence2.yaml")

    detector_config = _load_yaml(detector_cfg_path)
    # Apply CLI threshold override
    if args.detector == "dino":
        detector_config["box_threshold"] = args.threshold

    encoder_config = None
    if args.rerank:
        encoders_cfg = _load_yaml(os.path.join(config_dir, "encoders.yaml"))
        enc_cfg = encoders_cfg.get(args.encoder, {})
        encoder_config = {
            "encoder_type": args.encoder,
            **enc_cfg,
            "temperature": encoders_cfg.get("temperature", 0.07),
            "alpha": encoders_cfg.get("reranking_alpha", 0.5),
        }

    # --- Load image ---
    if not os.path.isfile(args.image):
        logger.error("Image file not found: %s", args.image)
        sys.exit(1)

    image = Image.open(args.image).convert("RGB")
    logger.info("Loaded image '%s' (%dx%d).", args.image, image.width, image.height)

    # --- Build pipeline ---
    logger.info("Building VisualGroundingPipeline (detector=%s, rerank=%s).", args.detector, args.rerank)
    pipeline = VisualGroundingPipeline(
        detector_type=args.detector,
        detector_config=detector_config,
        encoder_config=encoder_config,
    )

    # --- Run detection ---
    logger.info("Running grounding for query: '%s'", args.query)
    detection = pipeline.ground(image, args.query)

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Detections: {len(detection)}")
    print(f"{'='*60}")
    for i, (box, score, label) in enumerate(
        zip(detection.boxes, detection.scores, detection.labels)
    ):
        x1, y1, x2, y2 = box
        print(
            f"  [{i+1:2d}] {label:25s}  score={score:.3f}  "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
    print(f"{'='*60}\n")

    # --- Visualise and save ---
    vis = Visualizer()
    annotated = vis.draw_detections(image, detection, save_path=args.output)
    logger.info("Saved annotated image to '%s'.", args.output)


if __name__ == "__main__":
    main()
