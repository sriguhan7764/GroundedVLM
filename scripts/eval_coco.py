"""
CLI: Evaluate a GroundedVLM pipeline on COCO val2017.

Usage:
    python scripts/eval_coco.py \
        --coco-root data/coco \
        --ann-file data/coco/annotations/instances_val2017.json \
        --detector dino \
        --output-dir outputs/evaluation

    # With contrastive re-ranking:
    python scripts/eval_coco.py \
        --coco-root data/coco \
        --ann-file data/coco/annotations/instances_val2017.json \
        --detector dino \
        --rerank \
        --encoder clip \
        --output-dir outputs/evaluation \
        --max-images 500
"""

import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundedvlm.pipeline.visual_grounding import VisualGroundingPipeline
from groundedvlm.evaluation.coco_eval import COCOGroundingEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _print_table(metrics: dict) -> None:
    """Pretty-print a metrics dictionary as a table."""
    col_width = max(len(k) for k in metrics) + 2
    print("\n" + "=" * (col_width + 14))
    print(f"{'Metric':<{col_width}} {'Value':>10}")
    print("-" * (col_width + 14))
    for key, val in metrics.items():
        print(f"{key:<{col_width}} {val:>10.4f}")
    print("=" * (col_width + 14) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GroundedVLM on COCO val2017.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coco-root",
        default="data/coco",
        help="Root directory for COCO (should contain val2017/ sub-directory).",
    )
    parser.add_argument(
        "--ann-file",
        default="data/coco/annotations/instances_val2017.json",
        help="Path to COCO val2017 instance annotation JSON.",
    )
    parser.add_argument(
        "--detector",
        default="dino",
        choices=["dino", "florence2"],
        help="Detector backend.",
    )
    parser.add_argument(
        "--encoder",
        default="clip",
        choices=["clip", "align", "siglip"],
        help="Encoder for re-ranking (only used with --rerank).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable contrastive re-ranking.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory to save metrics JSON.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images evaluated (useful for quick tests).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="COCO category names to evaluate on (default: all 80).",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/",
        help="Directory containing YAML configs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load configs ---
    if args.detector == "dino":
        detector_config = _load_yaml(os.path.join(args.config_dir, "grounding_dino.yaml"))
    else:
        detector_config = _load_yaml(os.path.join(args.config_dir, "florence2.yaml"))

    encoder_config = None
    if args.rerank:
        encoders_cfg = _load_yaml(os.path.join(args.config_dir, "encoders.yaml"))
        enc_cfg = encoders_cfg.get(args.encoder, {})
        encoder_config = {
            "encoder_type": args.encoder,
            **enc_cfg,
            "temperature": encoders_cfg.get("temperature", 0.07),
            "alpha": encoders_cfg.get("reranking_alpha", 0.5),
        }

    # --- Build pipeline ---
    logger.info("Building pipeline (detector=%s, rerank=%s).", args.detector, args.rerank)
    pipeline = VisualGroundingPipeline(
        detector_type=args.detector,
        detector_config=detector_config,
        encoder_config=encoder_config,
    )

    # --- Evaluate ---
    evaluator = COCOGroundingEvaluator(ann_file=args.ann_file)
    image_dir = os.path.join(args.coco_root, "val2017")

    logger.info("Starting COCO evaluation…")
    metrics = evaluator.evaluate(
        pipeline=pipeline,
        image_dir=image_dir,
        categories=args.categories,
        max_images=args.max_images,
    )

    # --- Save and display ---
    out_file = os.path.join(
        args.output_dir,
        f"coco_eval_{args.detector}{'_rerank_' + args.encoder if args.rerank else ''}.json",
    )
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to '%s'.", out_file)

    _print_table(metrics)


if __name__ == "__main__":
    main()
