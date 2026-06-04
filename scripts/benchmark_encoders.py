"""
CLI: Benchmark CLIP, ALIGN, and SigLIP on cross-modal retrieval.

Usage:
    python scripts/benchmark_encoders.py \
        --dataset-path data/robotics_queries \
        --encoders clip,align,siglip \
        --output-dir outputs/benchmark

    # With re-ranking and custom k values:
    python scripts/benchmark_encoders.py \
        --dataset-path data/robotics_queries \
        --encoders clip,siglip \
        --with-reranking \
        --k-values 1 5 10 \
        --output-dir outputs/benchmark
"""

import argparse
import csv
import logging
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundedvlm.data.robotics_dataset import RoboticsQueryDataset
from groundedvlm.evaluation.retrieval_eval import RetrievalEvaluator
from groundedvlm.models.clip_encoder import CLIPEncoder
from groundedvlm.models.align_encoder import ALIGNEncoder
from groundedvlm.models.siglip_encoder import SigLIPEncoder
from groundedvlm.pipeline.contrastive_reranking import ContrastiveReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_encoder(encoder_name: str, encoders_cfg: dict) -> Any:
    """Instantiate the named encoder from config."""
    name = encoder_name.lower()
    device = encoders_cfg.get(name, {}).get("device", "cuda")

    if name == "clip":
        cfg = encoders_cfg.get("clip", {})
        return CLIPEncoder(
            model_name=cfg.get("model_name", "ViT-L-14"),
            pretrained=cfg.get("pretrained", "openai"),
            device=device,
        )
    elif name == "align":
        cfg = encoders_cfg.get("align", {})
        return ALIGNEncoder(
            model_id=cfg.get("model_id", "kakaobrain/align-base"),
            device=device,
        )
    elif name == "siglip":
        cfg = encoders_cfg.get("siglip", {})
        return SigLIPEncoder(
            model_id=cfg.get("model_id", "google/siglip-large-patch16-384"),
            device=device,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose clip, align, or siglip.")


def _print_comparison_table(
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> None:
    """Print a formatted comparison table of all encoder results."""
    k_headers = [f"R@{k}" for k in k_values]
    col_w = 12
    enc_w = 20
    headers = ["Encoder"] + k_headers + ["MRR"]
    separator = "-" * (enc_w + col_w * len(headers))

    print("\n" + "=" * (enc_w + col_w * len(headers)))
    print(f"{'Encoder':<{enc_w}}" + "".join(f"{h:>{col_w}}" for h in k_headers + ["MRR"]))
    print(separator)
    for enc_name, metrics in results.items():
        row = f"{enc_name:<{enc_w}}"
        for k in k_values:
            val = metrics.get(f"Recall@{k}", 0.0)
            row += f"{val:>{col_w}.4f}"
        row += f"{metrics.get('MRR', 0.0):>{col_w}.4f}"
        print(row)
    print("=" * (enc_w + col_w * len(headers)) + "\n")


def _save_csv(results: Dict[str, Dict[str, float]], out_path: str) -> None:
    """Save results as a CSV file."""
    if not results:
        return
    all_keys = list(next(iter(results.values())).keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder"] + all_keys)
        writer.writeheader()
        for enc_name, metrics in results.items():
            writer.writerow({"encoder": enc_name, **metrics})
    logger.info("Results saved to '%s'.", out_path)


def _save_bar_chart(
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    output_dir: str,
) -> None:
    """Save a grouped bar chart comparing Recall@k across encoders."""
    encoders = list(results.keys())
    n_enc = len(encoders)
    x = range(len(k_values))
    width = 0.8 / n_enc

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, enc_name in enumerate(encoders):
        values = [results[enc_name].get(f"Recall@{k}", 0.0) for k in k_values]
        offsets = [pos + (i - n_enc / 2 + 0.5) * width for pos in x]
        ax.bar(offsets, values, width=width, label=enc_name)

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Encoder Benchmark: Recall@k on Robotics Query Dataset")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"R@{k}" for k in k_values])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "recall_at_k_comparison.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Bar chart saved to '%s'.", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CLIP, ALIGN, SigLIP on robotics retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default="data/robotics_queries",
        help="Path to the robotics query dataset directory.",
    )
    parser.add_argument(
        "--encoders",
        default="clip,align,siglip",
        help="Comma-separated list of encoders to benchmark.",
    )
    parser.add_argument(
        "--with-reranking",
        action="store_true",
        help="Wrap each encoder in a ContrastiveReranker during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark",
        help="Directory to save results CSV and plots.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Recall@k cutoff values.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/",
        help="Directory containing YAML configuration files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    encoders_cfg = _load_yaml(os.path.join(args.config_dir, "encoders.yaml"))

    encoder_names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    logger.info("Encoders to benchmark: %s", encoder_names)

    # Load dataset (synthetic if path does not exist)
    logger.info("Loading robotics query dataset from '%s'.", args.dataset_path)
    dataset = RoboticsQueryDataset(root=args.dataset_path, split="val")

    all_results: Dict[str, Dict[str, float]] = {}

    for enc_name in encoder_names:
        logger.info("Evaluating encoder: %s", enc_name)
        try:
            encoder = _build_encoder(enc_name, encoders_cfg)
        except Exception as exc:
            logger.error("Failed to load encoder '%s': %s", enc_name, exc)
            continue

        evaluator = RetrievalEvaluator(encoder=encoder, k_values=args.k_values)
        metrics = evaluator.evaluate(dataset)
        all_results[enc_name] = metrics
        logger.info("  %s results: %s", enc_name, metrics)

    if not all_results:
        logger.error("No encoder produced results. Exiting.")
        sys.exit(1)

    _print_comparison_table(all_results, args.k_values)

    csv_path = os.path.join(args.output_dir, "encoder_benchmark.csv")
    _save_csv(all_results, csv_path)

    _save_bar_chart(all_results, args.k_values, args.output_dir)


if __name__ == "__main__":
    main()
