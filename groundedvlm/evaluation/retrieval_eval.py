"""
Retrieval evaluator for cross-modal image-text retrieval.

Computes Recall@k and MRR for a given dataset of (image, query, relevant_indices)
triples using any encoder with encode_image / encode_text methods.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from groundedvlm.utils.metrics import compute_recall_at_k, compute_mrr

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate image-text retrieval with Recall@k and MRR.

    Args:
        encoder: Any encoder exposing encode_image(images) and encode_text(texts).
        k_values: List of k values for Recall@k computation.
    """

    def __init__(self, encoder: Any, k_values: List[int] = None) -> None:
        self.encoder = encoder
        self.k_values: List[int] = k_values if k_values is not None else [1, 5, 10]
        self._gallery_embeds: torch.Tensor = None
        logger.debug(
            "RetrievalEvaluator initialised (k_values=%s).", self.k_values
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, dataset: Any) -> Dict[str, float]:
        """Evaluate retrieval performance over the full dataset.

        The dataset must expose:
            - ``dataset.images``: list of PIL Images (the gallery).
            - ``dataset.queries``: list of query strings.
            - ``dataset.relevant_indices``: list of lists; relevant_indices[i]
              contains the gallery indices relevant to query i.

        Args:
            dataset: Dataset object with the attributes described above.

        Returns:
            Dict with Recall@k for each k and MRR.
        """
        images = dataset.images
        queries = dataset.queries
        relevant_indices_list = dataset.relevant_indices

        logger.info(
            "Building gallery embeddings for %d images.", len(images)
        )
        self._build_gallery(images)

        logger.info("Evaluating %d queries.", len(queries))

        all_retrieved: List[List[int]] = []

        for query_text in tqdm(queries, desc="Retrieval"):
            text_embeds = self.encoder.encode_text([query_text])  # (1, D)
            top_k = max(self.k_values)
            retrieved = self._retrieve(text_embeds[0], self._gallery_embeds, k=top_k)
            all_retrieved.append(retrieved)

        metrics: Dict[str, float] = {}
        for k in self.k_values:
            recall_scores = [
                compute_recall_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(all_retrieved, relevant_indices_list)
            ]
            metrics[f"Recall@{k}"] = float(np.mean(recall_scores))

        mrr = compute_mrr(all_retrieved, relevant_indices_list)
        metrics["MRR"] = mrr

        logger.info("Retrieval evaluation complete: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_gallery(self, images: List[Any]) -> None:
        """Pre-compute and store gallery image embeddings.

        Args:
            images: List of PIL Images.
        """
        batch_size = 64
        all_embeds: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            embeds = self.encoder.encode_image(batch)
            all_embeds.append(embeds)

        self._gallery_embeds = torch.cat(all_embeds, dim=0)  # (G, D)
        logger.debug("Gallery embeddings built: shape %s.", self._gallery_embeds.shape)

    def _retrieve(
        self,
        query_embed: torch.Tensor,
        gallery_embeds: torch.Tensor,
        k: int,
    ) -> List[int]:
        """Retrieve top-k gallery indices for a single query embedding.

        Args:
            query_embed: Tensor of shape (D,).
            gallery_embeds: Tensor of shape (G, D).
            k: Number of top results to return.

        Returns:
            List of gallery indices sorted by descending similarity.
        """
        similarities = gallery_embeds @ query_embed  # (G,)
        k_actual = min(k, gallery_embeds.shape[0])
        top_k_indices = torch.topk(similarities, k=k_actual, largest=True).indices
        return top_k_indices.cpu().tolist()

    def _compute_recall_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int,
    ) -> float:
        """Recall@k for a single query.

        Args:
            retrieved: Ordered list of retrieved gallery indices.
            relevant: List of ground-truth relevant indices.
            k: Cutoff rank.

        Returns:
            Recall@k as a float in [0, 1].
        """
        return compute_recall_at_k(retrieved, relevant, k)

    def _compute_mrr(self, retrieved_ranks: List[List[int]]) -> float:
        """Mean Reciprocal Rank over a list of queries.

        Args:
            retrieved_ranks: List where each element is an ordered list of
                             retrieved indices for one query.

        Returns:
            MRR as a float.
        """
        # Use shared relevant_indices from the last evaluate() call.
        # This method is kept for backward-compatibility; compute_mrr in
        # metrics.py is the canonical implementation.
        return float(np.mean([1.0 / (r[0] + 1) if r else 0.0 for r in retrieved_ranks]))
