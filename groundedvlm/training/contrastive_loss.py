"""
InfoNCE Contrastive Loss with Hard Negative Mining and Learnable Temperature.

Theory
──────
The InfoNCE objective (van den Oord et al., 2018) for a batch of N
image-text pairs is:

    L = -1/N · Σᵢ [ log( exp(sᵢᵢ/τ) / Σⱼ exp(sᵢⱼ/τ) )
                  + log( exp(sᵢᵢ/τ) / Σⱼ exp(sⱼᵢ/τ) ) ]

where sᵢⱼ = <v_i, t_j> is the cosine similarity between image embedding v_i
and text embedding t_j, and τ is the temperature.

This is a symmetric cross-entropy over the similarity matrix S ∈ ℝ^{N×N}
with the diagonal as positive pairs and all off-diagonal entries as negatives.

Hard Negative Mining
────────────────────
With random in-batch negatives, easy negatives dominate the gradient.  We
implement two complementary hard-negative strategies:

1. **In-batch hard negatives**: For each anchor, we select the top-k
   hardest negatives (highest similarity with wrong labels) from the current
   batch, masking the others from the softmax denominator.

2. **False-negative masking**: In large-scale retrieval datasets, an
   "off-diagonal" pair may actually be semantically related (e.g., two images
   of the same object with different captions).  A similarity threshold τ_fn
   is used to identify and mask such pairs from the loss.

Learnable Temperature
─────────────────────
Following CLIP (Radford et al., 2021), τ is a learnable scalar initialised
to log(1/0.07) ≈ 2.65 and clipped to [0, log(100)] = [0, 4.61] to prevent
collapse.  The gradient of L w.r.t. τ has the form:

    ∂L/∂τ = 1/τ² · Σᵢ (pos_score_i - mean_neg_score_i)

which drives τ → 0 when positives are more similar than negatives
(sharper distribution) and τ → ∞ when negatives dominate.

Cross-Device Gather (Distributed Training)
──────────────────────────────────────────
In distributed settings each GPU only sees a batch slice of size N/K where
K is the world size.  We gather embeddings across all GPUs to construct a
larger (N,N) similarity matrix, dramatically increasing the number of
in-batch negatives.  This is implemented via ``torch.distributed.all_gather``
with a custom backward that only backpropagates gradients to the local shard.

References
──────────
- van den Oord et al. (2018): "Representation Learning with CPC"
- Radford et al. (2021): "Learning Transferable Visual Models" (CLIP)
- Robinson et al. (2021): "Contrastive Learning with Hard Negative Samples"
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed gather utility
# ---------------------------------------------------------------------------

class _AllGatherFunction(torch.autograd.Function):
    """
    All-gather with gradient flowing only to the local rank's shard.

    In the forward pass: gather embeddings from all ranks.
    In the backward pass: scatter the local shard's gradient back (no-op for
    remote shards because we don't own their parameters).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Only return the gradient slice corresponding to this rank
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        return grad_output[start:end].contiguous(), None, None


def gather_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Gather embeddings across all DDP ranks (no-op if not distributed)."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return embeddings
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return _AllGatherFunction.apply(embeddings, world_size, rank)


# ---------------------------------------------------------------------------
# InfoNCE Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) contrastive loss with optional hard
    negative mining and learnable temperature.

    Args:
        init_temperature: Initial temperature τ₀.  Stored as log(τ) for
            numerical stability (avoids exp overflow in gradients).
        learnable_temp: Whether τ is a learned parameter.
        temp_min: Minimum τ (corresponding to maximum sharpness).
        temp_max: Maximum τ (minimum sharpness, avoiding uniform distributions).
        hard_neg_k: If > 0, select the top-k hardest negatives per anchor and
            zero out all other negatives in the denominator.
        false_neg_threshold: Similarity threshold above which an off-diagonal
            pair is considered a false negative and masked from the loss.
            Set to None to disable false-negative masking.
        label_smoothing: Smoothing applied to the positive label distribution.
            0.0 = standard cross-entropy; 0.1 = 10% label smoothing.
        gather_distributed: If True, gather embeddings across DDP ranks.
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        learnable_temp: bool = True,
        temp_min: float = 0.01,
        temp_max: float = 1.0,
        hard_neg_k: int = 0,
        false_neg_threshold: Optional[float] = None,
        label_smoothing: float = 0.0,
        gather_distributed: bool = False,
    ) -> None:
        super().__init__()
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.hard_neg_k = hard_neg_k
        self.false_neg_threshold = false_neg_threshold
        self.label_smoothing = label_smoothing
        self.gather_distributed = gather_distributed

        # Store log(τ) so that τ = exp(log_τ) is always positive.
        log_temp = math.log(init_temperature)
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(log_temp))
        else:
            self.register_buffer("log_temperature", torch.tensor(log_temp))

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature τ, clamped to [temp_min, temp_max]."""
        return self.log_temperature.exp().clamp(self.temp_min, self.temp_max)

    def forward(
        self,
        image_embeds: torch.Tensor,   # (N, D) L2-normalised
        text_embeds: torch.Tensor,    # (N, D) L2-normalised
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute symmetric InfoNCE loss.

        Args:
            image_embeds: ``(N, D)`` L2-normalised image embeddings.
            text_embeds: ``(N, D)`` L2-normalised text embeddings.

        Returns:
            loss: Scalar loss tensor.
            metrics: Dict with ``loss_i2t``, ``loss_t2i``, ``temperature``,
                     ``accuracy_i2t``, ``accuracy_t2i``.
        """
        assert image_embeds.shape == text_embeds.shape, (
            f"Shape mismatch: {image_embeds.shape} vs {text_embeds.shape}"
        )
        N, D = image_embeds.shape

        # Optionally gather from all DDP ranks
        if self.gather_distributed:
            image_embeds = gather_embeddings(image_embeds)
            text_embeds = gather_embeddings(text_embeds)

        N_total = image_embeds.shape[0]

        # ── Similarity matrix S ∈ ℝ^{N×N} ──────────────────────────────
        # Both inputs are L2-normalised; matrix multiply gives cosine sims.
        τ = self.temperature
        S = (image_embeds @ text_embeds.T) / τ   # (N_total, N_total)

        # ── False-negative masking ───────────────────────────────────────
        # Identify off-diagonal pairs with high similarity — they are likely
        # true matches mislabelled as negatives — and exclude from softmax.
        if self.false_neg_threshold is not None:
            with torch.no_grad():
                # S_raw: (N, N) cosine similarities (without temperature)
                S_raw = image_embeds @ text_embeds.T          # in [-1, 1]
                eye = torch.eye(N_total, device=S_raw.device, dtype=torch.bool)
                false_neg_mask = (S_raw > self.false_neg_threshold) & ~eye
        else:
            false_neg_mask = None

        # ── Hard negative selection ──────────────────────────────────────
        if self.hard_neg_k > 0:
            S = self._apply_hard_neg_mask(S, N_total, false_neg_mask)
        elif false_neg_mask is not None:
            S = S.masked_fill(false_neg_mask, float("-inf"))

        # ── Symmetric cross-entropy loss ─────────────────────────────────
        labels = torch.arange(N_total, device=image_embeds.device)

        loss_i2t = F.cross_entropy(S, labels, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(S.T, labels, label_smoothing=self.label_smoothing)
        loss = (loss_i2t + loss_t2i) / 2.0

        # ── Accuracy (top-1 retrieval) ───────────────────────────────────
        with torch.no_grad():
            acc_i2t = (S.argmax(dim=1) == labels).float().mean().item()
            acc_t2i = (S.T.argmax(dim=1) == labels).float().mean().item()

        metrics = {
            "loss_i2t":    loss_i2t.item(),
            "loss_t2i":    loss_t2i.item(),
            "loss":        loss.item(),
            "temperature": τ.item(),
            "accuracy_i2t": acc_i2t,
            "accuracy_t2i": acc_t2i,
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Hard negative masking
    # ------------------------------------------------------------------

    def _apply_hard_neg_mask(
        self,
        S: torch.Tensor,           # (N, N) scaled similarities
        N: int,
        false_neg_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        For each anchor i, keep only the top-k hardest negatives; mask rest.

        "Hardest negative" = highest similarity score with a wrong label.

        Strategy:
          1. Detach S to avoid double-differentiating through the masking.
          2. Set the diagonal (positives) to -inf in the detached copy.
          3. Select top-k columns per row by score.
          4. Create a mask retaining those k positions + the diagonal.
          5. Apply the mask to the original (differentiable) S.
        """
        eye = torch.eye(N, device=S.device, dtype=torch.bool)

        with torch.no_grad():
            S_det = S.detach()
            if false_neg_mask is not None:
                S_det = S_det.masked_fill(false_neg_mask, float("-inf"))
            # Mask the diagonal so we rank only negatives
            S_neg = S_det.masked_fill(eye, float("-inf"))
            k = min(self.hard_neg_k, N - 1)
            # Top-k per row: (N, k) indices
            _, topk_idx = S_neg.topk(k, dim=1)

        # Build binary mask: keep diagonal + top-k negatives
        keep_mask = eye.clone()
        keep_mask.scatter_(1, topk_idx, True)
        if false_neg_mask is not None:
            keep_mask = keep_mask & ~false_neg_mask

        # Apply: set non-kept positions to -inf → excluded from softmax
        S_masked = S.masked_fill(~keep_mask, float("-inf"))
        return S_masked


# ---------------------------------------------------------------------------
# Margin-based Triplet Loss (alternative objective)
# ---------------------------------------------------------------------------

class TripletContrastiveLoss(nn.Module):
    """
    Batch-hard triplet loss for visual-language matching.

    For each anchor (image i), the hardest positive is the text j (j≠i) with
    the highest cosine similarity to i, and the hardest negative is the text k
    with the lowest similarity (but still above the margin threshold).

    L = max(0, d_ap - d_an + margin)

    where d_ap = 1 - s(i, j), d_an = 1 - s(i, k).

    Unlike InfoNCE, the triplet loss is not a lower bound on mutual information
    but tends to produce more uniform embedding spaces empirically.

    Args:
        margin: Triplet margin Δ.
        mining: One of ``"hard"`` (batch-hard), ``"semi-hard"``, or ``"all"``.
    """

    def __init__(self, margin: float = 0.2, mining: str = "hard") -> None:
        super().__init__()
        assert mining in ("hard", "semi-hard", "all")
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        N = image_embeds.shape[0]
        S = image_embeds @ text_embeds.T          # (N, N) cosine similarities
        D = 1.0 - S                               # cosine distances

        # Positive distances: d(i, i) for each i
        d_pos = D.diag().unsqueeze(1)             # (N, 1)

        if self.mining == "hard":
            # Hardest positive: max distance to the correct text (d_pos by definition)
            # Hardest negative: min distance to any incorrect text
            inf_mask = torch.eye(N, device=D.device).bool()
            D_neg = D.masked_fill(inf_mask, float("inf"))
            d_neg, _ = D_neg.min(dim=1, keepdim=True)  # (N, 1)

        elif self.mining == "semi-hard":
            # Semi-hard: negatives that are farther than the positive but within margin
            inf_mask = torch.eye(N, device=D.device).bool()
            D_neg = D.masked_fill(inf_mask, float("inf"))
            # d_an: negatives with d_pos < d_an < d_pos + margin
            valid = (D_neg > d_pos) & (D_neg < d_pos + self.margin)
            D_semi = D_neg.masked_fill(~valid, float("inf"))
            d_neg, _ = D_semi.min(dim=1, keepdim=True)
            # Fall back to hardest negative if no semi-hard found
            d_neg_fallback, _ = D_neg.min(dim=1, keepdim=True)
            no_semi = d_neg.isinf()
            d_neg = torch.where(no_semi, d_neg_fallback, d_neg)

        else:  # "all" — average over all valid triplets
            inf_mask = torch.eye(N, device=D.device).bool()
            D_neg = D.masked_fill(inf_mask, float("-inf"))
            losses_all = F.relu(d_pos - D_neg + self.margin)
            # Exclude positives on diagonal
            losses_all = losses_all.masked_fill(inf_mask, 0.0)
            loss = losses_all.sum() / (N * (N - 1))
            return loss, {"loss": loss.item()}

        loss = F.relu(d_pos - d_neg + self.margin).mean()
        return loss, {"loss": loss.item(), "d_pos": d_pos.mean().item(), "d_neg": d_neg.mean().item()}
