"""
Post-hoc Temperature Scaling for calibrated confidence in visual grounding.

Motivation
──────────
Detection confidence scores from Grounding DINO and Florence-2 are not
calibrated — a score of 0.9 does not mean the model is 90% likely to be
correct.  Temperature scaling (Guo et al., 2017) is a simple, single-parameter
post-hoc method that maps logits through a learned temperature τ:

    p̃ = softmax(z / τ)

The temperature τ is found by minimising the Negative Log-Likelihood (NLL)
on a held-out calibration set.  Crucially, the argmax (and therefore AP) is
unchanged — only confidence magnitudes are re-scaled.

Expected Calibration Error (ECE)
─────────────────────────────────
We measure calibration quality using ECE (Naeini et al., 2015):

    ECE = Σ_{b=1}^{B} |B_b| / n · |acc(B_b) - conf(B_b)|

where B_b are M confidence bins, acc(B_b) is the fraction of detections in
bin b that are correct (IoU ≥ threshold), and conf(B_b) is their mean
confidence.  A perfectly calibrated model has ECE = 0.

Vector Temperature Scaling
──────────────────────────
Scalar temperature scaling applies a single τ globally.  We also implement
*vector* temperature scaling (Kull et al., 2019) where τ is a per-class
vector, allowing different classes to be scaled independently:

    p̃_c = σ(z_c / τ_c)   (element-wise per-class scaling)

This is more flexible but uses C parameters (one per class) instead of 1.

References
──────────
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
  https://arxiv.org/abs/1706.04599
- Naeini et al. (2015): "Obtaining Well Calibrated Probabilities"
- Kull et al. (2019): "Beyond Temperature Scaling"
  https://arxiv.org/abs/1906.02629
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: torch.Tensor,   # (N,) predicted confidence scores
    is_correct: torch.Tensor,    # (N,) 1.0 = correct detection, 0.0 = incorrect
    num_bins: int = 15,
) -> Tuple[float, Dict]:
    """
    Compute ECE and per-bin calibration statistics.

    Args:
        confidences: Model confidence scores, in [0, 1].
        is_correct: Binary correctness indicators (e.g., IoU ≥ 0.5).
        num_bins: Number of equal-width confidence bins.

    Returns:
        ece: Scalar ECE value.
        bin_stats: Dict with per-bin ``accuracy``, ``confidence``, and ``count``.
    """
    confidences = confidences.float().cpu()
    is_correct = is_correct.float().cpu()
    N = confidences.shape[0]

    bins = torch.linspace(0.0, 1.0, num_bins + 1)
    bin_stats: Dict[str, List] = {"accuracy": [], "confidence": [], "count": []}
    ece = 0.0

    for i in range(num_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        in_bin = (confidences > lo) & (confidences <= hi)
        count = in_bin.sum().item()

        if count > 0:
            acc = is_correct[in_bin].mean().item()
            conf = confidences[in_bin].mean().item()
            ece += (count / N) * abs(acc - conf)
        else:
            acc, conf = 0.0, 0.0

        bin_stats["accuracy"].append(acc)
        bin_stats["confidence"].append(conf)
        bin_stats["count"].append(count)

    return ece, bin_stats


# ---------------------------------------------------------------------------
# Scalar Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Post-hoc scalar temperature scaling calibrator.

    The temperature τ is trained by minimising the binary cross-entropy
    (NLL) of scaled confidence scores against detection correctness labels
    on a calibration set.

    The model parameters (detector weights) are frozen — only τ is optimised.

    Args:
        init_temperature: Starting value for τ.  1.0 = identity (no scaling).
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        # Parametrised as log(τ) for unconstrained optimisation
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp().item()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by 1/τ.

        Args:
            logits: Raw detection logits of any shape.

        Returns:
            Scaled logits (same shape).
        """
        return logits / self.log_temperature.exp()

    def calibrate(
        self,
        logits: torch.Tensor,          # (N,) raw detection logits
        is_correct: torch.Tensor,      # (N,) binary correctness labels
        lr: float = 0.01,
        max_iters: int = 1000,
        tol: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Find optimal τ via L-BFGS on the NLL loss.

        L-BFGS is preferred over SGD because the 1-D loss surface is convex
        and L-BFGS finds the optimum in few evaluations.

        Args:
            logits: Pre-calibration detection confidence logits.
            is_correct: Ground-truth correctness for each detection.
            lr: L-BFGS step size.
            max_iters: Maximum optimiser iterations.
            tol: Convergence tolerance on |Δloss|.

        Returns:
            Dict with ``temperature``, ``nll_before``, ``nll_after``,
            ``ece_before``, ``ece_after``.
        """
        logits = logits.detach().float()
        labels = is_correct.detach().float()

        # Pre-calibration metrics
        probs_before = torch.sigmoid(logits)
        nll_before = nn.functional.binary_cross_entropy(
            probs_before.clamp(1e-7, 1 - 1e-7), labels
        ).item()
        ece_before, _ = expected_calibration_error(probs_before, labels)
        logger.info("Pre-calibration  NLL=%.4f  ECE=%.4f  τ=1.000", nll_before, ece_before)

        optimizer = optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iters)

        prev_loss = float("inf")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits / self.log_temperature.exp()
            loss = nn.functional.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            return loss

        for _ in range(max_iters // 10):
            loss = optimizer.step(closure)
            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

        # Post-calibration metrics
        with torch.no_grad():
            probs_after = torch.sigmoid(logits / self.log_temperature.exp())
        nll_after = nn.functional.binary_cross_entropy(
            probs_after.clamp(1e-7, 1 - 1e-7), labels
        ).item()
        ece_after, _ = expected_calibration_error(probs_after, labels)
        logger.info(
            "Post-calibration NLL=%.4f  ECE=%.4f  τ=%.4f",
            nll_after, ece_after, self.temperature,
        )

        return {
            "temperature": self.temperature,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "nll_reduction_pct": 100.0 * (nll_before - nll_after) / (nll_before + 1e-9),
            "ece_reduction_pct": 100.0 * (ece_before - ece_after) / (ece_before + 1e-9),
        }

    def scale_detection(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to a NumPy array of detection confidence scores.

        Scores are treated as probabilities: we apply logit → scale → sigmoid.

        Args:
            scores: Array of confidence values in (0, 1).

        Returns:
            Calibrated probabilities, same shape.
        """
        τ = self.temperature
        # Avoid log(0) or log(1): clip to (ε, 1-ε)
        eps = 1e-7
        scores_clipped = np.clip(scores, eps, 1.0 - eps)
        logits = np.log(scores_clipped / (1.0 - scores_clipped))  # logit transform
        scaled_logits = logits / τ
        return 1.0 / (1.0 + np.exp(-scaled_logits))               # sigmoid


# ---------------------------------------------------------------------------
# Vector Temperature Scaling (per-class)
# ---------------------------------------------------------------------------

class VectorTemperatureScaler(nn.Module):
    """
    Per-class temperature scaling calibrator.

    Each class c gets its own temperature τ_c, allowing heterogeneous
    calibration across different object categories (e.g., rare classes
    often have higher uncertainty than common classes).

    Args:
        num_classes: Number of categories C.
        init_temperature: Initial τ for all classes (scalar).
    """

    def __init__(self, num_classes: int, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        log_temp = np.log(init_temperature)
        self.log_temperatures = nn.Parameter(
            torch.full((num_classes,), log_temp, dtype=torch.float32)
        )

    @property
    def temperatures(self) -> torch.Tensor:
        """Per-class temperatures, shape (C,)."""
        return self.log_temperatures.exp()

    def forward(self, logits: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Scale logits using the temperature of each logit's class.

        Args:
            logits: ``(N,)`` detection logits.
            class_ids: ``(N,)`` integer class IDs for each detection.

        Returns:
            Scaled logits ``(N,)``.
        """
        τ = self.log_temperatures.exp()[class_ids]   # (N,)
        return logits / τ

    def calibrate_per_class(
        self,
        logits: torch.Tensor,       # (N,)
        labels: torch.Tensor,       # (N,) binary correctness
        class_ids: torch.Tensor,    # (N,) class index for each detection
        lr: float = 0.05,
        max_iters: int = 500,
    ) -> Dict[str, float]:
        """
        Optimise per-class temperatures simultaneously using Adam.

        Returns:
            Dict with mean NLL/ECE before and after, plus per-class τ values.
        """
        logits = logits.detach().float()
        labels = labels.detach().float()

        optimizer = optim.Adam([self.log_temperatures], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

        with torch.no_grad():
            nll_before = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()

        for _ in range(max_iters):
            optimizer.zero_grad()
            scaled = self.forward(logits, class_ids)
            loss = nn.functional.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            # Clip log_temperatures to keep τ in [0.01, 10]
            with torch.no_grad():
                self.log_temperatures.clamp_(np.log(0.01), np.log(10.0))
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            nll_after = nn.functional.binary_cross_entropy_with_logits(
                self.forward(logits, class_ids), labels
            ).item()

        return {
            "nll_before": nll_before,
            "nll_after": nll_after,
            "mean_temperature": self.temperatures.mean().item(),
            "temperatures": self.temperatures.detach().cpu().numpy().tolist(),
        }
