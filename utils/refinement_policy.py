"""Inference-time policy for risk-controlled local refinement.

The policy is deliberately independent of the model implementation.  Validation
code supplies one coarse coordinate, one refined coordinate, and the geometric
mean probability of the generated coarse-coordinate tokens.  A rejected update
falls back to the coarse coordinate and is recorded as an unchanged prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import math


Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class RefinementPolicy:
    """Configuration shared by COCO, MPII, and reliability analysis."""

    confidence_threshold: float = 0.5
    # Optional conservative checks.  They are disabled by default to preserve
    # the fixed 0.5 policy used for the reported experiment.
    heatmap_threshold: Optional[float] = None
    max_displacement: Optional[float] = None

    def accepts(
        self,
        sequence_confidence: float,
        *,
        heatmap_confidence: Optional[float] = None,
        coarse: Optional[Coordinate] = None,
        refined: Optional[Coordinate] = None,
    ) -> bool:
        """Return whether the local update is allowed for one keypoint."""

        if not math.isfinite(sequence_confidence):
            return False
        if sequence_confidence < self.confidence_threshold:
            return False
        if self.heatmap_threshold is not None:
            if heatmap_confidence is None or heatmap_confidence < self.heatmap_threshold:
                return False
        if self.max_displacement is not None:
            if coarse is None or refined is None:
                return False
            dx = float(refined[0]) - float(coarse[0])
            dy = float(refined[1]) - float(coarse[1])
            if math.hypot(dx, dy) > self.max_displacement:
                return False
        return True

    def apply(
        self,
        coarse: Coordinate,
        refined: Coordinate,
        sequence_confidence: float,
        *,
        heatmap_confidence: Optional[float] = None,
    ) -> Tuple[Coordinate, bool]:
        """Apply the gate and return ``(final_coordinate, was_refined)``."""

        use_refinement = self.accepts(
            sequence_confidence,
            heatmap_confidence=heatmap_confidence,
            coarse=coarse,
            refined=refined,
        )
        return (tuple(refined) if use_refinement else tuple(coarse), use_refinement)


def geometric_mean_probability(probabilities: Iterable[float]) -> float:
    """Compute a numerically stable geometric mean in [0, 1]."""

    values = [float(p) for p in probabilities]
    if not values or any((not math.isfinite(p) or p <= 0.0 or p > 1.0) for p in values):
        return 0.0
    return math.exp(sum(math.log(p) for p in values) / len(values))


def classify_outcome(
    coarse_error: float,
    final_error: float,
    tolerance: float = 0.5,
) -> str:
    """Classify a final prediction relative to its coarse prediction."""

    delta = float(final_error) - float(coarse_error)
    if delta < -tolerance:
        return "Improved"
    if delta > tolerance:
        return "Worsened"
    return "Unchanged"


def validate_refinement_policy(
    use_local_refiner: bool,
    use_confidence_gate: bool,
    confidence_threshold: float,
) -> None:
    """Validate the shared validation-time refinement configuration."""

    threshold = float(confidence_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "refinement confidence threshold must be finite and in [0, 1]"
        )
    if use_confidence_gate and not use_local_refiner:
        raise ValueError(
            "the confidence gate requires the local refiner to be enabled"
        )


def select_refinement_indices(
    sequence_confidence: Sequence[float],
    *,
    use_confidence_gate: bool,
    confidence_threshold: float,
    candidate_indices: Optional[Sequence[int]] = None,
):
    """Return the candidate indices accepted by the configured policy."""

    import numpy as np

    confidence = np.asarray(sequence_confidence, dtype=np.float64).reshape(-1)
    if candidate_indices is None:
        candidates = np.arange(confidence.size, dtype=np.int64)
    else:
        candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if candidates.size and (
        int(candidates.min()) < 0 or int(candidates.max()) >= confidence.size
    ):
        raise IndexError("refinement candidate index is out of range")
    if not use_confidence_gate:
        return candidates.copy()

    validate_refinement_policy(True, True, confidence_threshold)
    selected_confidence = confidence[candidates]
    accepted = np.isfinite(selected_confidence) & (
        selected_confidence >= float(confidence_threshold)
    )
    return candidates[accepted]


def merge_refined_coordinates(
    coarse_coordinates,
    refined_coordinates,
    refinement_indices: Sequence[int],
):
    """Merge selected local-refiner outputs into a copy of coarse coordinates."""

    import numpy as np

    coarse = np.asarray(coarse_coordinates)
    refined = np.asarray(refined_coordinates)
    indices = np.asarray(refinement_indices, dtype=np.int64).reshape(-1)
    if coarse.ndim != 2 or coarse.shape[1] != 2:
        raise ValueError("coarse coordinates must have shape [N, 2]")
    if refined.shape != (indices.size, 2):
        raise ValueError(
            "refined coordinates must have shape [len(refinement_indices), 2]"
        )
    if indices.size and (
        int(indices.min()) < 0 or int(indices.max()) >= coarse.shape[0]
    ):
        raise IndexError("refinement index is out of range")
    merged = coarse.copy()
    if indices.size:
        merged[indices] = refined.astype(merged.dtype, copy=False)
    return merged
