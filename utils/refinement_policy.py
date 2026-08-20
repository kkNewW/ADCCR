"""Inference-time policies for deciding when to run local refinement."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def validate_refinement_policy(
    use_local_refiner: bool,
    use_confidence_gate: bool,
    confidence_threshold: float,
) -> None:
    """Validate command-line options for confidence-gated refinement."""
    if use_confidence_gate and not use_local_refiner:
        raise ValueError(
            "Confidence gating requires --use-local-refiner."
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "refinement confidence threshold must be in [0, 1]."
        )


def select_refinement_indices(
    confidences: Iterable[float],
    *,
    use_confidence_gate: bool,
    confidence_threshold: float,
    candidate_indices: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return candidate indices whose confidence passes the policy."""
    values = np.asarray(confidences, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("confidences must be a one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("confidences must contain only finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("confidences must be probabilities in [0, 1].")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")

    if candidate_indices is None:
        candidates = np.arange(values.size, dtype=np.int64)
    else:
        candidates = np.asarray(
            tuple(candidate_indices),
            dtype=np.int64,
        )
        if candidates.ndim != 1:
            raise ValueError(
                "candidate_indices must be one-dimensional."
            )
        if np.any((candidates < 0) | (candidates >= values.size)):
            raise IndexError("candidate index is out of range.")
        if np.unique(candidates).size != candidates.size:
            raise ValueError("candidate_indices must be unique.")

    if not use_confidence_gate:
        return candidates
    return candidates[
        values[candidates] >= confidence_threshold
    ]


def merge_refined_coordinates(
    coarse_coordinates,
    refined_coordinates,
    refinement_indices,
) -> np.ndarray:
    """Merge refined coordinates with coarse-coordinate fallbacks."""
    coarse = np.asarray(coarse_coordinates, dtype=np.float32)
    refined = np.asarray(refined_coordinates, dtype=np.float32)
    indices = np.asarray(refinement_indices, dtype=np.int64)

    if coarse.ndim != 2 or coarse.shape[1] != 2:
        raise ValueError(
            "coarse_coordinates must have shape [N, 2]."
        )
    if refined.shape != (indices.size, 2):
        raise ValueError(
            "refined_coordinates must have shape "
            "[len(refinement_indices), 2]."
        )
    if np.any((indices < 0) | (indices >= coarse.shape[0])):
        raise IndexError("refinement index is out of range.")
    if np.unique(indices).size != indices.size:
        raise ValueError("refinement_indices must be unique.")

    merged = coarse.copy()
    merged[indices] = refined
    return merged
