"""Small integration helper for validation scripts.

Use this helper once per keypoint immediately after coarse decoding and local
refinement.  It prevents accidental reuse of a description or crop record
between symmetric keypoints and emits the raw fields required by the
reliability analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

try:
    from utils.refinement_policy import Coordinate, RefinementPolicy
except ModuleNotFoundError:  # Direct execution from the utils directory.
    from refinement_policy import Coordinate, RefinementPolicy


def apply_and_record(
    *,
    image_id: Any,
    person_id: Any,
    keypoint: str,
    description: str,
    crop_size: float,
    gt: Sequence[float],
    coarse: Sequence[float],
    refined: Sequence[float],
    sequence_confidence: float,
    policy: RefinementPolicy,
    heatmap_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    coarse_xy: Coordinate = (float(coarse[0]), float(coarse[1]))
    refined_xy: Coordinate = (float(refined[0]), float(refined[1]))
    final, accepted = policy.apply(
        coarse_xy,
        refined_xy,
        float(sequence_confidence),
        heatmap_confidence=heatmap_confidence,
    )
    return {
        "image_id": image_id,
        "person_id": person_id,
        "keypoint": keypoint,
        "description": description,
        "crop_size": float(crop_size),
        "gt": [float(gt[0]), float(gt[1])],
        "coarse": list(coarse_xy),
        "refined": list(refined_xy),
        "final": list(final),
        "sequence_confidence": float(sequence_confidence),
        "heatmap_confidence": heatmap_confidence,
        "accepted": accepted,
    }


def detailed_prediction_records(
    people: Iterable[Mapping[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Expand detailed person predictions to one raw row per visible keypoint.

    Validation stores the always-on refinement candidate separately from the
    policy-selected final coordinate.  Reliability analysis therefore remains
    possible even when a later policy falls back to the coarse prediction.
    """

    for person in people:
        names = list(person["keypoint_names"])
        descriptions = list(person["descriptions"])
        ground_truth = list(person["gt_keypoints_224"])
        coarse = list(person["coarse_keypoints_224"])
        refined = list(person["refined_keypoints_224"])
        final = list(person["final_keypoints_224"])
        confidence = list(person["coarse_confidence"])
        crop_sizes = list(person["refinement_crop_sizes"])
        expected = len(names)
        lengths = {
            "descriptions": len(descriptions),
            "gt": len(ground_truth) // 3,
            "coarse": len(coarse) // 2,
            "refined": len(refined) // 2,
            "final": len(final) // 2,
            "confidence": len(confidence),
            "crop_sizes": len(crop_sizes),
        }
        if any(length != expected for length in lengths.values()):
            raise ValueError(
                f"Inconsistent detailed prediction lengths for person "
                f"{person.get('annotation_id', person.get('ins_id'))}: {lengths}"
            )
        for index, keypoint in enumerate(names):
            gt = ground_truth[index * 3:index * 3 + 3]
            crop_size = float(crop_sizes[index])
            if float(gt[2]) <= 0.0 or crop_size <= 0.0:
                continue
            yield {
                "image_id": person.get("image_id"),
                "person_id": person.get("annotation_id", person.get("ins_id")),
                "keypoint": keypoint,
                "description": descriptions[index],
                "crop_size": crop_size,
                "gt": [float(gt[0]), float(gt[1])],
                "coarse": [
                    float(coarse[index * 2]),
                    float(coarse[index * 2 + 1]),
                ],
                "refined": [
                    float(refined[index * 2]),
                    float(refined[index * 2 + 1]),
                ],
                "final": [
                    float(final[index * 2]),
                    float(final[index * 2 + 1]),
                ],
                "sequence_confidence": float(confidence[index]),
            }


def write_raw_refinement_jsonl(
    people: Iterable[Mapping[str, Any]],
    destination: str | Path,
) -> int:
    """Write reliability-analysis JSONL and return the emitted row count."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in detailed_prediction_records(people):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
