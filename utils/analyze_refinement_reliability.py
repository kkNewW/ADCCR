"""Reproduce the always-on and confidence-gated refinement analysis.

Input is JSONL with one record per annotated keypoint.  Required fields:

    {"dataset": "COCO", "image_id": ..., "person_id": ..., "keypoint": "left_wrist",
     "crop_size": 96, "gt": [x, y], "coarse": [x, y],
     "refined": [x, y], "sequence_confidence": 0.73}

Coordinates are in the resized 224x224 person-instance system.  The utility
also accepts ``heatmap_confidence`` when an additional check is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from refinement_policy import RefinementPolicy, classify_outcome


Coordinate = Tuple[float, float]
BINS: Sequence[Tuple[str, float, float]] = (
    ("0-0.10", 0.0, 0.10),
    ("0.10-0.25", 0.10, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    (">0.50", 0.50, float("inf")),
)


def _point(value: Sequence[float]) -> Coordinate:
    if len(value) != 2:
        raise ValueError(f"coordinate must have two values, got {value!r}")
    return float(value[0]), float(value[1])


def _error(a: Coordinate, b: Coordinate) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _infinity_error(a: Coordinate, b: Coordinate) -> float:
    """Return the L-infinity error used only for Table 16 stratification."""

    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _bin(value: float) -> str:
    for index, (label, lo, hi) in enumerate(BINS):
        # Match the manuscript intervals exactly: 0 <= q <= 0.10,
        # followed by left-open/right-closed ranges.
        if (index == 0 and lo <= value <= hi) or (
            index > 0 and lo < value <= hi
        ):
            return label
    return BINS[-1][0]


def _read_jsonl(path: Path) -> List[MutableMapping[str, object]]:
    rows: List[MutableMapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            for key in ("gt", "coarse", "refined", "crop_size", "sequence_confidence"):
                if key not in record:
                    raise ValueError(f"line {line_number} is missing {key!r}")
            rows.append(record)
    return rows


def _summarise(rows: Iterable[Mapping[str, object]], tolerance: float) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("policy", "always_on")), str(row["error_bin"]))].append(row)

    output: List[dict] = []
    for policy in ("always_on", "gated"):
        for label, _, _ in BINS:
            subset = grouped.get((policy, label), [])
            n = len(subset)
            if not n:
                output.append({"policy": policy, "error_range": label, "n": 0})
                continue
            coarse = [float(r["coarse_error"]) for r in subset]
            final = [float(r["final_error"]) for r in subset]
            outcomes = Counter(str(r["outcome"]) for r in subset)
            mean_coarse = sum(coarse) / n
            mean_final = sum(final) / n
            reduction = 100.0 * (mean_coarse - mean_final) / mean_coarse if mean_coarse else 0.0
            output.append(
                {
                    "policy": policy,
                    "error_range": label,
                    "n": n,
                    "coverage": 100.0 * sum(bool(r["accepted"]) for r in subset) / n,
                    "coarse_error_px": mean_coarse,
                    "final_error_px": mean_final,
                    "reduction_percent": reduction,
                    "improved_percent": 100.0 * outcomes["Improved"] / n,
                    "unchanged_percent": 100.0 * outcomes["Unchanged"] / n,
                    "worsened_percent": 100.0 * outcomes["Worsened"] / n,
                    "tolerance_px": tolerance,
                }
            )
    return output


def analyse(
    rows: Iterable[MutableMapping[str, object]],
    policy: RefinementPolicy,
    tolerance: float = 0.5,
) -> Tuple[List[MutableMapping[str, object]], List[dict]]:
    enriched: List[MutableMapping[str, object]] = []
    for row in rows:
        gt = _point(row["gt"])  # type: ignore[arg-type]
        coarse = _point(row["coarse"])  # type: ignore[arg-type]
        refined = _point(row["refined"])  # type: ignore[arg-type]
        crop_size = float(row["crop_size"])
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        coarse_error = _error(coarse, gt)
        refined_error = _error(refined, gt)
        normalized_coarse_error = _infinity_error(coarse, gt) / crop_size
        confidence = float(row["sequence_confidence"])
        heatmap_confidence = row.get("heatmap_confidence")
        final, accepted = policy.apply(
            coarse,
            refined,
            confidence,
            heatmap_confidence=None if heatmap_confidence is None else float(heatmap_confidence),
        )
        gated_error = _error(final, gt)
        base = dict(row)
        base.update(
            {
                "coarse_error": coarse_error,
                "always_on_error": refined_error,
                "gated_error": gated_error,
                "normalized_coarse_error": normalized_coarse_error,
                "error_bin": _bin(normalized_coarse_error),
            }
        )

        always_on = dict(base)
        always_on.update(
            {
                "policy": "always_on",
                "final": list(refined),
                "final_error": refined_error,
                "accepted": True,
                "outcome": classify_outcome(coarse_error, refined_error, tolerance),
            }
        )
        gated = dict(base)
        gated.update(
            {
                "policy": "gated",
                "final": list(final),
                "final_error": gated_error,
                "accepted": accepted,
                "outcome": classify_outcome(coarse_error, gated_error, tolerance),
            }
        )
        enriched.extend((always_on, gated))
    return enriched, _summarise(enriched, tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="raw prediction JSONL")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--heatmap-threshold", type=float, default=None)
    parser.add_argument("--max-displacement", type=float, default=None)
    args = parser.parse_args()

    policy = RefinementPolicy(
        confidence_threshold=args.threshold,
        heatmap_threshold=args.heatmap_threshold,
        max_displacement=args.max_displacement,
    )
    raw = _read_jsonl(args.input)
    rows, summary = analyse(raw, policy, tolerance=args.tolerance)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"policy": policy.__dict__, "summary": summary, "records": rows}, indent=2),
        encoding="utf-8",
    )
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary[0].keys()) if summary else ["policy", "error_range", "n"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"records": len(raw), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
