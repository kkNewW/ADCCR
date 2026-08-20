"""Summarize coarse-to-refined keypoint error propagation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BIN_DEFINITIONS = (
    ("0-0.10", 0.0, 0.10, True),
    ("0.10-0.25", 0.10, 0.25, True),
    ("0.25-0.50", 0.25, 0.50, True),
    (">0.50", 0.50, np.inf, False),
)


def load_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return records


def index_records(records, label):
    indexed = {}
    for record in records:
        annotation_id = int(record["annotation_id"])
        if annotation_id in indexed:
            raise ValueError(
                f"Duplicate annotation_id {annotation_id} in {label}."
            )
        indexed[annotation_id] = record
    return indexed


def reshape_record(record):
    return {
        "coarse": np.asarray(
            record["coarse_keypoints_224"],
            dtype=np.float64,
        ).reshape(17, 2),
        "final": np.asarray(
            record["final_keypoints_224"],
            dtype=np.float64,
        ).reshape(17, 2),
        "ground_truth": np.asarray(
            record["gt_keypoints_224"],
            dtype=np.float64,
        ).reshape(17, 3),
        "crop_sizes": np.asarray(
            record["refinement_crop_sizes"],
            dtype=np.float64,
        ).reshape(17),
        "confidence": np.asarray(
            record["coarse_confidence"],
            dtype=np.float64,
        ).reshape(17),
        "applied": np.asarray(
            record["refinement_applied"],
            dtype=bool,
        ).reshape(17),
    }


def collect_keypoint_arrays(
    always_on_records,
    gated_records,
    confidence_threshold,
):
    always_index = index_records(always_on_records, "always-on run")
    gated_index = index_records(gated_records, "gated run")
    if always_index.keys() != gated_index.keys():
        missing_from_gated = sorted(
            always_index.keys() - gated_index.keys()
        )
        missing_from_always = sorted(
            gated_index.keys() - always_index.keys()
        )
        raise ValueError(
            "Runs contain different annotation IDs: "
            f"missing from gated={missing_from_gated[:5]}, "
            f"missing from always-on={missing_from_always[:5]}."
        )

    collected = {
        key: []
        for key in (
            "coarse",
            "ground_truth",
            "crop_sizes",
            "confidence",
            "always_final",
            "gated_final",
            "always_applied",
            "gated_applied",
        )
    }

    for annotation_id in sorted(always_index):
        always = reshape_record(always_index[annotation_id])
        gated = reshape_record(gated_index[annotation_id])

        for key in (
            "coarse",
            "ground_truth",
            "crop_sizes",
            "confidence",
        ):
            if not np.allclose(always[key], gated[key], atol=1e-6):
                raise ValueError(
                    f"{key} differs between runs for annotation "
                    f"{annotation_id}."
                )
        if not np.all(always["applied"]):
            raise ValueError(
                "The always-on run contains skipped keypoints for "
                f"annotation {annotation_id}."
            )

        expected_gated = (
            gated["confidence"] >= confidence_threshold
        )
        if not np.array_equal(gated["applied"], expected_gated):
            raise ValueError(
                "The gated run does not match confidence >= "
                f"{confidence_threshold} for annotation "
                f"{annotation_id}."
            )
        expected_final = np.where(
            gated["applied"][:, None],
            gated["final"],
            gated["coarse"],
        )
        if not np.allclose(
            expected_final,
            gated["final"],
            atol=1e-6,
        ):
            raise ValueError(
                "Skipped keypoints do not preserve their coarse "
                f"coordinates for annotation {annotation_id}."
            )

        collected["coarse"].append(always["coarse"])
        collected["ground_truth"].append(always["ground_truth"])
        collected["crop_sizes"].append(always["crop_sizes"])
        collected["confidence"].append(always["confidence"])
        collected["always_final"].append(always["final"])
        collected["gated_final"].append(gated["final"])
        collected["always_applied"].append(always["applied"])
        collected["gated_applied"].append(gated["applied"])

    return {
        key: np.concatenate(value, axis=0)
        for key, value in collected.items()
    }


def outcome_counts(coarse_error, final_error, tolerance_px):
    change = final_error - coarse_error
    improved = change < -tolerance_px
    unchanged = np.abs(change) <= tolerance_px
    worsened = change > tolerance_px
    if not np.all(improved | unchanged | worsened):
        raise RuntimeError("Outcome classes are not exhaustive.")
    return {
        "improved": int(improved.sum()),
        "unchanged": int(unchanged.sum()),
        "worsened": int(worsened.sum()),
    }


def summarize_subset(
    coarse_error,
    final_error,
    applied,
    mask,
    total_count,
    tolerance_px,
):
    count = int(mask.sum())
    if count == 0:
        raise ValueError("An analysis range contains no keypoints.")
    coarse_subset = coarse_error[mask]
    final_subset = final_error[mask]
    counts = outcome_counts(
        coarse_subset,
        final_subset,
        tolerance_px,
    )
    mean_coarse = float(coarse_subset.mean())
    mean_final = float(final_subset.mean())
    reduction = (
        100.0 * (mean_coarse - mean_final) / mean_coarse
        if mean_coarse > 0.0
        else 0.0
    )
    return {
        "count": count,
        "proportion_percent": 100.0 * count / total_count,
        "coverage_percent": 100.0 * float(applied[mask].mean()),
        "coarse_error_px": mean_coarse,
        "final_error_px": mean_final,
        "error_reduction_percent": reduction,
        "improved_count": counts["improved"],
        "unchanged_count": counts["unchanged"],
        "worsened_count": counts["worsened"],
        "improved_percent": 100.0 * counts["improved"] / count,
        "unchanged_percent": 100.0 * counts["unchanged"] / count,
        "worsened_percent": 100.0 * counts["worsened"] / count,
    }


def analyze_records(
    always_on_records,
    gated_records,
    *,
    confidence_threshold=0.5,
    tolerance_px=0.5,
):
    arrays = collect_keypoint_arrays(
        always_on_records,
        gated_records,
        confidence_threshold,
    )
    visible = arrays["ground_truth"][:, 2] > 0
    if not np.any(visible):
        raise ValueError("No annotated keypoints were found.")

    coarse = arrays["coarse"][visible]
    ground_truth = arrays["ground_truth"][visible, :2]
    crop_sizes = arrays["crop_sizes"][visible]
    if np.any(crop_sizes <= 0.0):
        raise ValueError("Crop sizes must be positive.")

    normalized_coarse_error = (
        np.max(np.abs(coarse - ground_truth), axis=1)
        / crop_sizes
    )
    coarse_error = np.linalg.norm(
        coarse - ground_truth,
        axis=1,
    )
    policy_data = {
        "always_on": {
            "final": arrays["always_final"][visible],
            "applied": arrays["always_applied"][visible],
        },
        "confidence_gated": {
            "final": arrays["gated_final"][visible],
            "applied": arrays["gated_applied"][visible],
        },
    }

    total_count = int(visible.sum())
    rows = []
    for policy, values in policy_data.items():
        final_error = np.linalg.norm(
            values["final"] - ground_truth,
            axis=1,
        )
        for label, lower, upper, include_upper in BIN_DEFINITIONS:
            if np.isinf(upper):
                mask = normalized_coarse_error > lower
            elif lower == 0.0:
                mask = normalized_coarse_error <= upper
            else:
                upper_test = (
                    normalized_coarse_error <= upper
                    if include_upper
                    else normalized_coarse_error < upper
                )
                mask = (
                    (normalized_coarse_error > lower)
                    & upper_test
                )
            row = summarize_subset(
                coarse_error,
                final_error,
                values["applied"],
                mask,
                total_count,
                tolerance_px,
            )
            row["policy"] = policy
            row["q_range"] = label
            rows.append(row)

        overall = summarize_subset(
            coarse_error,
            final_error,
            values["applied"],
            np.ones(total_count, dtype=bool),
            total_count,
            tolerance_px,
        )
        overall["policy"] = policy
        overall["q_range"] = "overall"
        rows.append(overall)

    return {
        "definitions": {
            "coordinate_system": "resized 224x224 person instance",
            "normalized_coarse_error": (
                "L-infinity(coarse - ground_truth) / crop_size"
            ),
            "confidence": (
                "geometric mean probability of generated tokens"
            ),
            "gate_rule": (
                "refine when confidence >= threshold; otherwise "
                "retain the coarse coordinate"
            ),
            "outcomes": {
                "improved": "final_error - coarse_error < -tolerance",
                "unchanged": (
                    "abs(final_error - coarse_error) <= tolerance"
                ),
                "worsened": "final_error - coarse_error > tolerance",
            },
        },
        "confidence_threshold": confidence_threshold,
        "tolerance_px": tolerance_px,
        "annotated_keypoint_count": total_count,
        "rows": rows,
    }


def write_csv(path, rows):
    fieldnames = (
        "policy",
        "q_range",
        "count",
        "proportion_percent",
        "coverage_percent",
        "coarse_error_px",
        "final_error_px",
        "error_reduction_percent",
        "improved_count",
        "unchanged_count",
        "worsened_count",
        "improved_percent",
        "unchanged_percent",
        "worsened_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows):
    print(
        "| Policy | q range | Count | Proportion | Coverage | "
        "Error (coarse/final) | Reduction | Improved | "
        "Unchanged | Worsened |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['q_range']} | "
            f"{row['count']} | {row['proportion_percent']:.1f}% | "
            f"{row['coverage_percent']:.1f}% | "
            f"{row['coarse_error_px']:.1f}/"
            f"{row['final_error_px']:.1f} | "
            f"{row['error_reduction_percent']:.1f}% | "
            f"{row['improved_percent']:.1f}% | "
            f"{row['unchanged_percent']:.1f}% | "
            f"{row['worsened_percent']:.1f}% |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--always-on-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gated-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/refinement_reliability/analysis"),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--tolerance-px",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be in [0, 1].")
    if args.tolerance_px < 0.0:
        parser.error("--tolerance-px must be non-negative.")

    analysis = analyze_records(
        load_records(args.always_on_predictions),
        load_records(args.gated_predictions),
        confidence_threshold=args.confidence_threshold,
        tolerance_px=args.tolerance_px,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        args.output_dir / "refinement_reliability.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    write_csv(
        args.output_dir / "refinement_reliability.csv",
        analysis["rows"],
    )
    print_markdown(analysis["rows"])


if __name__ == "__main__":
    main()
