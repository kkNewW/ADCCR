import unittest

import numpy as np

from utils.analyze_refinement_reliability import analyze_records
from utils.refinement_policy import (
    merge_refined_coordinates,
    select_refinement_indices,
    validate_refinement_policy,
)


class RefinementPolicyTests(unittest.TestCase):
    def test_confidence_gate_includes_threshold(self):
        indices = select_refinement_indices(
            [0.2, 0.5, 0.9],
            use_confidence_gate=True,
            confidence_threshold=0.5,
        )
        np.testing.assert_array_equal(indices, [1, 2])

    def test_gate_respects_candidate_subset(self):
        indices = select_refinement_indices(
            [0.9, 0.8, 0.7, 0.6],
            use_confidence_gate=False,
            confidence_threshold=0.5,
            candidate_indices=[1, 3],
        )
        np.testing.assert_array_equal(indices, [1, 3])

    def test_fallback_preserves_skipped_coordinates(self):
        coarse = np.asarray(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.float32,
        )
        merged = merge_refined_coordinates(
            coarse,
            [[20.0, 20.0]],
            [1],
        )
        np.testing.assert_allclose(
            merged,
            [[1.0, 1.0], [20.0, 20.0], [3.0, 3.0]],
        )

    def test_gate_requires_local_refiner(self):
        with self.assertRaises(ValueError):
            validate_refinement_policy(False, True, 0.5)


def make_record(final, applied, confidence):
    q_values = np.asarray(
        [0.05] * 4
        + [0.20] * 4
        + [0.40] * 4
        + [0.60] * 5,
        dtype=np.float64,
    )
    coarse = np.stack(
        [100.0 * q_values, np.zeros(17)],
        axis=1,
    )
    ground_truth = np.zeros((17, 3), dtype=np.float64)
    ground_truth[:, 2] = 1.0
    return {
        "annotation_id": 1,
        "coarse_keypoints_224": coarse.reshape(-1).tolist(),
        "final_keypoints_224": final.reshape(-1).tolist(),
        "gt_keypoints_224": ground_truth.reshape(-1).tolist(),
        "refinement_crop_sizes": [100.0] * 17,
        "coarse_confidence": confidence.tolist(),
        "refinement_applied": applied.tolist(),
    }


class RefinementAnalysisTests(unittest.TestCase):
    def test_outcomes_are_computed_from_error_change(self):
        q_values = np.asarray(
            [0.05] * 4
            + [0.20] * 4
            + [0.40] * 4
            + [0.60] * 5,
            dtype=np.float64,
        )
        coarse = np.stack(
            [100.0 * q_values, np.zeros(17)],
            axis=1,
        )
        always_final = coarse.copy()
        always_final[:, 0] -= 1.0
        confidence = np.asarray(
            [0.4, 0.6] * 8 + [0.6],
            dtype=np.float64,
        )
        gated_applied = confidence >= 0.5
        gated_final = coarse.copy()
        gated_final[gated_applied, 0] -= 1.0

        analysis = analyze_records(
            [
                make_record(
                    always_final,
                    np.ones(17, dtype=bool),
                    confidence,
                )
            ],
            [
                make_record(
                    gated_final,
                    gated_applied,
                    confidence,
                )
            ],
            confidence_threshold=0.5,
            tolerance_px=0.5,
        )

        self.assertEqual(analysis["annotated_keypoint_count"], 17)
        gated_overall = next(
            row
            for row in analysis["rows"]
            if row["policy"] == "confidence_gated"
            and row["q_range"] == "overall"
        )
        self.assertEqual(gated_overall["improved_count"], 9)
        self.assertEqual(gated_overall["unchanged_count"], 8)
        self.assertEqual(gated_overall["worsened_count"], 0)
        self.assertAlmostEqual(
            gated_overall["coverage_percent"],
            100.0 * 9 / 17,
        )


if __name__ == "__main__":
    unittest.main()
