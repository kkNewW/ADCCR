from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))

from refinement_policy import (  # noqa: E402
    RefinementPolicy,
    classify_outcome,
    geometric_mean_probability,
    merge_refined_coordinates,
    select_refinement_indices,
    validate_refinement_policy,
)
from analyze_refinement_reliability import _bin, _infinity_error, analyse  # noqa: E402
from refinement_trace import detailed_prediction_records  # noqa: E402


class RefinementPolicyTest(unittest.TestCase):
    def test_threshold_fallback(self):
        policy = RefinementPolicy(0.5)
        final, accepted = policy.apply((1, 1), (2, 2), 0.49)
        self.assertEqual(final, (1, 1))
        self.assertFalse(accepted)

    def test_threshold_accept(self):
        policy = RefinementPolicy(0.5)
        final, accepted = policy.apply((1, 1), (2, 2), 0.5)
        self.assertEqual(final, (2, 2))
        self.assertTrue(accepted)

    def test_optional_displacement_guard(self):
        policy = RefinementPolicy(0.5, max_displacement=1.0)
        final, accepted = policy.apply((0, 0), (2, 0), 0.9)
        self.assertEqual(final, (0, 0))
        self.assertFalse(accepted)

    def test_geometric_mean(self):
        self.assertAlmostEqual(geometric_mean_probability([0.25, 1.0]), 0.5)
        self.assertEqual(geometric_mean_probability([]), 0.0)

    def test_outcome_tolerance(self):
        self.assertEqual(classify_outcome(10, 9.4, tolerance=0.5), "Improved")
        self.assertEqual(classify_outcome(10, 10.4, tolerance=0.5), "Unchanged")
        self.assertEqual(classify_outcome(10, 10.6, tolerance=0.5), "Worsened")

    def test_refinement_policy_validation(self):
        validate_refinement_policy(True, True, 0.5)
        with self.assertRaises(ValueError):
            validate_refinement_policy(False, True, 0.5)
        with self.assertRaises(ValueError):
            validate_refinement_policy(True, True, 1.1)

    def test_select_refinement_indices(self):
        selected = select_refinement_indices(
            [0.9, 0.2, 0.5],
            use_confidence_gate=True,
            confidence_threshold=0.5,
        )
        self.assertEqual(selected.tolist(), [0, 2])
        candidates = select_refinement_indices(
            [0.9, 0.2, 0.5],
            use_confidence_gate=False,
            confidence_threshold=0.5,
            candidate_indices=[2, 0],
        )
        self.assertEqual(candidates.tolist(), [2, 0])

    def test_merge_refined_coordinates(self):
        merged = merge_refined_coordinates(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[20.0, 21.0], [30.0, 31.0]],
            [1, 2],
        )
        self.assertEqual(
            merged.tolist(),
            [[1.0, 1.0], [20.0, 21.0], [30.0, 31.0]],
        )

    def test_records_are_independent(self):
        rows = [
            {
                "image_id": 1,
                "person_id": 1,
                "keypoint": "left_wrist",
                "crop_size": 96,
                "gt": [10, 10],
                "coarse": [11, 10],
                "refined": [10, 10],
                "sequence_confidence": 0.9,
                "description": "left wrist",
            },
            {
                "image_id": 1,
                "person_id": 1,
                "keypoint": "right_wrist",
                "crop_size": 96,
                "gt": [20, 10],
                "coarse": [21, 10],
                "refined": [25, 10],
                "sequence_confidence": 0.1,
                "description": "right wrist",
            },
        ]
        records, summary = analyse(rows, RefinementPolicy(0.5))
        gated = [r for r in records if r["policy"] == "gated"]
        self.assertEqual(len(gated), 2)
        self.assertEqual(gated[0]["final"], [10.0, 10.0])
        self.assertEqual(gated[1]["final"], [21.0, 10.0])
        self.assertTrue(any(r["error_range"] == "0-0.10" for r in summary))

    def test_error_bins_match_table_16_boundaries(self):
        self.assertEqual(_bin(0.0), "0-0.10")
        self.assertEqual(_bin(0.10), "0-0.10")
        self.assertEqual(_bin(0.100001), "0.10-0.25")
        self.assertEqual(_bin(0.25), "0.10-0.25")
        self.assertEqual(_bin(0.50), "0.25-0.50")
        self.assertEqual(_bin(0.500001), ">0.50")

    def test_table_16_stratification_uses_infinity_norm(self):
        self.assertEqual(_infinity_error((0, 0), (3, 4)), 4.0)

    def test_detailed_predictions_expand_to_raw_jsonl_rows(self):
        people = [
            {
                "image_id": 9,
                "annotation_id": 11,
                "keypoint_names": ["left wrist", "right wrist"],
                "descriptions": ["left", "right"],
                "gt_keypoints_224": [10, 10, 2, 20, 20, 0],
                "coarse_keypoints_224": [12, 10, 18, 20],
                "refined_keypoints_224": [11, 10, 19, 20],
                "final_keypoints_224": [11, 10, 18, 20],
                "coarse_confidence": [0.9, 0.2],
                "refinement_crop_sizes": [96, 96],
            }
        ]
        records = list(detailed_prediction_records(people))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["person_id"], 11)
        self.assertEqual(records[0]["refined"], [11.0, 10.0])


if __name__ == "__main__":
    unittest.main()
