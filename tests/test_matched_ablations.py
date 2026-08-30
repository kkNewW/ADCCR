from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_matched_ablations import (  # noqa: E402
    _matched_rows,
    _validate_rows,
    build_jobs,
)

DESC_BANK_SPEC = importlib.util.spec_from_file_location(
    "adccr_desc_bank",
    ROOT / "datasets/desc_bank.py",
)
DESC_BANK_MODULE = importlib.util.module_from_spec(DESC_BANK_SPEC)
DESC_BANK_SPEC.loader.exec_module(DESC_BANK_MODULE)
DescriptionSampler = DESC_BANK_MODULE.DescriptionSampler


class MatchedAblationConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads(
            (ROOT / "configs/ablation_matrix.json").read_text(encoding="utf-8")
        )

    def test_single_canonical_matrix(self):
        self.assertFalse((ROOT / "configs/ablation_matched.json").exists())
        rows = _matched_rows(self.matrix)
        _validate_rows(rows)
        self.assertEqual(len(rows), 10)

    def test_all_rows_disable_refinement(self):
        for row in _matched_rows(self.matrix):
            self.assertFalse(row["overrides"]["stages.train.args.use_local_refiner"])
            self.assertFalse(row["overrides"]["stages.eval_coco.args.use_local_refiner"])

    def test_sampling_fixed_name_uses_name_only_in_both_stages(self):
        row = next(
            row
            for row in _matched_rows(self.matrix)
            if row["name"] == "sampling_fixed_name"
        )
        overrides = row["overrides"]
        self.assertTrue(overrides["stages.train.args.use_dynamic_desc"])
        self.assertEqual(overrides["stages.train.args.desc_mode"], "name_only")
        self.assertTrue(overrides["stages.eval_coco.args.use_dynamic_desc"])
        self.assertEqual(
            overrides["stages.eval_coco.args.eval_desc_mode"],
            "name_only",
        )

    def test_name_only_description_contains_only_the_label(self):
        bank = {
            "left wrist": {
                "name": ["left wrist"],
                "anatomy": ["An anatomy sentence."],
                "relation": ["A relation sentence."],
                "visual": ["A visual sentence."],
            }
        }
        description, mode = DescriptionSampler(
            bank
        ).build_description("left wrist", mode="name_only")
        self.assertEqual(mode, "name_only")
        self.assertEqual(description, "Target keypoint: left wrist.")

    def test_commands_use_existing_runner_interfaces(self):
        jobs = build_jobs(
            matrix=self.matrix,
            base_config=Path("configs/coco_full.json"),
            seed=1,
            workdir=Path("outputs/matched_ablations"),
        )
        for job in jobs:
            for command in (job["train"], job["eval"]):
                joined = " ".join(command)
                self.assertIn("utils/run_config.py", joined)
                self.assertNotIn(" train.py", joined)
                self.assertNotIn("--checkpoint", command)
                self.assertNotIn("--output", command)

    def test_enabled_refiner_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_rows(
                [
                    {
                        "name": "bad",
                        "result_groups": ["description_ablation"],
                        "overrides": {
                            "stages.train.args.use_local_refiner": True,
                            "stages.eval_coco.args.use_local_refiner": False,
                        },
                    },
                    {
                        "name": "sampling",
                        "result_groups": ["sampling_ablation"],
                        "overrides": {
                            "stages.train.args.use_local_refiner": False,
                            "stages.eval_coco.args.use_local_refiner": False,
                        },
                    },
                ]
            )

    def test_fixed_name_canonical_fallback_is_rejected(self):
        rows = copy.deepcopy(_matched_rows(self.matrix))
        for row in rows:
            if row["name"] == "sampling_fixed_name":
                row["overrides"][
                    "stages.eval_coco.args.use_dynamic_desc"
                ] = False
                row["overrides"][
                    "stages.eval_coco.args.eval_desc_mode"
                ] = "fixed"
        with self.assertRaisesRegex(ValueError, "fixed-name condition"):
            _validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
