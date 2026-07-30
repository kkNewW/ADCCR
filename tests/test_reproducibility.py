import ast
import json
import unittest
from pathlib import Path

from utils.inference import parse_normalized_coordinates
from utils.run_config import build_command, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class CoordinateParserTests(unittest.TestCase):
    def test_bracketed_pair(self):
        self.assertEqual(
            parse_normalized_coordinates(
                "The coordinate is [0.125, 0.875]."
            ),
            (0.125, 0.875),
        )

    def test_boundary_values(self):
        self.assertEqual(
            parse_normalized_coordinates("[0, 1.000]"),
            (0.0, 1.0),
        )

    def test_rejects_out_of_range_or_incomplete(self):
        self.assertIsNone(
            parse_normalized_coordinates("[1.2, 0.4]")
        )
        self.assertIsNone(
            parse_normalized_coordinates("[0.4]")
        )


class ReproducibilityConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            REPO_ROOT / "configs/coco_full.json"
        )

    def test_manuscript_defaults(self):
        args = self.config["stages"]["train"]["args"]
        expected = {
            "mm_projector_type": "mlp",
            "mm_projector_depth": 2,
            "use_dynamic_desc": True,
            "desc_mode": "dynamic",
            "use_local_refiner": True,
            "refiner_input_size": 128,
            "refiner_heatmap_size": 64,
            "refiner_sigma": 2.0,
            "refiner_noise_ratio": 0.25,
            "lambda_hm": 0.5,
            "num_train_epochs": 12,
            "learning_rate": 0.0005,
            "weight_decay": 0.05,
        }
        for key, value in expected.items():
            self.assertEqual(args[key], value)

    def test_all_required_reproduction_stages_exist(self):
        self.assertTrue(
            {
                "train",
                "eval_coco",
                "eval_humanart",
                "eval_mpii",
                "eval_novel",
                "profile",
            }.issubset(self.config["stages"])
        )

    def test_seed_is_applied_to_train_and_data(self):
        command, _, seed = build_command(
            self.config,
            "train",
            seed=5,
        )
        joined = " ".join(command)
        self.assertEqual(seed, 5)
        self.assertIn("--seed 5", joined)
        self.assertIn("--data-seed 5", joined)

    def test_novel_targets_are_excluded_from_training(self):
        with open(
            REPO_ROOT / "configs/novel_keypoints_mpii.json",
            encoding="utf-8",
        ) as handle:
            protocol = json.load(handle)
        training = set(protocol["training_keypoints"])
        unseen = {
            query["name"]
            for query in protocol["queries"]
            if query["status"] == "unseen"
        }
        self.assertTrue(unseen.isdisjoint(training))
        self.assertEqual(
            unseen,
            {"pelvis", "neck", "thorax",},
        )

    def test_all_python_files_parse(self):
        for folder in ("datasets", "models", "utils"):
            for path in (REPO_ROOT / folder).rglob("*.py"):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                )

    def test_no_character_offset_confidence_indexing(self):
        for name in ("valid2d.py", "valid2dmpii.py"):
            source = (
                REPO_ROOT / "utils" / name
            ).read_text(encoding="utf-8")
            self.assertNotIn("pred_kpt.find(", source)
            self.assertIn(
                "generation_sequence_confidence",
                source,
            )


if __name__ == "__main__":
    unittest.main()
