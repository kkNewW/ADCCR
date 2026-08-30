from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_all import EXPECTED_TABLES, validate_manifest  # noqa: E402


class ReproductionManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "configs/reproduction_manifest.json"
        cls.manifest = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_manifest_uses_expected_filename(self):
        self.assertTrue(self.path.is_file())
        self.assertFalse((ROOT / "configs/reproduction_manifest .json").exists())

    def test_every_manuscript_table_is_mapped(self):
        self.assertEqual(set(self.manifest["tables"]), EXPECTED_TABLES)
        validate_manifest(self.manifest, ROOT)

    def test_five_seed_runner_includes_all_compared_methods(self):
        methods = {item["name"] for item in self.manifest["methods"]}
        self.assertEqual(methods, {"LocLLM", "PoseLLM", "ADCCR"})

    def test_issue_three_entrypoints_are_complete(self):
        entrypoints = self.manifest["entrypoints"]
        self.assertIn("run_five_seed_all.py", entrypoints["main_cross_dataset_and_five_seed_tables_5_7_14"])
        self.assertIn("profile_efficiency_all.py", entrypoints["efficiency_table_6"])
        self.assertIn("refinement_reliability.sh", entrypoints["error_propagation_table_16"])


if __name__ == "__main__":
    unittest.main()
