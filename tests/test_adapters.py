from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from method_adapter import build_command  # noqa: E402
from profile_adapter import build_profile_command  # noqa: E402


class AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = json.loads(
            (ROOT / "configs/method_adapters.json").read_text(encoding="utf-8")
        )
        cls.protocol = json.loads(
            (ROOT / "configs/efficiency.json").read_text(encoding="utf-8")
        )["protocol"]

    def test_native_training_uses_run_config(self):
        command, cwd, _ = build_command(
            action="train",
            method_name="ADCCR",
            dataset_name=None,
            adapters=self.adapters,
            project_config=Path("configs/coco_full.json"),
            seed=3,
            checkpoint=ROOT / "unused",
            output=ROOT / "outputs/seed3",
        )
        self.assertEqual(cwd, ROOT)
        self.assertIn("utils/run_config.py", command)
        self.assertNotIn("train.py", command)

    def test_external_eval_uses_supported_native_flags(self):
        command, _, _ = build_command(
            action="eval",
            method_name="LocLLM",
            dataset_name="coco",
            adapters=self.adapters,
            project_config=Path("configs/coco_full.json"),
            seed=1,
            checkpoint=ROOT / "external/LocLLM/checkpoints/seed1",
            output=ROOT / "outputs/locllm",
        )
        joined = " ".join(command)
        self.assertIn("utils/valid2d.py", joined)
        self.assertIn("--model-name", command)
        self.assertIn("--output-dir", command)
        self.assertNotIn("--config", command)
        self.assertNotIn("--checkpoint", command)

    def test_locllm_mpii_uses_official_two_process_evaluation(self):
        command, _, _ = build_command(
            action="eval",
            method_name="LocLLM",
            dataset_name="mpii",
            adapters=self.adapters,
            project_config=Path("configs/coco_full.json"),
            seed=1,
            checkpoint=ROOT / "external/LocLLM/checkpoints/seed1",
            output=ROOT / "outputs/locllm_mpii",
        )
        self.assertIn("--nproc_per_node=2", command)

    def test_every_efficiency_method_has_a_profile_backend(self):
        dataset = self.adapters["datasets"]["coco"]
        for name, method in self.adapters["methods"].items():
            command, _, _ = build_profile_command(
                method_name=name,
                method=method,
                dataset=dataset,
                raw_output=ROOT / "outputs/profile.raw.json",
                protocol=self.protocol,
                seed=1,
            )
            self.assertTrue(command)
        self.assertTrue((ROOT / "tools/profile_adapter.py").is_file())


if __name__ == "__main__":
    unittest.main()
