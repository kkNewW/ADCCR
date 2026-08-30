"""Print the configuration-backed records used by manuscript Tables 1-4."""

from __future__ import annotations

import json
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_static_tables() -> dict:
    resources = runpy.run_path(str(REPO_ROOT / "datasets/constants.py"))
    sampler_module = runpy.run_path(str(REPO_ROOT / "datasets/desc_bank.py"))
    config = json.loads(
        (REPO_ROOT / "configs/coco_full.json").read_text(encoding="utf-8")
    )
    bank = resources["DESCRIPTION_BANK"]
    canonical = resources["KeypointLocationDescription"]
    questions = resources["KeypointLocationQuestion"]
    train = config["stages"]["train"]["args"]

    return {
        "table_1": {
            "structured_keypoints": len(bank),
            "field_counts": {
                field: sum(len(item[field]) for item in bank.values())
                for field in ("name", "anatomy", "relation", "visual")
            },
            "canonical_descriptions": len(canonical),
            "question_templates": sum(len(items) for items in questions.values()),
        },
        "table_2": {
            "modes": sampler_module["DescriptionSampler"].MODE_FIELDS,
            "probabilities": config["description"]["train_sampler_probabilities"],
        },
        "table_3": {
            "crop_sizes": config["refiner"]["crop_sizes"],
            "source": config["refiner"]["crop_size_source"],
        },
        "table_4": {
            "visual_encoder": train["dino_path"],
            "language_model": train["model_name_or_path"],
            "connector": {
                "type": train["mm_projector_type"],
                "depth": train["mm_projector_depth"],
                "trainable": train["tune_mm_mlp_adapter"],
            },
            "local_refiner": {
                "enabled": train["use_local_refiner"],
                "text_conditioned": train["refiner_use_text"],
            },
            "adaptation": {
                "vision_lora": train["lora_vision_enable"],
                "llm_lora": train["lora_llm_enable"],
            },
        },
    }


def main() -> None:
    print(json.dumps(build_static_tables(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
