import argparse
import json
from pathlib import Path


EXPECTED_CONFIG = {
    "model_type": "ADCCR",
    "mm_projector_type": "mlp",
    "mm_projector_depth": 2,
    "use_local_refiner": True,
    "refiner_input_size": 128,
    "refiner_heatmap_size": 64,
    "refiner_text_dim": 768,
    "refiner_feat_dim": 256,
    "refiner_sigma": 2.0,
    "refiner_noise_ratio": 0.25,
    "refiner_crop_scale": 1.0,
    "refiner_use_text": True,
    "lambda_hm": 0.5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint config is missing: {config_path}"
        )
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    mismatches = {}
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            mismatches[key] = {
                "expected": expected,
                "actual": actual,
            }

    weight_candidates = (
        checkpoint / "pytorch_model.bin",
        checkpoint / "model.safetensors",
        checkpoint / "pytorch_model.bin.index.json",
        checkpoint / "model.safetensors.index.json",
    )
    if not any(path.is_file() for path in weight_candidates):
        raise FileNotFoundError(
            "No Hugging Face model weights or weight index found "
            f"under {checkpoint}."
        )
    if mismatches:
        raise RuntimeError(
            "Checkpoint does not match the manuscript method:\n"
            + json.dumps(mismatches, indent=2)
        )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "status": "compatible",
                "checked": EXPECTED_CONFIG,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
