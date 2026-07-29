#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python - <<'PY'
import importlib.metadata
import json
import platform

packages = [
    "accelerate",
    "numpy",
    "opencv-python",
    "pycocotools",
    "scipy",
    "sentencepiece",
    "torch",
    "torchvision",
    "transformers",
]
print(json.dumps({
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in packages
    },
}, indent=2))
PY

python utils/check_checkpoint.py checkpoints/ckpts/coco
python -m unittest discover -s tests -v
python utils/run_config.py \
  --config configs/coco_full.json \
  --stage train \
  --dry-run
python utils/run_config.py \
  --config configs/coco_full.json \
  --stage eval_coco \
  --dry-run
