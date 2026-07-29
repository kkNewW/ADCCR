#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python utils/run_config.py \
  --config configs/coco_full.json \
  --stage eval_humanart \
  "$@"
