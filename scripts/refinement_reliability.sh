#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m utils.run_config \
    --config configs/coco_full.json \
    --stage eval_refinement_always_on \
    "$@"

python -m utils.run_config \
    --config configs/coco_full.json \
    --stage eval_refinement_confidence_gated \
    "$@"

python -m utils.run_config \
    --config configs/coco_full.json \
    --stage analyze_refinement_reliability \
    "$@"
