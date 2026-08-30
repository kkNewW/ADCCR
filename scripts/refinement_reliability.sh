#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# The validation stage emits one raw row per visible keypoint. The analysis
# wrapper then applies the fixed gate to the same always-on candidates and
# derives both Table 16 panels from those paired records.
python utils/run_config.py \
  --config configs/coco_full.json \
  --stage eval_refinement_always_on \
  "$@"
python scripts/run_error_propagation.py \
  --input results/refinement_reliability/raw_refinement_predictions.jsonl \
  --output results/refinement_reliability/analysis \
  --threshold 0.5
