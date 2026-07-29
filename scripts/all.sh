#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/reproduce_main_results.sh
bash scripts/reproduce_efficiency.sh
bash scripts/reproduce_cross_dataset.sh
bash scripts/reproduce_novel_keypoints.sh
bash scripts/reproduce_ablations.sh
bash scripts/reproduce_seed_stability.sh
