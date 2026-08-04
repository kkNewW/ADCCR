#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/main_results.sh
bash scripts/efficiency.sh
bash scripts/cross_dataset.sh
bash scripts/novel_keypoints.sh
bash scripts/ablations.sh
bash scripts/seed_stability.sh
