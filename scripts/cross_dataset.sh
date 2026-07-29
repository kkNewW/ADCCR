#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
bash scripts/valid_humanart.sh "$@"
bash scripts/valid_mpii.sh "$@"
