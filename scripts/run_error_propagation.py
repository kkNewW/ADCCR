"""Run the auditable error-propagation pipeline after validation.

Validation must save one JSONL record per keypoint containing coarse,
always-on refined, gated confidence, and ground-truth coordinates.  This
wrapper performs the analysis and writes the exact inputs and command metadata
alongside the table-ready CSV.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/refinement_reliability"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out_json = args.output / "reliability_records.json"
    out_csv = args.output / "reliability_summary.csv"
    command = [
        sys.executable,
        "utils/analyze_refinement_reliability.py",
        "--input", str(args.input),
        "--out-json", str(out_json),
        "--out-csv", str(out_csv),
        "--threshold", str(args.threshold),
    ]
    subprocess.run(command, check=True)
    (args.output / "analysis_command.json").write_text(json.dumps({
        "command": command,
        "threshold": args.threshold,
        "input": str(args.input),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
