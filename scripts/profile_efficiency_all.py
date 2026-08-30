"""Run one common efficiency protocol for every reported method.

Each adapter receives the same person crop and 17 queries, performs the warmup
and synchronized timed iterations, and writes a JSON object with latency_ms,
flops, and peak_memory_gb. Keeping the adapter interface identical prevents
baseline numbers from being silently measured under a different query or
preprocessing protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/efficiency.json"))
    parser.add_argument("--output", type=Path, default=Path("results/efficiency_results.csv"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = spec["protocol"]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in spec["methods"]:
        name = method["name"]
        json_path = args.output.parent / f"{name.replace(' ', '_')}_profile.json"
        command = shlex.split(
            method["profile_command"].format(json=json_path)
        )
        if args.execute:
            command.append("--execute")
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, check=True)
            metrics = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            metrics = {"latency_ms": None, "flops": None, "peak_memory_gb": None}
        rows.append({
            "method": name,
            "batch_persons": protocol["batch_persons"],
            "joint_predictions_per_person": protocol[
                "joint_predictions_per_person"
            ],
            "language_queries_per_person": protocol[
                "language_queries_per_person"
            ][name],
            "warmup": protocol["warmup"],
            "repeat": protocol["repeat"],
            "latency_ms_per_person": metrics.get("latency_ms"),
            "latency_ms_per_joint": (
                None
                if metrics.get("latency_ms") is None
                else metrics["latency_ms"]
                / protocol["joint_predictions_per_person"]
            ),
            "flops": metrics.get("flops"),
            "peak_memory_gb": metrics.get("peak_memory_gb"),
            "command": shlex.join(command),
        })
    fields = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output.with_suffix(".protocol.json")).write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
