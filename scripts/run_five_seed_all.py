"""Run or inspect the matched five-seed protocol for all three methods."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _adapter_command(
    *,
    action: str,
    method: str,
    seed: int,
    output: Path,
    adapters: Path,
    config: Path,
    checkpoint: Path | None = None,
    dataset: str | None = None,
    execute: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        "tools/method_adapter.py",
        action,
        "--method",
        method,
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--adapters",
        str(adapters),
        "--config",
        str(config),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if dataset is not None:
        command.extend(["--dataset", dataset])
    if execute:
        command.append("--execute")
    return command


def _write_seed_csv(rows: List[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "method", "seed", "metric", "value", "provenance"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/reproduction_manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/coco_full.json"),
    )
    parser.add_argument(
        "--adapters",
        type=Path,
        default=Path("configs/method_adapters.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/five_seed"),
    )
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    commands = []
    rows = []
    for method_item in spec["methods"]:
        method = method_item["name"] if isinstance(method_item, dict) else method_item
        for seed in spec["seeds"]:
            run_dir = args.output / method / f"seed_{seed}"
            checkpoint = run_dir / "checkpoint"
            train = _adapter_command(
                action="train",
                method=method,
                seed=seed,
                output=checkpoint,
                adapters=args.adapters,
                config=args.config,
                execute=args.execute,
            )
            commands.append({"method": method, "seed": seed, "stage": "train", "command": train})
            print(shlex.join(train))
            if args.execute:
                subprocess.run(train, cwd=REPO_ROOT, check=True)

            for dataset in spec["evaluation_datasets"]:
                evaluation_dir = run_dir / dataset
                evaluate = _adapter_command(
                    action="eval",
                    method=method,
                    seed=seed,
                    output=evaluation_dir,
                    adapters=args.adapters,
                    config=args.config,
                    checkpoint=checkpoint,
                    dataset=dataset,
                    execute=args.execute,
                )
                commands.append(
                    {"method": method, "seed": seed, "stage": f"eval_{dataset}", "command": evaluate}
                )
                print(shlex.join(evaluate))
                if args.execute:
                    subprocess.run(evaluate, cwd=REPO_ROOT, check=True)
                    metric = json.loads(
                        (evaluation_dir / "primary_metrics.json").read_text(encoding="utf-8")
                    )
                    rows.append(
                        {
                            "dataset": metric["dataset"],
                            "method": metric["method"],
                            "seed": metric["seed"],
                            "metric": metric["metric"],
                            "value": metric["value"],
                            "provenance": "executed_primary_metrics_json",
                        }
                    )

    commands_path = args.output / "commands.json"
    commands_path.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {commands_path}")
    if not args.execute:
        return

    seed_csv = args.output / "seed_metrics.csv"
    _write_seed_csv(rows, seed_csv)
    statistics_command = [
        sys.executable,
        "scripts/paired_seed_stats.py",
        "--input",
        str(seed_csv),
        "--out-summary",
        str(args.output / "seed_summary.json"),
        "--out-pairs",
        str(args.output / "paired_comparisons.json"),
    ]
    subprocess.run(statistics_command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
