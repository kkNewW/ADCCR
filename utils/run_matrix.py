import argparse
import copy
import csv
import json
from pathlib import Path

from utils.run_config import (
    REPO_ROOT,
    apply_override,
    load_config,
    run_stage,
)


def set_path(config, dotted_path, value):
    keys = dotted_path.split(".")
    cursor = config
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value


def write_summary(rows, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted(
        {key for row in rows for key in row.keys()}
    )
    with open(
        destination,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="configs/coco_full.json",
    )
    parser.add_argument(
        "--matrix",
        default="configs/ablation_matrix.json",
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="Run only an exact experiment name.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-path-check", action="store_true")
    args = parser.parse_args()

    base = load_config(REPO_ROOT / args.base)
    with open(
        REPO_ROOT / args.matrix,
        encoding="utf-8",
    ) as handle:
        matrix = json.load(handle)
    if matrix.get("schema_version") != 1:
        raise ValueError("Unsupported ablation matrix schema.")

    rows = []
    selected = set(args.select)
    for experiment in matrix["experiments"]:
        name = experiment["name"]
        if selected and name not in selected:
            continue
        config = copy.deepcopy(base)
        config["experiment"] = name
        for dotted_path, value in experiment[
            "overrides"
        ].items():
            apply_override(
                config,
                f"{dotted_path}={json.dumps(value)}",
            )

        checkpoint = f"checkpoints/ablations/{name}"
        result_dir = f"results/ablations/{name}"
        set_path(
            config,
            "stages.train.args.output_dir",
            checkpoint,
        )
        set_path(
            config,
            "stages.eval_coco.args.model_name",
            checkpoint,
        )
        set_path(
            config,
            "stages.eval_coco.args.output_dir",
            result_dir,
        )

        if not args.skip_training:
            run_stage(
                config,
                "train",
                seed=args.seed,
                dry_run=args.dry_run,
                check_paths=not args.skip_path_check,
            )
        if not args.skip_evaluation:
            run_stage(
                config,
                "eval_coco",
                seed=args.seed,
                dry_run=args.dry_run,
                check_paths=not args.skip_path_check,
            )

        metrics_path = REPO_ROOT / result_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
            row = {
                "experiment": name,
                "result_groups": ",".join(
                    experiment["result_groups"]
                ),
            }
            row.update(
                {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            joint_scores = metrics.get(
                "difficult_joint_metric",
                {},
            ).get("scores", {})
            for joint in ("wrist", "ankle", "elbow", "knee"):
                row[f"{joint}_pck"] = joint_scores.get(joint)
            rows.append(row)

    write_summary(
        rows,
        REPO_ROOT / "results/ablations/summary.csv",
    )


if __name__ == "__main__":
    main()
