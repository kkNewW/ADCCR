import argparse
import copy
import csv
import json
from collections import Counter

from utils.run_config import REPO_ROOT, load_config, run_stage


EVALUATION_SPECS = {
    "COCO": {
        "stage": "eval_coco",
        "directory": "coco",
        "scale": 100.0,
        "table_metrics": (
            "AP",
            "AP50",
            "AP75",
            "APM",
            "APL",
            "AR",
        ),
    },
    "Human-Art": {
        "stage": "eval_humanart",
        "directory": "humanart",
        "scale": 100.0,
        "table_metrics": (
            "AP",
            "AP50",
            "AP75",
            "APM",
            "APL",
            "AR",
        ),
    },
    "MPII": {
        "stage": "eval_mpii",
        "directory": "mpii",
        "scale": 1.0,
        "table_metrics": (
            "Shoulder",
            "Elbow",
            "Hip",
            "Knee",
            "Mean",
            "Mean@0.1",
        ),
    },
}


def read_numeric_metrics(
        metrics_path,
        scale,
        expected_seed,
):
    """Read per-run metrics and verify the recorded seed."""
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

    recorded_seed = metrics.get(
        "protocol",
        {},
    ).get("seed")

    if recorded_seed != expected_seed:
        raise RuntimeError(
            f"Seed mismatch in {metrics_path}: "
            f"expected {expected_seed}, "
            f"found {recorded_seed}"
        )

    return {
        name: float(value) * scale
        for name, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def write_csv(path, rows, fieldnames):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_table_runs(rows, seeds):
    counts = Counter(
        (
            row["dataset"],
            row["metric"],
            row["seed"],
        )
        for row in rows
    )

    problems = []
    for dataset, spec in EVALUATION_SPECS.items():
        for metric in spec["table_metrics"]:
            for seed in seeds:
                key = (dataset, metric, seed)
                count = counts.get(key, 0)

                if count == 0:
                    problems.append(
                        "missing: "
                        f"{dataset}/{metric}/seed_{seed}"
                    )
                elif count > 1:
                    problems.append(
                        f"duplicated ({count}): "
                        f"{dataset}/{metric}/seed_{seed}"
                    )

    if problems:
        raise RuntimeError(
            "Incomplete or duplicated five-seed results:\n- "
            + "\n- ".join(problems)
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/coco_full.json",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--skip-path-check",
        action="store_true",
    )

    args = parser.parse_args()

    base = load_config(
        REPO_ROOT / args.config
    )

    configured_seeds = tuple(
        base["seed_protocol"]["five_seed_runs"]
    )
    seeds = tuple(
        args.seeds
        if args.seeds is not None
        else configured_seeds
    )

    if seeds != configured_seeds:
        raise ValueError(
            "The five-seed run requires exactly: "
            + " ".join(map(str, configured_seeds))
        )

    output_root = (
        REPO_ROOT / "results/seed_stability"
    )

    rows = []

    for seed in seeds:
        config = copy.deepcopy(base)

        config["experiment"] = (
            f"adccr_seed_{seed}"
        )

        checkpoint = (
            f"checkpoints/seeds/seed_{seed}"
        )

        config["stages"]["train"]["args"][
            "output_dir"
        ] = checkpoint

        # Make all three evaluations use the checkpoint
        # trained with the current seed.
        for spec in EVALUATION_SPECS.values():
            result_dir = (
                "results/seed_stability/"
                f"{spec['directory']}/seed_{seed}"
            )

            eval_args = config[
                "stages"
            ][spec["stage"]]["args"]

            eval_args["model_name"] = checkpoint
            eval_args["output_dir"] = result_dir

        # Train one independent checkpoint.
        if not args.skip_training:
            run_stage(
                config,
                "train",
                seed=seed,
                dry_run=args.dry_run,
                check_paths=(
                    not args.skip_path_check
                ),
            )

        # Evaluate the same checkpoint on all datasets.
        for dataset, spec in (
            EVALUATION_SPECS.items()
        ):
            if not args.skip_evaluation:
                run_stage(
                    config,
                    spec["stage"],
                    seed=seed,
                    dry_run=args.dry_run,
                    check_paths=(
                        not args.skip_path_check
                    ),
                )

            if args.dry_run:
                continue

            result_dir = config[
                "stages"
            ][spec["stage"]]["args"][
                "output_dir"
            ]

            metrics_path = (
                REPO_ROOT
                / result_dir
                / "metrics.json"
            )

            if not metrics_path.exists():
                raise FileNotFoundError(
                    "Missing metrics for "
                    f"{dataset}, seed {seed}: "
                    f"{metrics_path}"
                )

            metrics = read_numeric_metrics(
                metrics_path,
                spec["scale"],
                expected_seed=seed,
            )

            for metric, value in metrics.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "seed": seed,
                        "value": value,
                    }
                )

    if args.dry_run:
        return

    validate_table_runs(
        rows,
        seeds,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_root / "runs.csv",
        rows,
        [
            "dataset",
            "metric",
            "seed",
            "value",
        ],
    )

    report = {
        "seeds": list(seeds),
        "runs": rows,
    }

    with open(
        output_root / "five_seed_runs.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
