import argparse
import copy
import csv
import json
import statistics
from collections import defaultdict

from utils.run_config import REPO_ROOT, load_config, run_stage


EXPECTED_SEEDS = (1, 2, 3, 4, 5)


EVALUATION_SPECS = {
    "COCO": {
        "stage": "eval_coco",
        "directory": "coco",
        "scale": 100.0,
        "primary_metric": "AP",
    },
    "Human-Art": {
        "stage": "eval_humanart",
        "directory": "humanart",
        "scale": 100.0,
        "primary_metric": "AP",
    },
    "MPII": {
        "stage": "eval_mpii",
        "directory": "mpii",
        "scale": 1.0,
        "primary_metric": "Mean",
    },
}


def read_numeric_metrics(metrics_path, scale):
    """Read scalar metrics and convert them to paper units."""
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

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


def validate_primary_runs(rows, seeds):
    """Require the primary metric for every dataset and seed."""
    observed = {
        (
            row["dataset"],
            row["metric"],
            row["seed"],
        )
        for row in rows
    }

    missing = []
    for dataset, spec in EVALUATION_SPECS.items():
        for seed in seeds:
            key = (
                dataset,
                spec["primary_metric"],
                seed,
            )
            if key not in observed:
                missing.append(key)

    if missing:
        details = ", ".join(
            f"{dataset}/{metric}/seed_{seed}"
            for dataset, metric, seed in missing
        )
        raise RuntimeError(
            "Incomplete five-seed primary results: "
            + details
        )


def summarize(rows):
    """Calculate the mean and sample SD for each metric."""
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["metric"],
        )
        grouped[key].append(row["value"])

    summaries = []

    for (dataset, metric), values in sorted(
        grouped.items()
    ):
        summaries.append(
            {
                "dataset": dataset,
                "metric": metric,
                "runs": len(values),
                "mean": statistics.fmean(values),
                "sample_SD": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
            }
        )

    return summaries


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
        default=list(EXPECTED_SEEDS),
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

    if tuple(args.seeds) != EXPECTED_SEEDS:
        raise ValueError(
            "The matched five-seed protocol requires "
            "exactly: 1 2 3 4 5"
        )

    base = load_config(
        REPO_ROOT / args.config
    )

    output_root = (
        REPO_ROOT / "results/seed_stability"
    )

    rows = []

    for seed in args.seeds:
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

    validate_primary_runs(
        rows,
        args.seeds,
    )

    summaries = summarize(rows)

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

    write_csv(
        output_root / "summary.csv",
        summaries,
        [
            "dataset",
            "metric",
            "runs",
            "mean",
            "sample_SD",
        ],
    )

    report = {
        "seeds": args.seeds,
        "runs": rows,
        "summaries": summaries,
    }

    with open(
        output_root / "summary.json",
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