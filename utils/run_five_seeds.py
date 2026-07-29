import argparse
import copy
import csv
import json
import statistics

from utils.run_config import REPO_ROOT, load_config, run_stage


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
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-path-check", action="store_true")
    args = parser.parse_args()

    base = load_config(REPO_ROOT / args.config)
    rows = []
    for seed in args.seeds:
        config = copy.deepcopy(base)
        config["experiment"] = f"adccr_seed_{seed}"
        checkpoint = f"checkpoints/seeds/seed_{seed}"
        result_dir = f"results/seed_stability/seed_{seed}"
        config["stages"]["train"]["args"][
            "output_dir"
        ] = checkpoint
        config["stages"]["eval_coco"]["args"][
            "model_name"
        ] = checkpoint
        config["stages"]["eval_coco"]["args"][
            "output_dir"
        ] = result_dir

        if not args.skip_training:
            run_stage(
                config,
                "train",
                seed=seed,
                dry_run=args.dry_run,
                check_paths=not args.skip_path_check,
            )
        if not args.skip_evaluation:
            run_stage(
                config,
                "eval_coco",
                seed=seed,
                dry_run=args.dry_run,
                check_paths=not args.skip_path_check,
            )

        metrics_path = REPO_ROOT / result_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
            rows.append({"seed": seed, "AP": metrics["AP"]})

    if not rows:
        return
    values = [row["AP"] for row in rows]
    summary = {
        "runs": rows,
        "mean_AP": statistics.fmean(values),
        "population_SD": statistics.pstdev(values),
        "sample_SD": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }
    output_dir = REPO_ROOT / "results/seed_stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    with open(
        output_dir / "runs.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "AP"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
