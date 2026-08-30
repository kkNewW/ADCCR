import argparse
import copy
import csv
import json
from pathlib import Path

from utils.prompt_variants import PROMPT_VARIANTS
from utils.run_config import (
    REPO_ROOT,
    load_config,
    run_stage,
)


def read_metrics(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(records, destination):
    canonical_ap = (
        records["canonical"]["AP"] * 100.0
    )
    rows = []

    for variant in records:
        metrics = records[variant]
        ap = metrics["AP"] * 100.0
        protocol = metrics.get("protocol", {})

        rows.append(
            {
                "variant": variant,
                "AP": round(ap, 3),
                "AP75": round(
                    metrics["AP75"] * 100.0,
                    3,
                ),
                "delta_AP": round(
                    ap - canonical_ap,
                    3,
                ),
                "prompt_sha256": protocol.get(
                    "prompt_variant_sha256"
                ),
            }
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with destination.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "AP",
                "AP75",
                "delta_AP",
                "prompt_sha256",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/coco_full.json",
    )
    parser.add_argument(
        "--prompt-file",
        default="configs/prompt_variants.json",
    )
    parser.add_argument(
        "--output-root",
        default="results/prompt_robustness",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=PROMPT_VARIANTS,
        default=list(PROMPT_VARIANTS),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
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

    if "canonical" not in args.variants:
        raise ValueError(
            "The canonical condition is required "
            "to compute delta_AP."
        )

    base_config = load_config(
        REPO_ROOT / args.config
    )
    records = {}

    for variant in args.variants:
        config = copy.deepcopy(base_config)
        config["experiment"] = (
            f"prompt_robustness_{variant}"
        )

        eval_args = config[
            "stages"
        ]["eval_coco"]["args"]

        eval_args["prompt_variant_file"] = (
            args.prompt_file
        )
        eval_args["prompt_variant"] = variant
        eval_args["output_dir"] = str(
            Path(args.output_root) / variant
        )

        run_stage(
            config,
            "eval_coco",
            seed=args.seed,
            dry_run=args.dry_run,
            check_paths=not args.skip_path_check,
        )

        if args.dry_run:
            continue

        metrics_path = (
            REPO_ROOT
            / eval_args["output_dir"]
            / "metrics.json"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing evaluation output: "
                f"{metrics_path}"
            )

        records[variant] = read_metrics(
            metrics_path
        )

    if not args.dry_run:
        write_summary(
            records,
            REPO_ROOT
            / args.output_root
            / "summary.csv",
        )


if __name__ == "__main__":
    main()