
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


T_CRIT_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
              12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
              17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def population_mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Return the arithmetic mean and population SD (``ddof=0``)."""

    if not values:
        raise ValueError("at least one value is required")
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, sd


def sample_sd(values: Sequence[float]) -> float:
    """Return the sample SD (``ddof=1``) used by t confidence intervals."""

    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(
        sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    )


def exact_sign_flip_p(differences: Sequence[float]) -> float:
    observed = abs(sum(differences) / len(differences))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(sum(s * d for s, d in zip(signs, differences)) / len(differences)))
    return sum(v >= observed - 1e-12 for v in values) / len(values)


def pairwise_ci(differences: Sequence[float]) -> Tuple[float, float, float]:
    mean = sum(differences) / len(differences)
    sd = sample_sd(differences)
    n = len(differences)
    if n < 2 or sd == 0.0:
        return mean, mean, mean
    tcrit = T_CRIT_975.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def holm_adjust(pairs: List[dict]) -> None:
    order = sorted(range(len(pairs)), key=lambda i: pairs[i]["p_exact"])
    adjusted = [0.0] * len(pairs)
    running = 0.0
    m = len(pairs)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pairs[index]["p_exact"])
        running = max(running, value)
        adjusted[index] = running
    for index, value in enumerate(adjusted):
        pairs[index]["p_holm"] = value


def load(path: Path) -> Dict[Tuple[str, str, str], Dict[int, float]]:
    grouped: Dict[Tuple[str, str, str], Dict[int, float]] = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["method"], row.get("metric", "metric"))
            grouped[key][int(row["seed"])] = float(row["value"])
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-pairs", type=Path, required=True)
    parser.add_argument("--reference", default="ADCCR")
    parser.add_argument(
        "--multiplicity-adjustment",
        choices=("none", "holm"),
        default="none",
        help=(
            "Multiple-comparison adjustment applied to exact p-values. "
            "The default 'none' matches the p_exact values reported in Table 14."
        ),
    )
    args = parser.parse_args()

    grouped = load(args.input)
    summary = []
    for (dataset, method, metric), by_seed in sorted(grouped.items()):
        values = [by_seed[s] for s in sorted(by_seed)]
        mean, population_sd = population_mean_sd(values)
        inference_sd = sample_sd(values)
        n = len(values)
        tcrit = T_CRIT_975.get(n - 1, 1.96)
        half = tcrit * inference_sd / math.sqrt(n) if n > 1 else 0.0
        summary.append({
            "dataset": dataset, "method": method, "metric": metric, "n": n,
            "mean": mean,
            "population_sd": population_sd,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "descriptive_sd_ddof": 0,
            "ci_sd_ddof": 1,
        })

    pairs = []
    datasets = sorted({dataset for dataset, _, _ in grouped})
    metrics = sorted({metric for _, _, metric in grouped})
    methods = sorted({method for _, method, _ in grouped})
    for dataset in datasets:
        for metric in metrics:
            if (dataset, args.reference, metric) not in grouped:
                continue
            ref = grouped[(dataset, args.reference, metric)]
            for method in methods:
                if method == args.reference or (dataset, method, metric) not in grouped:
                    continue
                other = grouped[(dataset, method, metric)]
                seeds = sorted(set(ref) & set(other))
                if len(seeds) < 2:
                    continue
                differences = [ref[s] - other[s] for s in seeds]
                delta, ci_low, ci_high = pairwise_ci(differences)
                pairs.append({
                    "dataset": dataset, "metric": metric, "reference": args.reference, "comparison": method,
                    "n": len(seeds), "delta_mean": delta,
                    "ci95_low": ci_low, "ci95_high": ci_high,
                    "p_exact": exact_sign_flip_p(differences),
                    "seeds": ",".join(map(str, seeds)),
                })
    if args.multiplicity_adjustment == "holm":
        holm_adjust(pairs)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    args.out_pairs.write_text(
        json.dumps(pairs, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "paired_comparisons": pairs}, indent=2))


if __name__ == "__main__":
    main()
