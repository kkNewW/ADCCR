"""Generate and optionally execute the matched Tables 11/12 ablations.

``configs/ablation_matrix.json`` is the only ablation source of truth. Rows
tagged ``description_ablation`` or ``sampling_ablation`` are selected here and
must explicitly disable the local refiner in both training and evaluation.
The Table 12 ``sampling_fixed_name`` row is also checked to ensure that both
stages use the name-only description path rather than the canonical fallback.
Commands are routed through ``utils/run_config.py``; no separate ``train.py``
or unsupported validation arguments are required.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHED_GROUPS = {"description_ablation", "sampling_ablation"}
REFINER_KEYS = (
    "stages.train.args.use_local_refiner",
    "stages.eval_coco.args.use_local_refiner",
)
FIXED_NAME_ROW = "sampling_fixed_name"
FIXED_NAME_REQUIREMENTS = {
    "stages.train.args.use_dynamic_desc": True,
    "stages.train.args.desc_mode": "name_only",
    "stages.eval_coco.args.use_dynamic_desc": True,
    "stages.eval_coco.args.eval_desc_mode": "name_only",
}


def _matched_rows(matrix: Mapping[str, object]) -> List[dict]:
    rows = []
    for row in matrix.get("experiments", []):
        groups = set(row.get("result_groups", []))
        if groups & MATCHED_GROUPS:
            rows.append(dict(row))
    return rows


def _validate_rows(rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No description/sampling ablation rows were found.")
    names = set()
    seen_groups = set()
    fixed_name_seen = False
    for row in rows:
        name = str(row.get("name", ""))
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate ablation row name: {name!r}")
        names.add(name)
        seen_groups.update(set(row.get("result_groups", [])) & MATCHED_GROUPS)
        overrides = row.get("overrides", {})
        for key in REFINER_KEYS:
            if overrides.get(key) is not False:
                raise ValueError(
                    f"{name}: matched language ablations require {key}=false"
                )
        if name == FIXED_NAME_ROW:
            fixed_name_seen = True
            for key, expected in FIXED_NAME_REQUIREMENTS.items():
                if overrides.get(key) != expected:
                    raise ValueError(
                        f"{name}: {key} must be {expected!r} so the "
                        "Table 12 fixed-name condition uses only the "
                        "keypoint label"
                    )
    if seen_groups != MATCHED_GROUPS:
        raise ValueError(
            "The matrix must contain both description_ablation and "
            "sampling_ablation rows."
        )
    if not fixed_name_seen:
        raise ValueError(
            "The matrix must contain the Table 12 sampling_fixed_name row."
        )


def _override_expression(key: str, value: object) -> str:
    return f"{key}={json.dumps(value, separators=(',', ':'))}"


def _runner_command(
    *,
    config: Path,
    stage: str,
    seed: int,
    overrides: Mapping[str, object],
) -> List[str]:
    command = [
        sys.executable,
        "utils/run_config.py",
        "--config",
        config.as_posix(),
        "--stage",
        stage,
        "--seed",
        str(seed),
    ]
    for key, value in sorted(overrides.items()):
        command.extend(["--set", _override_expression(key, value)])
    return command


def build_jobs(
    *,
    matrix: Mapping[str, object],
    base_config: Path,
    seed: int,
    workdir: Path,
) -> List[dict]:
    rows = _matched_rows(matrix)
    _validate_rows(rows)
    jobs = []
    for row in rows:
        name = str(row["name"])
        groups = sorted(set(row["result_groups"]) & MATCHED_GROUPS)
        output = workdir / name
        train_output = output / "checkpoint"
        eval_output = output / "eval_coco"
        overrides = dict(row["overrides"])

        train_overrides = dict(overrides)
        train_overrides["stages.train.args.output_dir"] = train_output.as_posix()
        eval_overrides = dict(overrides)
        eval_overrides["stages.eval_coco.args.model_name"] = train_output.as_posix()
        eval_overrides["stages.eval_coco.args.output_dir"] = eval_output.as_posix()

        jobs.append(
            {
                "name": name,
                "result_groups": groups,
                "seed": seed,
                "local_refiner": False,
                "overrides": overrides,
                "train": _runner_command(
                    config=base_config,
                    stage="train",
                    seed=seed,
                    overrides=train_overrides,
                ),
                "eval": _runner_command(
                    config=base_config,
                    stage="eval_coco",
                    seed=seed,
                    overrides=eval_overrides,
                ),
            }
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/ablation_matrix.json"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/coco_full.json"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("outputs/matched_ablations"),
    )
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = build_jobs(
        matrix=matrix,
        base_config=args.base_config,
        seed=args.seed,
        workdir=args.workdir,
    )
    args.workdir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        print("TRAIN", shlex.join(job["train"]))
        print("EVAL ", shlex.join(job["eval"]))
        if args.execute:
            subprocess.run(job["train"], cwd=REPO_ROOT, check=True)
            subprocess.run(job["eval"], cwd=REPO_ROOT, check=True)

    destination = args.workdir / "jobs.json"
    destination.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(jobs)} matched jobs to {destination}")


if __name__ == "__main__":
    main()
