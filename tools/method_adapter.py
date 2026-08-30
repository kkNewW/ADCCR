"""Unified train/evaluate adapter for ADCCR, LocLLM, and PoseLLM.

ADCCR is routed through the repository's executable configuration runner.
LocLLM and PoseLLM are routed to pinned official source trees using their
native ``utils/train2d.py`` and validation entrypoints.  Evaluation outputs are
normalized to one ``primary_metrics.json`` schema, which lets the five-seed
runner build one CSV without method-specific parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_adapters(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported method-adapter schema.")
    return spec


def _absolute_from_project(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _absolute_from_repo(value: str | Path, repo: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _native_command(
    action: str,
    method: Mapping[str, object],
    dataset: Mapping[str, object] | None,
    *,
    project_config: Path,
    seed: int,
    checkpoint: Path,
    output: Path,
) -> List[str]:
    if action == "train":
        stage = "train"
        overrides = {
            "stages.train.args.output_dir": output.as_posix(),
        }
    else:
        assert dataset is not None
        stage = str(dataset["stage"])
        overrides = {
            f"stages.{stage}.args.model_name": checkpoint.as_posix(),
            f"stages.{stage}.args.output_dir": output.as_posix(),
        }
    command = [
        sys.executable,
        "utils/run_config.py",
        "--config",
        project_config.as_posix(),
        "--stage",
        stage,
        "--seed",
        str(seed),
    ]
    for key, value in sorted(overrides.items()):
        command.extend(["--set", f"{key}={json.dumps(value)}"])
    return command


def _external_value(key: str, value: object, repo: Path) -> object:
    project_paths = {"data_path", "image_folder"}
    repo_paths = {"model_name_or_path", "llama_path", "dino_path"}
    if key in project_paths:
        return _absolute_from_project(str(value))
    if key in repo_paths:
        return _absolute_from_repo(str(value), repo)
    return value


def _append_argument(command: List[str], key: str, value: object) -> None:
    flag = "--" + key
    if isinstance(value, bool):
        command.extend([flag, str(value)])
    elif value is not None:
        command.extend([flag, str(value)])


def _external_command(
    action: str,
    method: Mapping[str, object],
    dataset: Mapping[str, object] | None,
    *,
    seed: int,
    checkpoint: Path,
    output: Path,
) -> Tuple[List[str], Path, dict]:
    repo = _absolute_from_project(str(method["repo_dir"]))
    torchrun = shutil.which("torchrun") or "torchrun"
    if action == "eval" and dataset is not None:
        eval_processes = method.get("eval_nproc_per_node", {})
        nproc_per_node = eval_processes.get(
            str(dataset.get("name")), method["nproc_per_node"]
        )
    else:
        nproc_per_node = method["nproc_per_node"]
    command = [
        torchrun,
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={method['master_port']}",
    ]
    if action == "train":
        command.append("utils/train2d.py")
        for key, raw_value in method["train_args"].items():
            _append_argument(
                command,
                key,
                _external_value(key, raw_value, repo),
            )
        _append_argument(command, "output_dir", output)
        _append_argument(command, "seed", seed)
        _append_argument(command, "data_seed", seed)
    else:
        assert dataset is not None
        command.append(str(dataset["entrypoint"]))
        _append_argument(command, "model-name", checkpoint)
        _append_argument(
            command,
            "question-file",
            _absolute_from_project(str(dataset["annotation"])),
        )
        _append_argument(
            command,
            "image-folder",
            _absolute_from_project(str(dataset["image_folder"])),
        )
        _append_argument(command, "output-dir", output)
        _append_argument(command, "conv-format", "keypoint")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(method["gpus"])
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = str(repo)
    environment["PYTHONHASHSEED"] = str(seed)
    return command, repo, environment


def build_command(
    *,
    action: str,
    method_name: str,
    dataset_name: str | None,
    adapters: Mapping[str, object],
    project_config: Path,
    seed: int,
    checkpoint: Path,
    output: Path,
) -> Tuple[List[str], Path, dict]:
    method = adapters["methods"][method_name]
    dataset = None if dataset_name is None else dict(adapters["datasets"][dataset_name])
    if dataset is not None:
        dataset["name"] = dataset_name
    if method["backend"] == "native":
        command = _native_command(
            action,
            method,
            dataset,
            project_config=project_config,
            seed=seed,
            checkpoint=checkpoint,
            output=output,
        )
        return command, REPO_ROOT, os.environ.copy()
    if method["backend"] == "external_llm":
        return _external_command(
            action,
            method,
            dataset,
            seed=seed,
            checkpoint=checkpoint,
            output=output,
        )
    raise ValueError(f"{method_name} does not support {action} through this adapter.")


def _validate_external_repo(method_name: str, method: Mapping[str, object], repo: Path) -> None:
    if not (repo / ".git").is_dir():
        raise FileNotFoundError(
            f"{method_name} source tree is missing at {repo}. Run "
            "`python scripts/fetch_baseline_code.py --method "
            f"{method_name}` first."
        )
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != method["revision"]:
        raise RuntimeError(
            f"{method_name} is at {current}, expected pinned revision "
            f"{method['revision']}."
        )


def _normalize_command(
    *,
    dataset_name: str,
    method_name: str,
    seed: int,
    dataset: Mapping[str, object],
    output: Path,
) -> List[str]:
    command = [
        sys.executable,
        "tools/metric_normalizer.py",
        "--dataset",
        dataset_name,
        "--method",
        method_name,
        "--seed",
        str(seed),
        "--predictions",
        str(output / "pred_kpt.json"),
        "--output",
        str(output / "primary_metrics.json"),
    ]
    if dataset_name in {"coco", "human_art"}:
        command.extend(
            ["--annotation", str(_absolute_from_project(str(dataset["annotation"])))]
        )
    else:
        command.extend(
            ["--ground-truth", str(_absolute_from_project(str(dataset["ground_truth"])))]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "eval"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", choices=("coco", "human_art", "mpii"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
    args = parser.parse_args()

    if args.action == "eval" and args.dataset is None:
        parser.error("eval requires --dataset")
    spec = load_adapters(_absolute_from_project(args.adapters))
    if args.method not in spec["methods"]:
        parser.error(f"unknown method: {args.method}")
    method = spec["methods"][args.method]
    checkpoint = args.checkpoint
    if checkpoint is None:
        if method["backend"] == "native":
            checkpoint = _absolute_from_project(str(method["checkpoint"]))
        else:
            repo = _absolute_from_project(str(method["repo_dir"]))
            checkpoint = _absolute_from_repo(str(method["checkpoint"]), repo)
    else:
        checkpoint = checkpoint.absolute()
    output = args.output.absolute()
    project_config = args.config
    if project_config.is_absolute():
        project_config = project_config.relative_to(REPO_ROOT)

    command, cwd, environment = build_command(
        action=args.action,
        method_name=args.method,
        dataset_name=args.dataset,
        adapters=spec,
        project_config=project_config,
        seed=args.seed,
        checkpoint=checkpoint,
        output=output,
    )
    print(f"cwd={cwd}")
    print(shlex.join(command))
    if not args.execute:
        return

    if method["backend"] == "external_llm":
        _validate_external_repo(args.method, method, cwd)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)
    if args.action == "eval":
        dataset = spec["datasets"][args.dataset]
        normalize = _normalize_command(
            dataset_name=args.dataset,
            method_name=args.method,
            seed=args.seed,
            dataset=dataset,
            output=output,
        )
        print(shlex.join(normalize))
        subprocess.run(normalize, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
