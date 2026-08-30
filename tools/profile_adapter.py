"""Dispatch every Table 6 method to a common profiling result schema."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_profile_command(
    *,
    method_name: str,
    method: Mapping[str, object],
    dataset: Mapping[str, object],
    raw_output: Path,
    protocol: Mapping[str, object],
    seed: int,
) -> Tuple[List[str], Path, dict]:
    backend = method["backend"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(method.get("gpus", "0"))
    if backend == "native":
        raw_dir = raw_output.parent / (raw_output.stem + "_native")
        command = [
            sys.executable,
            "utils/profile_efficiency.py",
            "--model-name",
            str(_project_path(str(method["checkpoint"]))),
            "--annotation-file",
            str(_project_path(str(dataset["annotation"]))),
            "--image-folder",
            str(_project_path(str(dataset["image_folder"]))),
            "--output-dir",
            str(raw_dir),
            "--warmup",
            str(protocol["warmup"]),
            "--repetitions",
            str(protocol["repeat"]),
            "--seed",
            str(seed),
        ]
        return command, REPO_ROOT, {**environment, "ADCCR_RAW_DIR": str(raw_dir)}
    if backend == "external_llm":
        repo = _project_path(str(method["repo_dir"]))
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/profile_external_llm.py"),
            "--repo-dir",
            str(repo),
            "--model-class",
            str(method["model_class"]),
            "--checkpoint",
            str(repo / str(method["checkpoint"])),
            "--annotation-file",
            str(_project_path(str(dataset["annotation"]))),
            "--image-folder",
            str(_project_path(str(dataset["image_folder"]))),
            "--output",
            str(raw_output),
            "--warmup",
            str(protocol["warmup"]),
            "--repetitions",
            str(protocol["repeat"]),
            "--seed",
            str(seed),
        ]
        return command, REPO_ROOT, environment
    if backend == "vitpose":
        repo = _project_path(str(method["repo_dir"]))
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/profile_vitpose.py"),
            "--repo-dir",
            str(repo),
            "--config-file",
            str(repo / str(method["config_file"])),
            "--checkpoint",
            str(_project_path(str(method["checkpoint"]))),
            "--output",
            str(raw_output),
            "--warmup",
            str(protocol["warmup"]),
            "--repetitions",
            str(protocol["repeat"]),
            "--seed",
            str(seed),
        ]
        return command, REPO_ROOT, environment
    raise ValueError(f"Unsupported profiling backend: {backend}")


def _require_inputs(method_name: str, method: Mapping[str, object]) -> None:
    if method["backend"] in {"external_llm", "vitpose"}:
        repo = _project_path(str(method["repo_dir"]))
        if not repo.is_dir():
            raise FileNotFoundError(
                f"{method_name} source is missing at {repo}; run "
                f"`python scripts/fetch_baseline_code.py --method {method_name} --execute`."
            )


def _normalize(method_name: str, raw: Mapping[str, object], protocol: Mapping[str, object]) -> dict:
    latency = raw["latency_ms"]
    mean_latency = float(latency["mean"] if isinstance(latency, dict) else latency)
    peak_bytes = int(raw["peak_cuda_memory_bytes"])
    return {
        "method": method_name,
        "latency_ms": mean_latency,
        "flops": int(raw["approximate_flops"]),
        "peak_memory_gb": peak_bytes / (1024.0 ** 3),
        "parameters": raw.get("parameters", {}),
        "protocol": dict(protocol),
        "backend_protocol": raw.get("protocol", {}),
        "hardware_software": raw.get("hardware_software", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument(
        "--adapters",
        type=Path,
        default=Path("configs/method_adapters.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/efficiency.json"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the profiler; without this flag only print the resolved command.",
    )
    args = parser.parse_args()
    adapters = json.loads(_project_path(args.adapters).read_text(encoding="utf-8"))
    efficiency = json.loads(_project_path(args.protocol).read_text(encoding="utf-8"))
    if args.method not in adapters["methods"]:
        parser.error(f"unknown method: {args.method}")
    method = adapters["methods"][args.method]
    protocol = efficiency["protocol"]
    raw_output = args.profile_json.with_suffix(".raw.json").absolute()
    command, cwd, environment = build_profile_command(
        method_name=args.method,
        method=method,
        dataset=adapters["datasets"]["coco"],
        raw_output=raw_output,
        protocol=protocol,
        seed=args.seed,
    )
    print(f"cwd={cwd}")
    print(shlex.join(command))
    if not args.execute:
        return
    _require_inputs(args.method, method)
    subprocess.run(command, cwd=cwd, env=environment, check=True)
    if method["backend"] == "native":
        raw_output = Path(environment["ADCCR_RAW_DIR"]) / "efficiency.json"
    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    normalized = _normalize(args.method, raw, protocol)
    args.profile_json.parent.mkdir(parents=True, exist_ok=True)
    args.profile_json.write_text(
        json.dumps(normalized, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(normalized, indent=2))


if __name__ == "__main__":
    main()
