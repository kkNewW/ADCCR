import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported or missing config schema_version.")
    return config


def parse_value(raw_value):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def apply_override(config, expression):
    if "=" not in expression:
        raise ValueError(
            f"Override must use dotted.path=value: {expression}"
        )
    dotted_path, raw_value = expression.split("=", 1)
    keys = dotted_path.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise KeyError(f"Unknown override path: {dotted_path}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise KeyError(f"Unknown override path: {dotted_path}")
    cursor[keys[-1]] = parse_value(raw_value)


def format_placeholders(value, seed):
    if isinstance(value, str):
        return value.format(seed=seed)
    if isinstance(value, dict):
        return {
            key: format_placeholders(item, seed)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            format_placeholders(item, seed)
            for item in value
        ]
    return value


def cli_name(key):
    return "--" + key.replace("_", "-")


def build_command(config, stage_name, seed=None):
    if stage_name not in config["stages"]:
        raise KeyError(f"Unknown stage: {stage_name}")
    stage = copy.deepcopy(config["stages"][stage_name])
    runtime = config["runtime"]
    default_seed = config.get(
        "seed_protocol", {}
    ).get("single_run_seed", 1)
    seed = int(
        seed
        if seed is not None
        else stage["args"].get("seed", default_seed)
    )
    stage = format_placeholders(stage, seed)
    if "seed" in stage["args"]:
        stage["args"]["seed"] = seed
    if "data_seed" in stage["args"]:
        stage["args"]["data_seed"] = seed

    if stage["launcher"] == "torchrun":
        launcher_path = shutil.which("torchrun") or "torchrun"
        command = [
            launcher_path,
            "--nnodes=1",
            f"--nproc_per_node={runtime['nproc_per_node']}",
            f"--master_port={runtime['master_port']}",
            stage["entrypoint"],
        ]
    elif stage["launcher"] == "python":
        command = [
            os.environ.get("PYTHON", "python"),
            stage["entrypoint"],
        ]
    else:
        raise ValueError(
            f"Unsupported launcher: {stage['launcher']}"
        )

    hf_boolean_style = stage_name == "train"
    for key, value in stage["args"].items():
        if value is None:
            continue
        flag = cli_name(key)
        if isinstance(value, bool):
            if hf_boolean_style:
                command.extend([flag, str(value)])
            elif value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command, stage, seed


def validate_inputs(stage):
    path_keys = (
        "model_name_or_path",
        "llama_path",
        "dino_path",
        "data_path",
        "image_folder",
        "model_name",
        "question_file",
        "ground_truth",
        "protocol",
        "annotation_file",
        "prompt_variant_file",
    )
    missing = []
    for key in path_keys:
        value = stage["args"].get(key)
        if value and not (REPO_ROOT / value).exists():
            missing.append(value)
    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n- " + "\n- ".join(missing)
        )


def save_resolved_config(config, stage_name, stage, seed):
    output_dir = stage["args"].get("output_dir")
    if not output_dir:
        return
    destination = REPO_ROOT / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "experiment": config["experiment"],
        "stage": stage_name,
        "seed": seed,
        "runtime": config["runtime"],
        "resolved_stage": stage,
    }
    with open(
        destination / "resolved_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(snapshot, handle, indent=2)


def run_stage(
    config,
    stage_name,
    seed=None,
    dry_run=False,
    check_paths=True,
):
    command, stage, resolved_seed = build_command(
        config,
        stage_name,
        seed=seed,
    )
    print(shlex.join(command))
    if dry_run:
        return 0
    if stage["launcher"] == "torchrun" and shutil.which(
        "torchrun"
    ) is None:
        raise RuntimeError("torchrun is not available.")
    if check_paths:
        validate_inputs(stage)
    save_resolved_config(
        config,
        stage_name,
        stage,
        resolved_seed,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = config["runtime"][
        "cuda_visible_devices"
    ]
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONHASHSEED"] = str(resolved_seed)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    ).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/coco_full.json",
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-path-check",
        action="store_true",
    )
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    for expression in args.overrides:
        apply_override(config, expression)
    run_stage(
        config,
        args.stage,
        seed=args.seed,
        dry_run=args.dry_run,
        check_paths=not args.skip_path_check,
    )


if __name__ == "__main__":
    main()
