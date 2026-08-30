"""Clone the official baseline repositories at the pinned revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/method_adapters.json"),
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=("LocLLM", "PoseLLM", "ViTPose-L"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    spec = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    selected = args.method or ["LocLLM", "PoseLLM", "ViTPose-L"]
    for name in selected:
        item = spec["methods"][name]
        destination = REPO_ROOT / item["repo_dir"]
        commands = [
            ["git", "clone", "--no-checkout", item["repository"], str(destination)],
            ["git", "-C", str(destination), "checkout", "--detach", item["revision"]],
        ]
        for command in commands:
            print(" ".join(command))
        if not args.execute:
            continue
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing baseline directory: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
