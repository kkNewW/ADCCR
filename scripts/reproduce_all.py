"""Validate and optionally execute every manuscript-table command."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {str(index) for index in range(1, 17)}


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction-manifest schema.")
    return manifest


def validate_manifest(manifest: dict, root: Path = REPO_ROOT) -> None:
    tables = manifest.get("tables", {})
    table_ids = set(tables)
    if table_ids != EXPECTED_TABLES:
        missing = sorted(EXPECTED_TABLES - table_ids, key=int)
        extra = sorted(table_ids - EXPECTED_TABLES)
        raise ValueError(
            f"Table manifest mismatch; missing={missing}, extra={extra}"
        )

    for table_id, spec in tables.items():
        if not str(spec.get("command", "")).strip():
            raise ValueError(f"Table {table_id} has no runnable command.")
        for field in ("scripts", "configs"):
            paths = spec.get(field, [])
            if not paths:
                raise ValueError(f"Table {table_id} has no {field} mapping.")
            for relative in paths:
                if not (root / relative).is_file():
                    raise FileNotFoundError(
                        f"Table {table_id} references missing {field[:-1]}: "
                        f"{relative}"
                    )

    if not manifest.get("reproduction_steps"):
        raise ValueError("No complete reproduction steps were configured.")


def resolve_command(command: str) -> list[str]:
    resolved = shlex.split(command)
    if resolved and resolved[0] == "python":
        resolved[0] = sys.executable
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/reproduction_manifest.json"),
    )
    args = parser.parse_args()
    manifest_path = (
        args.manifest
        if args.manifest.is_absolute()
        else REPO_ROOT / args.manifest
    )
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)

    for step in manifest["reproduction_steps"]:
        command = resolve_command(step["command"])
        print(f"{step['id']}: {shlex.join(command)}")
        if args.execute:
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    if not args.execute:
        print("Manifest validated. Re-run with --execute to launch all tables.")


if __name__ == "__main__":
    main()
