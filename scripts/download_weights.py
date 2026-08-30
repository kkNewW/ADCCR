"""Download the ADCCR review checkpoint from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="kk618/ADCCR")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("checkpoints/ckpts"),
    )
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download files; without this flag only print the download plan.",
    )
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    plan = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(args.local_dir),
        "authenticated": bool(token),
        "source": f"https://huggingface.co/{args.repo_id}",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned requirements before downloading weights."
        ) from exc

    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    metadata = args.local_dir / "download_source.json"
    metadata.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {args.repo_id} to {args.local_dir}")


if __name__ == "__main__":
    main()
