"""Download the ADCCR review checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DINO_REPOSITORY_PATH = "pretrained/dinov2_vitl14_pretrain.pth"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        default="kk618/ADCCR",
    )
    parser.add_argument(
        "--revision",
        default="main",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("checkpoints/ckpts"),
        help="Directory used to store the ADCCR checkpoint.",
    )
    parser.add_argument(
        "--pretrained-file",
        type=Path,
        default=Path(
            "checkpoints/model_weights/dinov2_vitl14_pretrain.pth"
        ),
        help="Destination of the DINOv2 initialization checkpoint.",
    )
    parser.add_argument(
        "--token",
        default=None,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download files; without this flag, only print the download plan.",
    )

    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    plan = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "checkpoint_dir": str(args.local_dir / "coco"),
        "pretrained_file": str(args.pretrained_file),
        "authenticated": bool(token),
        "source": f"https://huggingface.co/{args.repo_id}",
    }

    print(json.dumps(plan, indent=2))

    if not args.execute:
        return

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned requirements before downloading weights."
        ) from exc

    # Download only the ADCCR checkpoint files.
    # The repository's coco/ directory will be saved as:
    # checkpoints/ckpts/coco/
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=["coco/*"],
    )

    # Download DINOv2 through the Hugging Face cache.
    pretrained_source = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=DINO_REPOSITORY_PATH,
            revision=args.revision,
            token=token,
        )
    )

    # Copy DINOv2 to the location expected by config.json.
    args.pretrained_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        pretrained_source,
        args.pretrained_file,
    )

    metadata = args.local_dir / "download_source.json"
    metadata.write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Downloaded ADCCR checkpoint to "
        f"{args.local_dir / 'coco'}"
    )
    print(
        f"Installed DINOv2 weights at "
        f"{args.pretrained_file}"
    )


if __name__ == "__main__":
    main()