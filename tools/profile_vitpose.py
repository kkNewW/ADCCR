"""Profile ViTPose-L model forward under the person-instance protocol."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmpose.models import build_posenet

    if not torch.cuda.is_available():
        raise RuntimeError("Efficiency profiling requires a CUDA GPU.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = Config.fromfile(str(args.config_file.resolve()))
    model = build_posenet(config.model)
    load_checkpoint(model, str(args.checkpoint.resolve()), map_location="cpu")
    model = model.cuda().eval()
    image_size = config.data_cfg.get("image_size", [192, 256])
    width, height = int(image_size[0]), int(image_size[1])
    image = torch.randn(1, 3, height, width, device="cuda")

    @torch.no_grad()
    def inference_once():
        if not hasattr(model, "forward_dummy"):
            raise RuntimeError(
                "The pinned ViTPose model lacks forward_dummy; this is required "
                "to exclude preprocessing from the timed region."
            )
        output = model.forward_dummy(image)
        heatmaps = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(heatmaps, dict):
            heatmaps = next(iter(heatmaps.values()))
        # Include the deterministic heatmap-to-coordinate argmax in the timed
        # region so the scope matches the coordinate decoding used by LLMs.
        flat = heatmaps.flatten(start_dim=-2)
        indices = flat.argmax(dim=-1)
        return torch.stack(
            (indices.remainder(heatmaps.shape[-1]), indices.div(heatmaps.shape[-1], rounding_mode="floor")),
            dim=-1,
        )

    for _ in range(args.warmup):
        inference_once()
    torch.cuda.synchronize()
    measurements = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        inference_once()
        end.record()
        torch.cuda.synchronize()
        measurements.append(float(start.elapsed_time(end)))
    peak_memory = torch.cuda.max_memory_allocated()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        profile_memory=True,
    ) as profiler:
        inference_once()
    approximate_flops = int(
        sum(event.flops or 0 for event in profiler.key_averages())
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result = {
        "protocol": {
            "scope": (
                "one prepared person crop; one forward pass outputs 17 joints; "
                "model forward and heatmap decoding included"
            ),
            "preprocessing_in_timed_region": False,
            "batch_size_persons": 1,
            "joint_predictions": 17,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "input_size": [width, height],
            "seed": args.seed,
            "flops_tool": "torch.profiler(with_flops=True)",
        },
        "latency_ms": {
            "mean": statistics.fmean(measurements),
            "median": statistics.median(measurements),
            "min": min(measurements),
            "max": max(measurements),
        },
        "peak_cuda_memory_bytes": int(peak_memory),
        "approximate_flops": approximate_flops,
        "parameters": {
            "method_trainable_total": trainable_parameters,
            "model_total": total_parameters,
        },
        "hardware_software": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
