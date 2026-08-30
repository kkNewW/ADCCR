"""Profile a pinned LocLLM/PoseLLM checkout under the ADCCR protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import statistics
import sys
from pathlib import Path


COORDINATE = re.compile(
    r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]"
)


def parameter_counts(model) -> dict:
    groups = {"vision_lora": 0, "llm_lora": 0, "connector": 0}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("vision_model.") and "lora_" in name:
            groups["vision_lora"] += count
        elif name.startswith("model.") and "lora_" in name:
            groups["llm_lora"] += count
        elif name.startswith("mm_projector."):
            groups["connector"] += count
    groups["method_trainable_total"] = sum(groups.values())
    groups["model_total"] = sum(parameter.numel() for parameter in model.parameters())
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    models = importlib.import_module("models")
    dataset_module = importlib.import_module("datasets.coco")
    validation = importlib.import_module("utils.valid2d")
    model_class = getattr(models, args.model_class)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Efficiency profiling requires a CUDA GPU.")

    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), use_fast=False, padding_side="left"
    )
    model = model_class.from_pretrained(str(checkpoint), use_cache=True).cuda().eval()
    for name, parameter in model.model.named_parameters():
        if "lora_" not in name:
            parameter.data = parameter.data.bfloat16()
    model.lm_head.to(torch.bfloat16)

    dataset = dataset_module.COCODataset(
        tokenizer=None,
        data_path=str(args.annotation_file.resolve()),
        multimodal_cfg={
            "image_folder": str(args.image_folder.resolve()),
            "image_size": 224,
            "crop_size": 224,
            "conv_format": "keypoint",
        },
        is_train=False,
    )
    sample = dataset[args.sample_index]
    collator = validation.DataCollatorForSupervisedDataset(
        image_token_len=model.config.num_patches,
        conv_format="keypoint",
    )
    _, prompts, images, has_images = collator([sample])
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tokenized.input_ids.cuda()
    attention_mask = tokenized.attention_mask.cuda()
    batch_images = torch.cat(images, dim=0).cuda()

    @torch.no_grad()
    def inference_once():
        output_ids = model.generate(
            input_ids,
            images=batch_images,
            has_images=has_images,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
        decoded = []
        for input_id, output_id in zip(input_ids, output_ids):
            text = tokenizer.decode(
                output_id[input_id.shape[0]:], skip_special_tokens=True
            )
            match = COORDINATE.search(text)
            decoded.append(None if match is None else (float(match[1]), float(match[2])))
        return decoded

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

    result = {
        "protocol": {
            "scope": (
                "one prepared 224x224 person instance, 17 keypoint queries; "
                "vision encoding, language generation, and coordinate decoding included"
            ),
            "preprocessing_in_timed_region": False,
            "batch_size_persons": 1,
            "keypoint_queries": 17,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "sample_index": args.sample_index,
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
        "parameters": parameter_counts(model),
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
