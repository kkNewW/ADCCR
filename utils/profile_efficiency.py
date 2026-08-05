import argparse
import json
import os
import platform
import statistics

import numpy as np
import torch
from transformers import AutoTokenizer

from datasets.coco import COCODataset
from datasets.constants import (
    COCO_KEYPOINT_NAME,
    CROP_SIZE_MAP,
)
from models import ADCCRModel
from utils.inference import parse_normalized_coordinates
from utils.reproducibility import set_global_seed
from utils.valid2d import DataCollatorForSupervisedDataset


def component_parameter_counts(model):
    groups = {
        "vision_lora": 0,
        "llm_lora": 0,
        "connector": 0,
        "description_projection": 0,
        "local_refiner": 0,
    }
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("vision_model.") and "lora_" in name:
            groups["vision_lora"] += count
        elif name.startswith("model.") and "lora_" in name:
            groups["llm_lora"] += count
        elif name.startswith("mm_projector."):
            groups["connector"] += count
        elif name.startswith("description_projection."):
            groups["description_projection"] += count
        elif name.startswith("local_refiner."):
            groups["local_refiner"] += count
    groups["method_trainable_total"] = sum(groups.values())
    groups["model_total"] = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    return groups


def prepare_person_batch(model, tokenizer, args):
    dataset = COCODataset(
        tokenizer=None,
        data_path=args.annotation_file,
        multimodal_cfg={
            "image_folder": args.image_folder,
            "image_size": 224,
            "crop_size": 224,
            "conv_format": "keypoint",
        },
        is_train=False,
    )
    sample = dataset[args.sample_index]
    collator = DataCollatorForSupervisedDataset(
        image_token_len=model.config.num_patches,
        conv_format="keypoint",
        use_dynamic_desc=True,
        eval_desc_mode="all",
    )
    results, prompts, images, has_images = collator([sample])
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    batch_images = torch.cat(images, dim=0).cuda()
    descriptions = [result["description"] for result in results]
    description_tokens = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    crop_sizes = torch.tensor(
        [
            CROP_SIZE_MAP[result["kpt_name"]]
            for result in results
        ],
        dtype=torch.float32,
        device=batch_images.device,
    )
    crop_sizes *= getattr(
        model.config,
        "refiner_crop_scale",
        1.0,
    )
    return {
        "input_ids": tokenized.input_ids.cuda(),
        "attention_mask": tokenized.attention_mask.cuda(),
        "images": batch_images,
        "has_images": has_images,
        "description_ids": description_tokens.input_ids.cuda(),
        "description_mask": (
            description_tokens.attention_mask.cuda()
        ),
        "crop_sizes": crop_sizes,
        "sample": {
            "index": args.sample_index,
            "image_id": int(sample["image_id"]),
            "annotation_id": int(sample["annotation_id"]),
        },
    }


@torch.no_grad()
def inference_once(model, tokenizer, batch, max_new_tokens):
    generated = model.generate(
        batch["input_ids"],
        images=batch["images"],
        has_images=batch["has_images"],
        attention_mask=batch["attention_mask"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
    ).sequences

    coarse = np.zeros((len(generated), 2), dtype=np.float32)
    for index, (input_id, output_id) in enumerate(
        zip(batch["input_ids"], generated)
    ):
        text = tokenizer.decode(
            output_id[input_id.shape[0]:],
            skip_special_tokens=True,
        )
        parsed = parse_normalized_coordinates(text)
        if parsed is not None:
            coarse[index] = (
                np.asarray(parsed) * model.config.crop_size
            )

    if model.config.use_local_refiner:
        model.refine_coordinates(
            images=batch["images"],
            coarse_xy=torch.tensor(
                coarse,
                device=batch["images"].device,
            ),
            crop_sizes=batch["crop_sizes"],
            desc_input_ids=batch["description_ids"],
            desc_attention_mask=batch["description_mask"],
        )


def measure_latency(model, tokenizer, batch, args):
    for _ in range(args.warmup):
        inference_once(
            model,
            tokenizer,
            batch,
            args.max_new_tokens,
        )
    torch.cuda.synchronize()

    measurements = []
    for _ in range(args.repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        inference_once(
            model,
            tokenizer,
            batch,
            args.max_new_tokens,
        )
        end.record()
        torch.cuda.synchronize()
        measurements.append(float(start.elapsed_time(end)))
    return measurements


def measure_flops(model, tokenizer, batch, args):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
        profile_memory=True,
    ) as profiler:
        inference_once(
            model,
            tokenizer,
            batch,
            args.max_new_tokens,
        )
    return int(
        sum(
            event.flops or 0
            for event in profiler.key_averages()
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Efficiency profiling requires a CUDA GPU.")
    set_global_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        padding_side="left",
    )
    model = ADCCRModel.from_pretrained(
        args.model_name,
        use_cache=True,
    ).cuda().eval()
    for name, parameter in model.model.named_parameters():
        if "lora_" not in name:
            parameter.data = parameter.data.bfloat16()
    model.lm_head.to(torch.bfloat16)

    batch = prepare_person_batch(model, tokenizer, args)
    torch.cuda.reset_peak_memory_stats()
    measurements = measure_latency(
        model,
        tokenizer,
        batch,
        args,
    )
    peak_memory = torch.cuda.max_memory_allocated()
    approximate_flops = measure_flops(
        model,
        tokenizer,
        batch,
        args,
    )

    output = {
        "protocol": {
            "scope": (
                "one person instance with 17 keypoint queries; "
                "vision encoding, language decoding and local "
                "refinement included; disk I/O and person-crop "
                "preprocessing excluded"
            ),
            "precision": "bfloat16 for frozen LLM weights",
            "batch_size_persons": 1,
            "keypoint_queries": 17,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "max_new_tokens": args.max_new_tokens,
            "sample": batch["sample"],
            "seed": args.seed,
            "flops_tool": "torch.profiler(with_flops=True)",
            "flops_note": (
                "Approximate: only operators for which PyTorch "
                "registers a FLOP formula are counted."
            ),
        },
        "latency_ms": {
            "mean": statistics.fmean(measurements),
            "median": statistics.median(measurements),
            "min": min(measurements),
            "max": max(measurements),
        },
        "peak_cuda_memory_bytes": peak_memory,
        "approximate_flops": approximate_flops,
        "parameters": component_parameter_counts(model),
        "hardware_software": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "efficiency.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
