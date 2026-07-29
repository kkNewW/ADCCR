import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from scipy.io import loadmat
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.conversation import conv_keypoint
from models import ADCCRModel
from utils.inference import parse_normalized_coordinates
from utils.reproducibility import set_global_seed
from utils.valid2dmpii import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    MPIIDataset,
    PREFIX_IMAGE,
    disable_torch_init,
    transform_preds,
)


def load_protocol(path):
    with open(path, encoding="utf-8") as handle:
        protocol = json.load(handle)

    training_keypoints = set(protocol["training_keypoints"])
    for query in protocol["queries"]:
        is_training_keypoint = query["name"] in training_keypoints
        if query["status"] == "unseen" and is_training_keypoint:
            raise ValueError(
                f"{query['name']} is marked unseen but appears "
                "in training_keypoints."
            )
        if query["status"] == "seen" and not is_training_keypoint:
            raise ValueError(
                f"{query['name']} is marked seen but is absent "
                "from training_keypoints."
            )
    return protocol


class NovelCollator:
    def __init__(self, image_token_len, queries):
        self.image_token_len = image_token_len
        self.queries = queries

    def __call__(self, instances):
        prompts = []
        images = []
        results = []
        for instance in instances:
            image = instance["images"].unsqueeze(0)
            for query in self.queries:
                conversation = conv_keypoint.copy()
                conversation.messages = []
                conversation.append_message(
                    conversation.roles[0],
                    query["description"],
                )
                conversation.append_message(
                    conversation.roles[1],
                    (
                        f"Where is the {query['name']} of this "
                        "person in this image? Please provide "
                        "its coordinates."
                    ),
                )
                conversation.append_message(
                    conversation.roles[2],
                    None,
                )
                prompt = (
                    PREFIX_IMAGE
                    + self.image_token_len
                    * DEFAULT_IMAGE_PATCH_TOKEN
                    + "\n"
                    + conversation.get_prompt()
                )
                prompts.append(prompt)
                images.append(image)
                results.append(
                    {
                        "instance_id": instance["instance_id"],
                        "c": instance["c"],
                        "s": instance["s"],
                        **query,
                    }
                )
        return results, prompts, images


def decode_batch(
    model,
    tokenizer,
    images,
    prompts,
    max_new_tokens,
):
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tokenized.input_ids.cuda()
    attention_mask = tokenized.attention_mask.cuda()
    image_tensor = torch.cat(images, dim=0).cuda()
    has_images = [True] * len(prompts)

    generated = model.generate(
        input_ids,
        images=image_tensor,
        has_images=has_images,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
    ).sequences

    coordinates = np.zeros((len(prompts), 2), dtype=np.float32)
    valid = np.zeros(len(prompts), dtype=bool)
    for index, (input_id, output_id) in enumerate(
        zip(input_ids, generated)
    ):
        text = tokenizer.decode(
            output_id[input_id.shape[0]:],
            skip_special_tokens=True,
        ).strip()
        parsed = parse_normalized_coordinates(text)
        if parsed is None:
            print(f"Format error for query {index}: {text}")
            continue
        coordinates[index] = np.asarray(parsed) * model.config.crop_size
        valid[index] = True
    return coordinates, valid, image_tensor


@torch.no_grad()
def worker(model, tokenizer, dataset, protocol, args):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    indices = list(range(rank, len(dataset), world_size))
    subset = torch.utils.data.Subset(dataset, indices)
    queries = protocol["queries"]
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_workers,
        collate_fn=NovelCollator(
            model.config.num_patches,
            queries,
        ),
    )

    predictions = []
    for results, prompts, images in tqdm(loader):
        coordinates, valid, image_tensor = decode_batch(
            model,
            tokenizer,
            images,
            prompts,
            args.max_new_tokens,
        )

        if args.use_local_refiner:
            descriptions = [item["description"] for item in results]
            description_tokens = tokenizer(
                descriptions,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            )
            crop_sizes = torch.tensor(
                [item["crop_size"] for item in results],
                device=image_tensor.device,
                dtype=torch.float32,
            )
            crop_sizes *= getattr(
                model.config,
                "refiner_crop_scale",
                1.0,
            )
            refined, _ = model.refine_coordinates(
                images=image_tensor,
                coarse_xy=torch.tensor(
                    coordinates,
                    device=image_tensor.device,
                ),
                crop_sizes=crop_sizes,
                desc_input_ids=(
                    description_tokens.input_ids.cuda()
                ),
                desc_attention_mask=(
                    description_tokens.attention_mask.cuda()
                ),
            )
            coordinates = refined.float().cpu().numpy()

        coordinates = transform_preds(
            coordinates,
            results[0]["c"],
            results[0]["s"],
            (model.config.crop_size, model.config.crop_size),
        )
        predictions.append(
            {
                "instance_id": int(results[0]["instance_id"]),
                "coordinates": coordinates.tolist(),
                "valid": valid.tolist(),
            }
        )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, predictions)
    if rank != 0:
        return

    merged = [
        item
        for rank_predictions in gathered
        for item in rank_predictions
    ]
    merged.sort(key=lambda item: item["instance_id"])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "predictions.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(merged, handle, indent=2)

    metrics = evaluate_predictions(
        merged,
        protocol,
        args.ground_truth,
    )
    metrics["protocol"]["seed"] = args.seed
    metrics["protocol"]["local_refiner"] = (
        args.use_local_refiner
    )
    with open(
        os.path.join(args.output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


def _joint_index(dataset_joints, name):
    matches = np.where(dataset_joints == name)[1]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely resolve MPII joint {name}.")
    return int(matches[0])


def _target_from_spec(spec, dataset_joints, positions, visible):
    if isinstance(spec, str):
        index = _joint_index(dataset_joints, spec)
        return positions[index], visible[index]

    if spec.get("operation") != "midpoint":
        raise ValueError(f"Unsupported ground-truth spec: {spec}")
    indices = [
        _joint_index(dataset_joints, name)
        for name in spec["joints"]
    ]
    midpoint = positions[indices].mean(axis=0)
    midpoint_visible = visible[indices].all(axis=0)
    return midpoint, midpoint_visible


def evaluate_predictions(predictions, protocol, ground_truth_file):
    ground_truth = loadmat(ground_truth_file)
    dataset_joints = ground_truth["dataset_joints"]
    positions = ground_truth["pos_gt_src"]
    visible = 1 - ground_truth["jnt_missing"]
    headboxes = ground_truth["headboxes_src"]
    head_sizes = np.linalg.norm(
        headboxes[1] - headboxes[0],
        axis=0,
    )
    metric = protocol["metric"]
    head_sizes *= metric["head_size_bias"]

    query_scores = []
    query_counts = []
    groups = defaultdict(list)
    for query_index, query in enumerate(protocol["queries"]):
        target, target_visible = _target_from_spec(
            query["ground_truth"],
            dataset_joints,
            positions,
            visible,
        )
        predicted = np.asarray(
            [
                item["coordinates"][query_index]
                for item in predictions
            ],
            dtype=np.float64,
        ).T
        prediction_valid = np.asarray(
            [
                item["valid"][query_index]
                for item in predictions
            ],
            dtype=bool,
        )
        errors = np.linalg.norm(predicted + 1.0 - target, axis=0)
        normalized = errors / head_sizes
        valid_mask = target_visible.astype(bool) & prediction_valid
        count = int(valid_mask.sum())
        score = (
            100.0
            * np.logical_and(
                normalized <= metric["threshold"],
                valid_mask,
            ).sum()
            / count
            if count
            else None
        )
        query_scores.append(score)
        query_counts.append(count)
        groups[query["group"]].append(query_index)

    grouped_scores = {}
    for group, indices in groups.items():
        hits_equivalent = sum(
            (query_scores[index] or 0.0)
            * query_counts[index]
            / 100.0
            for index in indices
        )
        count = sum(query_counts[index] for index in indices)
        grouped_scores[group] = (
            100.0 * hits_equivalent / count
            if count
            else None
        )

    return {
        "metric": metric,
        "scores": grouped_scores,
        "per_query_scores": {
            query["name"]: query_scores[index]
            for index, query in enumerate(protocol["queries"])
        },
        "sample_counts": {
            query["name"]: query_counts[index]
            for index, query in enumerate(protocol["queries"])
        },
        "protocol": {
            "dataset": protocol["dataset"],
            "ground_truth": ground_truth_file,
            "unseen_training_check": "passed",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-local-refiner", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
    )
    parser.add_argument("--dataloader-workers", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    set_global_seed(args.seed)
    disable_torch_init()

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(args.model_name),
        use_fast=False,
        padding_side="left",
    )
    model = ADCCRModel.from_pretrained(
        os.path.expanduser(args.model_name),
        use_cache=True,
    )
    for name, parameter in model.model.named_parameters():
        if "lora_" not in name:
            parameter.data = parameter.data.bfloat16()
    model.lm_head.to(torch.bfloat16)
    model = model.cuda().eval()

    protocol = load_protocol(args.protocol)
    dataset = MPIIDataset(args.question_file, args.image_folder)
    worker(model, tokenizer, dataset, protocol, args)


if __name__ == "__main__":
    main()
