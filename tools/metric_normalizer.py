"""Normalize baseline prediction files to the five-seed CSV metric scale."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict


def coco_metrics(annotation: Path, predictions: Path) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(str(predictions))
    evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "APM", "APL", "AR", "AR50", "AR75", "ARM", "ARL")
    return {name: float(value) * 100.0 for name, value in zip(names, evaluator.stats)}


def mpii_metrics(ground_truth: Path, predictions: Path) -> Dict[str, float]:
    import numpy as np
    from scipy.io import loadmat

    rows = json.loads(predictions.read_text(encoding="utf-8"))
    num_instances = max(int(row["ins_id"]) for row in rows) + 1
    preds = np.zeros((num_instances, 16, 2), dtype=np.float64)
    for row in rows:
        preds[int(row["ins_id"])] = np.asarray(row["keypoints"]).reshape(16, 3)[:, :2]
    preds = preds + 1.0

    gt = loadmat(str(ground_truth))
    dataset_joints = gt["dataset_joints"]
    joint_missing = gt["jnt_missing"]
    target = gt["pos_gt_src"]
    headboxes = gt["headboxes_src"]
    prediction = np.transpose(preds, [1, 2, 0])

    error = np.linalg.norm(prediction - target, axis=1)
    head_sizes = np.linalg.norm(headboxes[1] - headboxes[0], axis=0) * 0.6
    scaled_error = error / np.multiply(head_sizes, np.ones((len(error), 1)))
    visible = 1 - joint_missing
    scaled_error = np.multiply(scaled_error, visible)
    joint_count = np.sum(visible, axis=1)
    pckh = 100.0 * np.sum(np.multiply(scaled_error <= 0.5, visible), axis=1) / joint_count

    def index(name: str) -> int:
        return int(np.where(dataset_joints == name)[1][0])

    values = OrderedDict(
        [
            ("Head", pckh[index("head")]),
            ("Shoulder", 0.5 * (pckh[index("lsho")] + pckh[index("rsho")])),
            ("Elbow", 0.5 * (pckh[index("lelb")] + pckh[index("relb")])),
            ("Wrist", 0.5 * (pckh[index("lwri")] + pckh[index("rwri")])),
            ("Hip", 0.5 * (pckh[index("lhip")] + pckh[index("rhip")])),
            ("Knee", 0.5 * (pckh[index("lkne")] + pckh[index("rkne")])),
            ("Ankle", 0.5 * (pckh[index("lank")] + pckh[index("rank")])),
        ]
    )
    masked_count = np.ma.array(joint_count, mask=False)
    masked_pckh = np.ma.array(pckh, mask=False)
    masked_count.mask[6:10] = True
    masked_pckh.mask[6:10] = True
    ratio = masked_count / np.sum(masked_count).astype(np.float64)
    values["PCKh@0.5"] = np.sum(masked_pckh * ratio)
    return {key: float(value) for key, value in values.items()}


def normalize(
    *,
    dataset: str,
    method: str,
    seed: int,
    predictions: Path,
    output: Path,
    annotation: Path | None = None,
    ground_truth: Path | None = None,
) -> dict:
    if dataset in {"coco", "human_art"}:
        if annotation is None:
            raise ValueError("COCO-format evaluation requires --annotation")
        metrics = coco_metrics(annotation, predictions)
        primary_name = "AP"
    elif dataset == "mpii":
        if ground_truth is None:
            raise ValueError("MPII evaluation requires --ground-truth")
        metrics = mpii_metrics(ground_truth, predictions)
        primary_name = "PCKh@0.5"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    record = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "metric": primary_name,
        "value": metrics[primary_name],
        "scale": "percentage_points",
        "metrics": metrics,
        "prediction_file": str(predictions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("coco", "human_art", "mpii"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()
    result = normalize(
        dataset=args.dataset,
        method=args.method,
        seed=args.seed,
        predictions=args.predictions,
        output=args.output,
        annotation=args.annotation,
        ground_truth=args.ground_truth,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
