from collections import defaultdict

import numpy as np

from datasets.constants import COCO_KEYPOINT_NAME


def coco_per_joint_pck(
    detailed_predictions,
    threshold=0.1,
):
    """
    Compute PCK using ``threshold * max(box_width, box_height)``.

    This metric is used only for the difficult-joint ablation report;
    standard COCO results continue to use official COCO OKS AP/AR.
    """
    hits = defaultdict(int)
    totals = defaultdict(int)

    for item in detailed_predictions:
        prediction = np.asarray(
            item["keypoints"],
            dtype=np.float64,
        ).reshape(17, 3)
        ground_truth = np.asarray(
            item["gt_keypoints"],
            dtype=np.float64,
        ).reshape(17, 3)
        bbox = np.asarray(item["bbox"], dtype=np.float64)
        normalization = max(float(bbox[2]), float(bbox[3]))
        if normalization <= 0:
            continue

        for index, name in enumerate(COCO_KEYPOINT_NAME):
            if ground_truth[index, 2] <= 0:
                continue
            error = np.linalg.norm(
                prediction[index, :2]
                - ground_truth[index, :2]
            )
            hits[name] += int(
                error <= threshold * normalization
            )
            totals[name] += 1

    scores = {
        name: (
            100.0 * hits[name] / totals[name]
            if totals[name]
            else None
        )
        for name in COCO_KEYPOINT_NAME
    }

    paired_groups = {
        "wrist": ("left wrist", "right wrist"),
        "ankle": ("left ankle", "right ankle"),
        "elbow": ("left elbow", "right elbow"),
        "knee": ("left knee", "right knee"),
    }
    for group_name, names in paired_groups.items():
        group_hits = sum(hits[name] for name in names)
        group_total = sum(totals[name] for name in names)
        scores[group_name] = (
            100.0 * group_hits / group_total
            if group_total
            else None
        )

    return {
        "name": "PCK@0.1-max(person-box-width,height)",
        "threshold": threshold,
        "scores": scores,
        "counts": dict(totals),
    }
