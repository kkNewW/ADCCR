from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[1]

resources = runpy.run_path(
    str(REPO_ROOT / "datasets" / "constants.py")
)

COCO_KEYPOINT_NAME = resources["COCO_KEYPOINT_NAME"]
DESCRIPTION_BANK = resources["DESCRIPTION_BANK"]
KeypointLocationDescription = resources[
    "KeypointLocationDescription"
]
KeypointLocationQuestion = resources[
    "KeypointLocationQuestion"
]

REQUIRED_FIELDS = {
    "name",
    "anatomy",
    "relation",
    "visual",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_resources():
    require(
        set(DESCRIPTION_BANK)
        == set(COCO_KEYPOINT_NAME),
        "The structured description bank must contain "
        "exactly the 17 COCO keypoints.",
    )

    for keypoint, fields in DESCRIPTION_BANK.items():
        require(
            set(fields) == REQUIRED_FIELDS,
            f"{keypoint}: invalid semantic fields.",
        )

        for field_name in REQUIRED_FIELDS:
            values = fields[field_name]
            require(
                isinstance(values, list) and values,
                f"{keypoint}/{field_name}: "
                "at least one entry is required.",
            )

            require(
                all(
                    isinstance(value, str)
                    and value.strip()
                    for value in values
                ),
                f"{keypoint}/{field_name}: "
                "entries must be non-empty strings.",
            )

    require(
        set(KeypointLocationDescription)
        == set(KeypointLocationQuestion),
        "Canonical descriptions and question templates "
        "must cover the same keypoints.",
    )

    require(
        len(KeypointLocationDescription) == 24,
        "Expected 24 canonical keypoint descriptions.",
    )

    require(
        all(
            len(templates) == 10
            for templates
            in KeypointLocationQuestion.values()
        ),
        "Each keypoint must provide 10 question templates.",
    )

    return {
        "structured_keypoints": len(DESCRIPTION_BANK),
        "semantic_fields": sum(
            len(fields[field_name])
            for fields in DESCRIPTION_BANK.values()
            for field_name in REQUIRED_FIELDS
        ),
        "canonical_descriptions": len(
            KeypointLocationDescription
        ),
        "question_templates": sum(
            len(templates)
            for templates
            in KeypointLocationQuestion.values()
        ),
    }


if __name__ == "__main__":
    summary = validate_resources()
    for name, value in summary.items():
        print(f"{name}: {value}")