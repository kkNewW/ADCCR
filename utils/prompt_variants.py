import hashlib
import json
import re
from pathlib import Path


PROMPT_VARIANTS = (
    "canonical",
    "paraphrase",
    "shortened",
    "corrupted",
    "alternative",
)

DESCRIPTION_VARIANTS = {
    "canonical",
    "paraphrase",
    "shortened",
    "corrupted",
}


def prompt_file_sha256(path):
    if not path:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_prompt(description, question):
    text = f"{description} {question}".lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class PromptVariantBank:
    def __init__(self, path, expected_keypoints):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("schema_version") != 1:
            raise ValueError(
                "Unsupported or missing prompt schema_version."
            )

        self.variants = tuple(payload["variants"])
        if self.variants != PROMPT_VARIANTS:
            raise ValueError(
                f"Expected variants {PROMPT_VARIANTS}, "
                f"got {self.variants}."
            )

        self.question_templates = payload[
            "question_templates"
        ]
        self.description_sources = payload[
            "description_sources"
        ]
        self.descriptions = payload["descriptions"]

        expected = set(expected_keypoints)
        actual = set(self.descriptions)
        if actual != expected:
            raise ValueError(
                "Prompt keypoints do not match the evaluation set: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}."
            )

        if set(self.question_templates) != set(PROMPT_VARIANTS):
            raise ValueError(
                "Question-template variants are incomplete."
            )
        if set(self.description_sources) != set(PROMPT_VARIANTS):
            raise ValueError(
                "Description sources are incomplete."
            )

        for variant, source in self.description_sources.items():
            if source not in DESCRIPTION_VARIANTS:
                raise ValueError(
                    f"Invalid description source for {variant}: "
                    f"{source}."
                )

        for keypoint_name, descriptions in self.descriptions.items():
            if set(descriptions) != DESCRIPTION_VARIANTS:
                raise ValueError(
                    f"Incomplete descriptions for {keypoint_name}."
                )
            for variant, text in descriptions.items():
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Empty {variant} description for "
                        f"{keypoint_name}."
                    )

        self.sha256 = prompt_file_sha256(self.path)

    def get(self, variant, keypoint_name):
        if variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt variant: {variant}."
            )

        source = self.description_sources[variant]
        description = self.descriptions[
            keypoint_name
        ][source]
        question = self.question_templates[variant].format(
            keypoint=keypoint_name
        )
        return description, question