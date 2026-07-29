import math
import random


class DescriptionSampler:
    DEFAULT_PROBS = {
        "name_only": 0.15,
        "name_anatomy": 0.25,
        "name_relation": 0.25,
        "name_anatomy_relation": 0.20,
        "all": 0.15,
    }

    VALID_MODES = {
        "name_only",
        "name_anatomy",
        "name_relation",
        "name_anatomy_relation",
        "all",
    }

    MODE_FIELDS = {
        "name_only": (),
        "name_anatomy": (
            "anatomy",
        ),
        "name_relation": (
            "relation",
        ),
        "name_anatomy_relation": (
            "anatomy",
            "relation",
        ),
        "all": (
            "anatomy",
            "relation",
            "visual",
        ),
    }

    def __init__(
        self,
        description_bank,
        probs=None,
        strategy="default",
    ):
        self.description_bank = description_bank
        if probs is not None and strategy != "default":
            raise ValueError(
                "Provide either explicit probabilities or a "
                "named strategy, not both."
            )
        if probs is not None:
            self.probs = probs
        elif strategy == "default":
            self.probs = self.DEFAULT_PROBS.copy()
        elif strategy == "uniform":
            probability = 1.0 / len(self.VALID_MODES)
            self.probs = {
                mode: probability
                for mode in sorted(self.VALID_MODES)
            }
        else:
            raise ValueError(
                "Description sampling strategy must be "
                "'default' or 'uniform'."
            )

        unknown_modes = (
            set(self.probs)
            - self.VALID_MODES
        )
        if unknown_modes:
            raise ValueError(
                f"Unknown description modes: "
                f"{sorted(unknown_modes)}"
            )

        probability_sum = sum(
            self.probs.values()
        )
        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Description probabilities must sum "
                f"to 1.0, got {probability_sum}."
            )

    def sample_mode(self):
        modes = list(self.probs.keys())
        probabilities = [
            self.probs[mode]
            for mode in modes
        ]

        return random.choices(
            modes,
            weights=probabilities,
            k=1,
        )[0]

    def build_description(
        self,
        keypoint_name,
        mode=None,
    ):
        if keypoint_name not in self.description_bank:
            raise KeyError(
                f"Description bank does not contain "
                f"{keypoint_name!r}."
            )

        mode = mode or self.sample_mode()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid description mode: {mode}"
            )

        item = self.description_bank[
            keypoint_name
        ]

        name = random.choice(item["name"])

        sentences = [
            f"Target keypoint: {name}."
        ]

        for field_name in self.MODE_FIELDS[mode]:
            candidates = item.get(
                field_name,
                []
            )

            if candidates:
                sentences.append(
                    random.choice(candidates)
                )

        return " ".join(sentences), mode
