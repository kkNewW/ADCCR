import random


class DescriptionSampler:
    def __init__(self, description_bank, probs=None):
        self.description_bank = description_bank
        self.probs = probs or {
            "name_only": 0.15,
            "name_anatomy": 0.25,
            "name_relation": 0.25,
            "name_anatomy_relation": 0.20,
            "all": 0.15
        }

    def sample_mode(self):
        keys = list(self.probs.keys())
        vals = list(self.probs.values())
        return random.choices(keys, weights=vals, k=1)[0]

    def build_description(self, kp_name, mode=None):
        item = self.description_bank[kp_name]
        mode = mode or self.sample_mode()

        parts = []

        if mode == "name_only":
            parts.append(random.choice(item["name"]))

        elif mode == "name_anatomy":
            parts.append(random.choice(item["name"]))
            parts.append(random.choice(item["anatomy"]))

        elif mode == "name_relation":
            parts.append(random.choice(item["name"]))
            parts.append(random.choice(item["relation"]))

        elif mode == "name_anatomy_relation":
            parts.append(random.choice(item["name"]))
            parts.append(random.choice(item["anatomy"]))
            parts.append(random.choice(item["relation"]))

        elif mode == "all":
            parts.append(random.choice(item["name"]))
            parts.append(random.choice(item["anatomy"]))
            parts.append(random.choice(item["relation"]))
            if "visual" in item and len(item["visual"]) > 0:
                parts.append(random.choice(item["visual"]))

        return " ".join(parts), mode
