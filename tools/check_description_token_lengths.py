import argparse
import sys
from itertools import product

from transformers import AutoTokenizer

from datasets.constants import DESCRIPTION_BANK


def get_candidates(item, field_name):
    """
    Return every candidate of one semantic field.

    [None] is used for an absent field so that itertools.product
    can still enumerate the remaining fields.
    """
    candidates = item.get(field_name, [])
    return candidates if candidates else [None]


def build_all_descriptions(description_bank):
    """
    Exhaustively construct every possible description under `all` mode.

    This follows DescriptionSampler.build_description():
        Target keypoint: {name}.
        + anatomy
        + relation
        + visual
    """
    descriptions = []

    for keypoint_name, item in description_bank.items():
        combinations = product(
            get_candidates(item, "name"),
            get_candidates(item, "anatomy"),
            get_candidates(item, "relation"),
            get_candidates(item, "visual"),
        )

        for variant_index, (
            name,
            anatomy,
            relation,
            visual,
        ) in enumerate(combinations, start=1):
            parts = [
                f"Target keypoint: {name}."
            ]

            for field_text in (
                anatomy,
                relation,
                visual,
            ):
                if field_text:
                    parts.append(field_text)

            description = " ".join(parts)

            descriptions.append(
                {
                    "keypoint": keypoint_name,
                    "variant": variant_index,
                    "description": description,
                }
            )

    return descriptions


def count_tokens(tokenizer, text):
    """
    Match the tokenization behavior used by training/evaluation.

    Do not enable truncation here, otherwise descriptions longer
    than the limit cannot be detected.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    return len(encoded["input_ids"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=(
            "./checkpoints/model_weights/"
            "vicuna-7b-v1.5"
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=False,
    )

    # Match utils/train2d.py.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    descriptions = build_all_descriptions(
        DESCRIPTION_BANK
    )

    results = []

    for item in descriptions:
        token_count = count_tokens(
            tokenizer,
            item["description"],
        )

        results.append(
            {
                **item,
                "tokens": token_count,
                "over_limit": (
                    token_count > args.max_length
                ),
            }
        )

    # Print the longest descriptions first.
    results.sort(
        key=lambda item: item["tokens"],
        reverse=True,
    )

    print(
        f"{'Keypoint':<20}"
        f"{'Variant':>10}"
        f"{'Tokens':>10}"
        f"{'Status':>12}"
    )
    print("-" * 52)

    for item in results:
        status = (
            "TRUNCATED"
            if item["over_limit"]
            else "OK"
        )

        print(
            f"{item['keypoint']:<20}"
            f"{item['variant']:>10}"
            f"{item['tokens']:>10}"
            f"{status:>12}"
        )

    maximum = max(
        item["tokens"]
        for item in results
    )

    over_limit = [
        item
        for item in results
        if item["over_limit"]
    ]

    print()
    print(
        "Tokenizer:",
        args.tokenizer_path,
    )
    print(
        "Number of all-mode variants:",
        len(results),
    )
    print(
        "Maximum token length:",
        maximum,
    )
    print(
        "Configured maximum:",
        args.max_length,
    )
    print(
        "Descriptions exceeding limit:",
        len(over_limit),
    )

    if over_limit:
        print()
        print(
            "The current max_length would truncate "
            "at least one description."
        )

        print()
        print("Longest description:")
        print(results[0]["description"])

        # Non-zero exit code makes the script usable in CI.
        sys.exit(1)

    print()
    print(
        "All all-mode descriptions fit within "
        f"max_length={args.max_length}."
    )


if __name__ == "__main__":
    main()