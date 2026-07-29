import re


COORDINATE_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
)


def parse_normalized_coordinates(text):
    """Parse the first normalized ``[x, y]`` pair from model text."""
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    search_text = (
        bracket_match.group(1)
        if bracket_match is not None
        else text
    )
    values = COORDINATE_PATTERN.findall(search_text)
    if len(values) < 2:
        return None

    x, y = float(values[0]), float(values[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


def generation_sequence_confidence(
    generation_output,
    input_length,
):
    """
    Return the geometric mean probability of generated tokens.

    Hugging Face generation scores are indexed by generation step and
    token ID, not by character offsets in the decoded string.
    """
    import torch

    step_scores = generation_output.scores
    if not step_scores:
        return torch.ones(
            generation_output.sequences.shape[0],
            device=generation_output.sequences.device,
        )
    generated_tokens = generation_output.sequences[
        :,
        input_length:input_length + len(step_scores),
    ]
    selected_log_probabilities = []
    for step, logits in enumerate(step_scores):
        log_probabilities = torch.log_softmax(
            logits.float(),
            dim=-1,
        )
        selected_log_probabilities.append(
            log_probabilities.gather(
                1,
                generated_tokens[:, step:step + 1],
            ).squeeze(1)
        )
    return torch.stack(
        selected_log_probabilities,
        dim=1,
    ).mean(dim=1).exp()
