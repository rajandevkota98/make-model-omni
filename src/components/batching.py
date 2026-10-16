"""Examples → padded tensors, with a per-token type so the loss can be split into tool-call vs reply."""
import random

from src.services.data.format import IGNORE_INDEX, labels_for

PROMPT, REPLY, TOOL_CALL = 0, 1, 2


def token_types(input_ids: list[int], prompt_len: int, open_id: int, close_id: int) -> list[int]:
    """0 = prompt (ignored), 1 = spoken reply / end token, 2 = inside <tool_call> … </tool_call> (tags included)."""
    types, inside = [PROMPT] * prompt_len, False
    for token in input_ids[prompt_len:]:
        if token == open_id:
            inside = True
        types.append(TOOL_CALL if inside else REPLY)
        if token == close_id:
            inside = False
    return types


def epoch_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def collate(examples: list[dict], pad_id: int, open_id: int, close_id: int, device: str = "cuda") -> dict:
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    pad = lambda seq, value: seq + [value] * (width - len(seq))
    rows = {
        "input_ids": [pad(e["input_ids"], pad_id) for e in examples],
        "attention_mask": [pad([1] * len(e["input_ids"]), 0) for e in examples],
        "labels": [pad(labels_for(e), IGNORE_INDEX) for e in examples],
        "types": [pad(token_types(e["input_ids"], e["prompt_len"], open_id, close_id), PROMPT) for e in examples],
    }
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in rows.items()}
