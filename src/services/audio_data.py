"""Audio training data: rows from shard indexes, a shard-grouped epoch plan, and batch collation.

Why shard groups: shards sit on a network Volume, and reading whole shards in order is much faster
than jumping to random clips. Each step loads `group_shards` random shards, shuffles their clips
together (sorted by length inside micro-batches to waste little padding), and the next group is
loaded in the background while the GPU works.
"""
import random
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterator

import numpy as np

from src.components.merge import IGNORE_INDEX, audio_tokens, build_example
from src.services.speech import read_index, shard_paths

ShardKey = tuple[str, str, int]


def load_rows(sources: list[dict], max_seconds: float, seed: int = 0) -> list[dict]:
    rows = []
    for source in sources:
        picked = [{**r, "audio_dir": source["audio_dir"], "source": source["name"]}
                  for split in source["splits"] for r in read_index(source["audio_dir"], split)
                  if r.get("text") and r["n_samples"] <= max_seconds * 16000]
        if source.get("limit"):
            picked = random.Random(seed).sample(picked, min(source["limit"], len(picked)))
        rows += picked
    return rows


def shard_key(row: dict) -> ShardKey:
    return row["audio_dir"], row["split"], row["shard"]


def plan_epoch(rows: list[dict], micro_batch: int, group_shards: int, seed: int, epoch: int) -> list[tuple[list[ShardKey], list[int]]]:
    """[(shards to have loaded, row indexes of one micro-batch)], in training order."""
    rng = random.Random(f"{seed}:{epoch}")
    by_shard: dict[ShardKey, list[int]] = {}
    for i, row in enumerate(rows):
        by_shard.setdefault(shard_key(row), []).append(i)
    keys = sorted(by_shard)
    rng.shuffle(keys)
    plan = []
    for g in range(0, len(keys), group_shards):
        group = keys[g:g + group_shards]
        members = sorted((i for key in group for i in by_shard[key]), key=lambda i: rows[i]["n_samples"])
        batches = [members[i:i + micro_batch] for i in range(0, len(members), micro_batch)]
        rng.shuffle(batches)
        plan += [(group, batch) for batch in batches]
    return plan


def load_shards(keys: list[ShardKey]) -> dict[ShardKey, np.ndarray]:
    return {key: np.load(shard_paths(*key)[0]) for key in keys}


def iterate_audio(rows: list[dict], plan: list[tuple[list[ShardKey], list[int]]]) -> Iterator[tuple[list[dict], list[np.ndarray]]]:
    """(rows, float32 waveforms) per micro-batch, loading the next shard group in the background."""
    groups = list(dict.fromkeys(tuple(group) for group, _ in plan))
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures: dict[tuple, Future] = {}
        for group in groups[:2]:
            futures[group] = pool.submit(load_shards, list(group))
        current, loaded = None, {}
        for group, batch in plan:
            group = tuple(group)
            if group != current:
                loaded, current = futures.pop(group).result(), group
                nxt = groups.index(group) + 2
                if nxt < len(groups):
                    futures[groups[nxt]] = pool.submit(load_shards, list(groups[nxt]))
            picked = [rows[i] for i in batch]
            yield picked, [loaded[shard_key(r)][r["offset"]: r["offset"] + r["n_samples"]].astype(np.float32) / 32767 for r in picked]


def target_ids(tokenizer, text: str, end_id: int) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"] + [end_id]


def collate(rows: list[dict], waveforms: list[np.ndarray], prompt: dict, tokenizer, cfg: dict) -> dict:
    """Right-padded training batch (input_ids, labels, attention_mask) + the raw waveforms."""
    import torch

    k = cfg["adapter"]["frame_stack"]
    examples = [build_example(prompt["before"], prompt["after"], prompt["placeholder_id"],
                              audio_tokens(len(w), cfg["encoder"], k), target_ids(tokenizer, r["text"], prompt["end_id"]))
                for r, w in zip(rows, waveforms)]
    width = max(len(e["input_ids"]) for e in examples)
    pad = tokenizer.pad_token_id

    def padded(key: str, fill: int) -> "torch.Tensor":
        return torch.tensor([e[key] + [fill] * (width - len(e[key])) for e in examples])

    return {"input_ids": padded("input_ids", pad), "labels": padded("labels", IGNORE_INDEX),
            "attention_mask": torch.tensor([[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"])) for e in examples]),
            "waveforms": waveforms, "rows": rows}


def prompt_parts(tokenizer, pcfg: dict, placeholder_id: int) -> dict:
    from src.components.merge import split_prompt

    text = tokenizer.apply_chat_template([{"role": "user", "content": f"{pcfg['placeholder']}\n{pcfg['instruction']}"}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    before, after = split_prompt(tokenizer(text, add_special_tokens=False)["input_ids"], placeholder_id)
    return {"before": before, "after": after, "placeholder_id": placeholder_id, "end_id": tokenizer.convert_tokens_to_ids("<|im_end|>")}


def steps_per_epoch(n_micro: int, grad_accum: int) -> int:
    return -(-n_micro // grad_accum)
