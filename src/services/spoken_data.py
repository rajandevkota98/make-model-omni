"""Step 6 data: spoken examples (audio slots filled with Kokoro clips) mixed with typed and ASR replay.

All clips of a split are held in RAM (Kokoro train ≈ 7 GB), so any clip can be picked for any slot.
Every micro-batch comes from one source (spoken / text / asr); the mix happens across micro-batches.
"""
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.components.merge import audio_tokens
from src.services.speech import read_index, shard_paths

SOURCES = ("spoken", "text", "asr")


# ── clips ───────────────────────────────────────────────────────────────────

def load_pool(audio_dir: str, splits: list[str], max_shards: int | None = None, workers: int = 8) -> dict:
    """{"rows": index rows, "by_text": text → rows, "audio": shard key → int16 array}."""
    rows = [{**r, "audio_dir": audio_dir} for split in splits for r in read_index(audio_dir, split)]
    keys = sorted({(audio_dir, r["split"], r["shard"]) for r in rows})
    if max_shards:
        keys = keys[:max_shards]
        rows = [r for r in rows if (audio_dir, r["split"], r["shard"]) in set(keys)]
    with ThreadPoolExecutor(workers) as pool:
        audio = dict(zip(keys, pool.map(lambda k: np.load(shard_paths(*k)[0]), keys)))
    by_text: dict[str, list[dict]] = {}
    for row in rows:
        by_text.setdefault(row["text"], []).append(row)
    return {"rows": rows, "by_text": by_text, "audio": audio}


def waveform(pool: dict, row: dict) -> np.ndarray:
    data = pool["audio"][(row["audio_dir"], row["split"], row["shard"])]
    return data[row["offset"]: row["offset"] + row["n_samples"]].astype(np.float32) / 32767


def pick_clip(pool: dict, text: str, rng: random.Random, variant: str | None = None) -> dict:
    """A random clip of `text`; variant 'clean' / 'noisy' restricts to that kind (test)."""
    clips = pool["by_text"][text]
    if variant:
        clips = [c for c in clips if (c.get("noise") is None) == (variant == "clean")] or clips
    return rng.choice(clips)


# ── examples ────────────────────────────────────────────────────────────────

def fill_slots(example: dict, clips: list[dict], pool: dict, placeholder_id: int, ecfg: dict, k: int) -> dict:
    """segments + one clip per slot → input_ids with the right number of placeholders, plus waveforms."""
    waves = [waveform(pool, c) for c in clips]
    ids: list[int] = list(example["segments"][0])
    for wave, segment in zip(waves, example["segments"][1:]):
        ids += [placeholder_id] * audio_tokens(len(wave), ecfg, k) + list(segment)
    return {"input_ids": ids + example["target"], "prompt_len": len(ids), "waveforms": waves,
            "kind": example["kind"], "id": example["id"], "source": "spoken"}


def spoken_item(example: dict, pool: dict, seed: str, placeholder_id: int, ecfg: dict, k: int, variant: str | None = None) -> dict:
    rng = random.Random(f"{seed}:{example['id']}")
    clips = [pick_clip(pool, text, rng, variant) for text in example["audio_texts"]]
    return fill_slots(example, clips, pool, placeholder_id, ecfg, k)


def text_item(example: dict) -> dict:
    return {"input_ids": example["input_ids"], "prompt_len": example["prompt_len"], "waveforms": [],
            "kind": example["kind"], "id": example["id"], "source": "text"}


def asr_item(row: dict, pool: dict, prompt: dict, tokenizer, ecfg: dict, k: int) -> dict:
    wave = waveform(pool, row)
    ids = prompt["before"] + [prompt["placeholder_id"]] * audio_tokens(len(wave), ecfg, k) + prompt["after"]
    target = tokenizer(row["text"], add_special_tokens=False)["input_ids"] + [prompt["end_id"]]
    return {"input_ids": ids + target, "prompt_len": len(ids), "waveforms": [wave], "kind": "asr", "id": row["clip_id"], "source": "asr"}


# ── epoch plan + batches ────────────────────────────────────────────────────

def replay_count(n_spoken: int, fraction: float) -> int:
    """`fraction` of the final mix: n / (n_spoken + n) = fraction."""
    return round(n_spoken * fraction / (1 - fraction)) if fraction else 0


def plan_epoch(counts: dict[str, int], micro_batch: int, seed: int, epoch: int) -> list[tuple[str, list[int]]]:
    """[(source, indexes into that source's list)] — every source shuffled, micro-batches interleaved."""
    rng = random.Random(f"{seed}:{epoch}")
    plan = []
    for source in SOURCES:
        order = list(range(counts.get(source, 0)))
        rng.shuffle(order)
        plan += [(source, order[i:i + micro_batch]) for i in range(0, len(order), micro_batch)]
    rng.shuffle(plan)
    return plan


def collate_items(items: list[dict], pad_id: int, open_id: int, close_id: int) -> dict:
    """Training batch: ids / labels / attention mask / token types + all waveforms in slot order."""
    from src.components.batching import collate

    batch = collate(items, pad_id, open_id, close_id, device="cpu")
    batch["waveforms"] = [w for item in items for w in item["waveforms"]]
    batch["items"] = [{k: v for k, v in item.items() if k not in ("input_ids", "waveforms")} for item in items]
    return batch
