"""Stage 7b · speech_plan: decide every clip to synthesise (text, voice, speed, noise...) before any GPU runs.

One row per clip. The plan is pure and seeded, so the same data + config always gives the same clips;
Modal only executes it (apps/synth.py). Which voices go where:
  train  each unique user text → `train_per_text` distinct train voices (frequent short answers → all)
  dev    each text → `dev_per_text` train voice
  test   each text → one held-out test voice, planned twice: clean and noisy
"""
import hashlib
import random
from collections import Counter
from pathlib import Path

from src.services.data.augment import augmented_path
from src.services.data.split import SPLITS
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, write_jsonl


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def user_texts(conversations: list[dict]) -> Counter:
    """Every user turn's text → how many times it appears in the split."""
    return Counter(m["content"] for item in conversations for m in item["messages"] if m["role"] == "user")


def uniform(rng: random.Random, bounds: list[float], digits: int = 2) -> float:
    return round(rng.uniform(*bounds), digits)


def clip_params(rng: random.Random, acfg: dict, noisy: bool | None) -> dict:
    """Speed, gain, padding and noise for one clip. noisy=None → random (clean_prob), else forced."""
    noisy = rng.random() >= acfg["clean_prob"] if noisy is None else noisy
    return {"speed": uniform(rng, acfg["speed"]), "gain_db": uniform(rng, acfg["gain_db"], 1),
            "pad_ms": [rng.randint(*acfg["pad_ms"]), rng.randint(*acfg["pad_ms"])],
            "noise": rng.choice(acfg["noise_types"]) if noisy else None,
            "snr_db": uniform(rng, acfg["snr_db"], 1) if noisy else None,
            "seed": rng.randrange(2 ** 31)}


def voices_for(split: str, count: int, rng: random.Random, scfg: dict) -> list[tuple[str, bool | None]]:
    """[(voice, noisy)] for one text. noisy None = random."""
    c, v = scfg["clips"], scfg["voices"]
    if split == "test":
        voice = rng.choice(v["test"])
        return [(voice, False), (voice, True)][: 2 * c["test_per_text"]]
    if split == "dev":
        return [(voice, None) for voice in rng.sample(v["train"], c["dev_per_text"])]
    k = len(v["train"]) if count >= c["frequent_min_count"] else c["train_per_text"]
    return [(voice, None) for voice in rng.sample(v["train"], k)]


def plan_split(split: str, texts: Counter, scfg: dict) -> list[dict]:
    rng = random.Random(f"{scfg['augment']['seed']}:{split}")
    rows = []
    for text in sorted(texts):  # sorted → stable order regardless of file order
        for voice, noisy in voices_for(split, texts[text], rng, scfg):
            params = clip_params(rng, scfg["augment"], noisy)
            variant = "n" if params["noise"] else "c"
            rows.append({"clip_id": f"{split}-{text_key(text)}-{voice}-{variant}", "split": split, "text": text,
                         "text_key": text_key(text), "voice": voice, **params})
    rng.shuffle(rows)  # mix voices and noise types inside every shard (babble needs other clips nearby)
    size = scfg["shards"]["size"]
    return [{**row, "shard": i // size} for i, row in enumerate(rows)]


def check_voices(scfg: dict) -> None:
    v = scfg["voices"]
    if set(v["train"]) & set(v["test"]):
        raise ValueError("test voices must be held out from train voices")


# ── stage ───────────────────────────────────────────────────────────────────

def plan_path(scfg: dict, split: str) -> Path:
    return Path(scfg["paths"]["plan_dir"]) / f"{split}.jsonl"


def inputs(cfg: dict) -> list[Path]:
    return [augmented_path(cfg, name) for name in SPLITS] + [Path(cfg["speech_config"])]


def outputs(cfg: dict) -> list[Path]:
    scfg = load_yaml(cfg["speech_config"])
    return [plan_path(scfg, name) for name in SPLITS]


def run(cfg: dict) -> dict:
    scfg = load_yaml(cfg["speech_config"])
    check_voices(scfg)
    stats, rows_in = {}, 0
    for name in SPLITS:
        texts = user_texts(read_jsonl(augmented_path(cfg, name)))
        rows = plan_split(name, texts, scfg)
        write_jsonl(plan_path(scfg, name), rows)
        rows_in += len(texts)
        stats[name] = {"texts": len(texts), "clips": len(rows), "shards": rows[-1]["shard"] + 1 if rows else 0,
                       "voices": dict(sorted(Counter(r["voice"] for r in rows).items())),
                       "noise": dict(sorted(Counter(r["noise"] or "clean" for r in rows).items()))}
    return {"in": rows_in, "out": sum(s["clips"] for s in stats.values()), "rejected": 0, "splits": stats}
