"""Turn planned clips (speech_plan rows) into 16 kHz int16 audio shards.

Per clip:  Kokoro(text, voice, speed) → resample 24k→16k → pad silence → add noise at SNR → gain → limit peak
Per shard: one `shard-00012.npy` (all clips back to back, int16) + `shard-00012.jsonl` (where each clip starts).
Loading one clip later is a slice: audio[offset : offset + n_samples].
"""
from pathlib import Path

import numpy as np

from src.utils.audio import apply_gain, babble, colored_noise, limit_peak, mix_at_snr, pad, power, resample, to_int16
from src.utils.io import read_jsonl, write_jsonl

MIN_SPEECH_SECONDS = 0.25


def make_noise(row: dict, n: int, pool: list[np.ndarray], index: int, acfg: dict, rng: np.random.Generator) -> np.ndarray:
    if row["noise"] == "babble":
        choices = [i for i in range(len(pool)) if i != index]
        picked = rng.choice(choices, size=min(acfg["babble_voices"], len(choices)), replace=False) if choices else []
        return babble([pool[i] for i in picked], n, rng)
    return colored_noise(n, row["noise"], rng)


def render_clip(speech: np.ndarray, row: dict, pool: list[np.ndarray], index: int, scfg: dict) -> np.ndarray:
    """Speech at 16 kHz (float32) → final augmented clip (float32)."""
    rate, acfg = scfg["audio"]["sample_rate"], scfg["augment"]
    rng = np.random.default_rng(row["seed"])
    audio = pad(speech, row["pad_ms"][0], row["pad_ms"][1], rate)
    if row["noise"]:
        audio = mix_at_snr(audio, make_noise(row, len(audio), pool, index, acfg, rng), row["snr_db"], reference=speech)
    return limit_peak(apply_gain(audio, row["gain_db"]), scfg["audio"]["peak"])


def clip_problems(speech: np.ndarray, rate: int) -> list[str]:
    problems = []
    if len(speech) < MIN_SPEECH_SECONDS * rate:
        problems.append("too_short")
    if power(speech) < 1e-6:
        problems.append("silent")
    return problems


def synth_shard(engine: dict, rows: list[dict], scfg: dict, synthesize) -> tuple[np.ndarray, list[dict]]:
    """All clips of one shard → (concatenated int16 audio, index rows)."""
    rate = scfg["audio"]["sample_rate"]
    pool = [resample(synthesize(engine, r["text"], r["voice"], r["speed"]), scfg["tts"]["sample_rate"], rate) for r in rows]
    pieces, index, offset = [], [], 0
    for i, (row, speech) in enumerate(zip(rows, pool)):
        clip = to_int16(render_clip(speech, row, pool, i, scfg))
        pieces.append(clip)
        index.append({**row, "offset": offset, "n_samples": len(clip), "seconds": round(len(clip) / rate, 3),
                      "speech_seconds": round(len(speech) / rate, 3), "problems": clip_problems(speech, rate)})
        offset += len(clip)
    return (np.concatenate(pieces) if pieces else np.zeros(0, np.int16)), index


# ── shard files ─────────────────────────────────────────────────────────────

def shard_paths(audio_dir: str, split: str, shard: int) -> tuple[Path, Path]:
    base = Path(audio_dir) / split / f"shard-{shard:05d}"
    return base.with_suffix(".npy"), base.with_suffix(".jsonl")


def shard_done(audio_dir: str, split: str, shard: int) -> bool:
    return all(p.exists() for p in shard_paths(audio_dir, split, shard))


def write_shard(audio_dir: str, split: str, shard: int, audio: np.ndarray, index: list[dict]) -> None:
    """Index written last: a shard counts as done only when both files exist."""
    npy, jsonl = shard_paths(audio_dir, split, shard)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, audio)
    write_jsonl(jsonl, index)


def load_clip(audio_dir: str, row: dict) -> np.ndarray:
    npy, _ = shard_paths(audio_dir, row["split"], row["shard"])
    return np.load(npy, mmap_mode="r")[row["offset"]: row["offset"] + row["n_samples"]]


def read_index(audio_dir: str, split: str) -> list[dict]:
    return [row for path in sorted((Path(audio_dir) / split).glob("shard-*.jsonl")) for row in read_jsonl(path)]
