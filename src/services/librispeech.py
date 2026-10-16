"""LibriSpeech → int16 audio shards (same format as step 4).

Two sources, same output:
  hf       Hugging Face parquet files (fast CDN; one file per Modal task) ← used
  openslr  the original .tar.gz, streamed (openslr serves ~0.5 MB/s, too slow for the 30 GB subsets)

Inside each .tar.gz: LibriSpeech/<subset>/<speaker>/<chapter>/<id>.flac plus one <speaker>-<chapter>.trans.txt
with lines "<id> TRANSCRIPT IN CAPITALS". Audio is written shard by shard as it streams; transcripts are
joined in at the end, and each shard's index (.jsonl) is written last, so a shard with an index is complete.
"""
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

from src.services.speech import shard_paths
from src.utils.io import write_jsonl


def parse_transcripts(text: str) -> dict[str, str]:
    """'84-121123-0000 GO DO YOU HEAR' → {'84-121123-0000': 'go do you hear'}"""
    pairs = (line.split(" ", 1) for line in text.splitlines() if " " in line)
    return {uid: words.strip().lower() for uid, words in pairs}


def stream_members(url: str) -> Iterator[tuple[str, bytes]]:
    """(member name, bytes) for every .flac and .trans.txt, straight from the HTTP stream."""
    with urllib.request.urlopen(url, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith((".flac", ".trans.txt")):
                yield member.name, tar.extractfile(member).read()


def decode_flac(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="int16")
    return audio, rate


def flush(buffer: list[tuple[str, np.ndarray]], audio_dir: str, subset: str, shard: int) -> list[dict]:
    """Write one shard's audio; return its index rows (text added later)."""
    npy, _ = shard_paths(audio_dir, subset, shard)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, np.concatenate([audio for _, audio in buffer]))
    rows, offset = [], 0
    for uid, audio in buffer:
        rows.append({"clip_id": uid, "split": subset, "shard": shard, "offset": offset, "n_samples": len(audio),
                     "seconds": round(len(audio) / 16000, 3), "speaker": uid.split("-")[0], "source": "librispeech"})
        offset += len(audio)
    return rows


def convert(members: Iterator[tuple[str, bytes]], audio_dir: str, subset: str, shard_size: int, rate: int, max_seconds: float) -> dict:
    texts: dict[str, str] = {}
    buffer: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []
    dropped = {"too_long": 0, "wrong_rate": 0}
    for name, data in members:
        if name.endswith(".trans.txt"):
            texts.update(parse_transcripts(data.decode("utf-8")))
            continue
        audio, clip_rate = decode_flac(data)
        if clip_rate != rate:
            dropped["wrong_rate"] += 1
        elif len(audio) > max_seconds * rate:
            dropped["too_long"] += 1
        else:
            buffer.append((Path(name).stem, audio))
        if len(buffer) == shard_size:
            rows += flush(buffer, audio_dir, subset, len(rows) // shard_size)
            buffer = []
    if buffer:
        rows += flush(buffer, audio_dir, subset, len(rows) // shard_size)
    return write_indexes(rows, texts, audio_dir, subset, dropped)


def write_indexes(rows: list[dict], texts: dict[str, str], audio_dir: str, subset: str, dropped: dict) -> dict:
    missing = [r["clip_id"] for r in rows if r["clip_id"] not in texts]
    by_shard: dict[int, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(row["shard"], []).append({**row, "text": texts.get(row["clip_id"], "")})
    for shard, shard_rows in by_shard.items():
        write_jsonl(shard_paths(audio_dir, subset, shard)[1], shard_rows)
    return {"subset": subset, "clips": len(rows), "shards": len(by_shard), "hours": round(sum(r["seconds"] for r in rows) / 3600, 1),
            "missing_text": len(missing), **dropped}


def subset_done(audio_dir: str, subset: str) -> bool:
    folder = Path(audio_dir) / subset
    return folder.exists() and (folder / "DONE").exists()


def mark_done(audio_dir: str, subset: str, summary: dict) -> None:
    (Path(audio_dir) / subset / "DONE").write_text(json.dumps(summary) + "\n")


# ── Hugging Face parquet ────────────────────────────────────────────────────

def parquet_clips(path: str) -> Iterator[tuple[str, bytes, str]]:
    """(utterance id, flac bytes, transcript) for every row, one row group at a time."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        for row in parquet.read_row_group(group, columns=["id", "audio", "text"]).to_pylist():
            yield row["id"], row["audio"]["bytes"], row["text"]


def convert_parquet(path: str, audio_dir: str, subset: str, file_index: int, shard_size: int, rate: int, max_seconds: float) -> dict:
    """One parquet file → shards numbered file_index·100 + j, so files never collide."""
    texts, buffer, rows, dropped = {}, [], [], {"too_long": 0, "wrong_rate": 0}
    for uid, flac, text in parquet_clips(path):
        audio, clip_rate = decode_flac(flac)
        if clip_rate != rate:
            dropped["wrong_rate"] += 1
            continue
        if len(audio) > max_seconds * rate:
            dropped["too_long"] += 1
            continue
        texts[uid] = text.strip().lower()
        buffer.append((uid, audio))
        if len(buffer) == shard_size:
            rows += flush(buffer, audio_dir, subset, file_index * 100 + len(rows) // shard_size)
            buffer = []
    if buffer:
        rows += flush(buffer, audio_dir, subset, file_index * 100 + len(rows) // shard_size)
    return write_indexes(rows, texts, audio_dir, subset, dropped)


def part_done_path(audio_dir: str, subset: str, file_index: int) -> Path:
    return Path(audio_dir) / subset / f"part-{file_index:04d}.done"
