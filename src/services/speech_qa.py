"""Round-trip check: does a strong ASR model hear the words we asked Kokoro to say?

Word error rate (WER) = (words substituted + deleted + inserted) / words in the reference.
Both sides go through Whisper's English normaliser first, so "ten" vs "10" or "it's" vs "it is"
don't count as errors.
"""
import random
from collections import defaultdict

import numpy as np


def load_asr(model_id: str) -> dict:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
    return {"processor": processor, "model": model}


def transcribe(asr: dict, clips: list[np.ndarray], rate: int) -> list[str]:
    import torch

    processor, model = asr["processor"], asr["model"]
    features = processor([c.astype(np.float32) for c in clips], sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(features.input_features.to("cuda", torch.float16), language="en", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)


def normaliser(asr: dict):
    tokenizer = asr["processor"].tokenizer
    if hasattr(tokenizer, "normalize"):
        return tokenizer.normalize
    return lambda text: " ".join("".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split())


def sample_rows(rows: list[dict], per_voice: int, seed: int = 0) -> list[dict]:
    by_voice = defaultdict(list)
    for row in rows:
        by_voice[row["voice"]].append(row)
    rng = random.Random(seed)
    return [r for voice in sorted(by_voice) for r in rng.sample(by_voice[voice], min(per_voice, len(by_voice[voice])))]


def wer_by(results: list[dict], key) -> dict:
    import jiwer

    groups = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return {name: {"n": len(g), "wer": round(jiwer.wer([r["ref"] for r in g], [r["hyp"] or "-" for r in g]), 4)}
            for name, g in sorted(groups.items())}


def summarise(results: list[dict], max_wer_clean: float) -> dict:
    clean = [r for r in results if not r["noise"]]
    by_voice_clean = wer_by(clean, lambda r: r["voice"])
    return {"n": len(results), "wer_all": wer_by(results, lambda r: "all")["all"]["wer"],
            "by_voice_clean": by_voice_clean, "by_noise": wer_by(results, lambda r: r["noise"] or "clean"),
            "flagged_voices": [v for v, s in by_voice_clean.items() if s["wer"] > max_wer_clean],
            "worst": sorted(results, key=lambda r: -r["wer"])[:25]}
