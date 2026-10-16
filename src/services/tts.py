"""Text-to-speech engines behind a registry: engine name → (load, synthesize).

    engine = load_engine(tts_cfg)              # once per container
    audio  = synthesize(engine, text, voice, speed)   # float32 at tts_cfg["sample_rate"]
"""
from typing import Callable

import numpy as np


# ── kokoro ──────────────────────────────────────────────────────────────────

def load_kokoro(tts_cfg: dict) -> dict:
    """One shared model on the GPU; one G2P pipeline per accent (a = American, b = British)."""
    import torch
    from kokoro import KModel, KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id=tts_cfg["repo_id"]).to(device).eval()
    pipelines = {lang: KPipeline(lang_code=lang, repo_id=tts_cfg["repo_id"], model=model) for lang in ("a", "b")}
    return {"name": "kokoro", "pipelines": pipelines}


def kokoro_speak(engine: dict, text: str, voice: str, speed: float) -> np.ndarray:
    """Kokoro splits long text into chunks; we join them back into one waveform."""
    pipeline = engine["pipelines"][voice[0]]
    chunks = []
    for result in pipeline(text, voice=voice, speed=speed):
        audio = result.audio if hasattr(result, "audio") else result[2]
        if audio is not None:
            chunks.append(audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


ENGINES: dict[str, tuple[Callable[[dict], dict], Callable[..., np.ndarray]]] = {
    "kokoro": (load_kokoro, kokoro_speak),
}


def load_engine(tts_cfg: dict) -> dict:
    return ENGINES[tts_cfg["engine"]][0](tts_cfg)


def synthesize(engine: dict, text: str, voice: str, speed: float) -> np.ndarray:
    return ENGINES[engine["name"]][1](engine, text, voice, speed)
