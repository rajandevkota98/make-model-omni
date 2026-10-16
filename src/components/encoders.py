"""Frozen Whisper-small encoder ("the ears"). The decoder is dropped; only the encoder is kept.

Whisper always reads a 30 s window (3000 mel frames → 1500 output frames). Short clips are padded
to 30 s, run once, and the padded part of the output is sliced off, so a 3 s clip gives 150 frames.
"""
import math

import numpy as np
import torch

SAMPLE_RATE = 16000


def load_whisper(ecfg: dict) -> dict:
    from transformers import WhisperFeatureExtractor, WhisperModel

    extractor = WhisperFeatureExtractor.from_pretrained(ecfg["id"], revision=ecfg["revision"])
    model = WhisperModel.from_pretrained(ecfg["id"], revision=ecfg["revision"], dtype=torch.bfloat16)
    encoder = model.get_encoder().to("cuda").eval()
    encoder.requires_grad_(False)
    return {"extractor": extractor, "encoder": encoder}


def encoder_frames(n_samples: int, ecfg: dict) -> int:
    return min(ecfg["max_frames"], math.ceil(n_samples * ecfg["frames_per_second"] / SAMPLE_RATE))


@torch.no_grad()
def audio_features(whisper: dict, waveforms: list[np.ndarray], ecfg: dict) -> list[torch.Tensor]:
    """Waveforms (float32, 16 kHz) → one [frames, 768] tensor per clip, padding removed."""
    mel = whisper["extractor"](waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", device="cuda").input_features
    hidden = whisper["encoder"](mel.to("cuda", torch.bfloat16)).last_hidden_state
    return [hidden[i, : encoder_frames(len(w), ecfg)] for i, w in enumerate(waveforms)]
