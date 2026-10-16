"""Pure numpy audio helpers. Float32 waveforms in [-1, 1], mono.

How noise mixing works (signal-to-noise ratio, SNR):
  SNR in dB = 10 · log10(speech power / noise power). 20 dB = speech 100× stronger than noise
  (quiet room); 5 dB = only ~3× stronger (busy café). We scale the noise to hit a chosen SNR.
"""
from math import gcd

import numpy as np

EPS = 1e-10


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if src_rate == dst_rate:
        return audio.astype(np.float32)
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def power(audio: np.ndarray) -> float:
    return float(np.mean(audio.astype(np.float64) ** 2)) + EPS


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """white = flat hiss · pink = 1/f (rain, fans) · brown = 1/f² (rumble, traffic)."""
    white = rng.standard_normal(n)
    exponent = {"white": 0.0, "pink": 0.5, "brown": 1.0}[kind]
    if exponent == 0.0:
        return white.astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0
    shaped = np.fft.irfft(spectrum / freqs ** exponent, n)
    return (shaped / (np.std(shaped) + EPS)).astype(np.float32)


def babble(others: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Background chatter: other utterances looped to length n, randomly offset, summed."""
    mix = np.zeros(n, dtype=np.float32)
    for clip in others:
        if len(clip) == 0:
            continue
        looped = np.tile(clip, n // len(clip) + 2)
        start = int(rng.integers(0, len(clip)))
        mix += looped[start:start + n] / np.sqrt(power(clip))
    return mix


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, reference: np.ndarray | None = None) -> np.ndarray:
    """speech + noise scaled to `snr_db`. `reference` = the speech without padding, so silence doesn't count."""
    scale = np.sqrt(power(speech if reference is None else reference) / (power(noise) * 10 ** (snr_db / 10)))
    return (speech + scale * noise[: len(speech)]).astype(np.float32)


def pad(audio: np.ndarray, lead_ms: int, trail_ms: int, rate: int) -> np.ndarray:
    return np.concatenate([np.zeros(rate * lead_ms // 1000, np.float32), audio, np.zeros(rate * trail_ms // 1000, np.float32)])


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * 10 ** (gain_db / 20)).astype(np.float32)


def limit_peak(audio: np.ndarray, peak: float) -> np.ndarray:
    top = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio * (peak / top) if top > peak else audio


def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def from_int16(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32767


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    return 10 * np.log10(power(clean) / power(noisy - clean))
