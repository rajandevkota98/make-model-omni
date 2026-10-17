"""Microphone and speaker on the laptop, via ALSA's arecord / aplay (no Python audio libraries)."""
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from src.utils.audio import resample, to_int16


def wav_to_pcm(path: str) -> bytes:
    """Any wav → 16 kHz mono int16 bytes."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    if data.dtype == np.int16:
        audio /= 32767
    elif data.dtype == np.int32:
        audio /= 2 ** 31
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return to_int16(resample(audio, rate, 16000)).tobytes()


def new_recording_path(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"rec-{time.strftime('%Y%m%d-%H%M%S')}.wav"


def record_seconds(path: Path, seconds: float) -> str:
    subprocess.run(["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(int(seconds)), str(path)], check=True)
    return str(path)


def record_until_enter(path: Path, prompt: str) -> str:
    """Start arecord now, stop it when Enter is pressed (SIGINT lets arecord finish the wav header)."""
    proc = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(path)])
    input(prompt)
    proc.send_signal(signal.SIGINT)
    proc.wait()
    return str(path)


def play(pcm: bytes, rate: int) -> None:
    """Blocking playback of int16 mono PCM."""
    if pcm:
        subprocess.run(["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1", "-"], input=pcm, check=False)
