"""Speak replies with the system TTS for now (spd-say). The trained Talker replaces this in step 8."""
import subprocess
from typing import Callable

from client.executor.shell import fill, first_installed


def make_speaker(speech_cfg: dict, enabled: bool) -> Callable[[str], None]:
    command = first_installed([speech_cfg["command"]]) if enabled and speech_cfg["enabled"] else None

    def speak(text: str) -> None:
        if command and text:
            subprocess.Popen(fill(command, {"text": text}), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return speak
