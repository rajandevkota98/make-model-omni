"""Playground server: the step 6 model (hears + calls tools) as a Modal class. Nothing is executed.

    uv run modal deploy -m apps.playground
    uv run python -m scripts.try_model --text "turn the volume down"
    uv run python -m scripts.try_model --talk          # speak, hear the reply
"""
import modal

from apps.common import VOLUME_PATH, playground_image, volume
from src.utils.config import load_yaml

CFG = load_yaml("configs/playground.yaml")
app = modal.App(CFG["app_name"])
M = CFG["modal"]


@app.cls(image=playground_image, gpu=M["gpu"], cpu=M["cpu"], memory=M["memory_mb"], volumes={VOLUME_PATH: volume},
         scaledown_window=M["scaledown_window"], timeout=M["timeout_seconds"])
class Playground:
    @modal.enter()
    def load(self) -> None:
        from src.services.playground import load_playground

        self.pg = load_playground(CFG)

    @modal.method()
    def turn(self, mode: str, messages: list[dict], audio: list[bytes], speak: bool = False) -> dict:
        from src.services.playground import run_turn

        return run_turn(self.pg, mode, messages, audio, speak)
