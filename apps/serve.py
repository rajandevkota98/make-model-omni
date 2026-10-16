"""POC server: the trained step 3 model as a Modal class the laptop client calls.

    uv run modal deploy -m apps.serve          # deploy (scales to zero when idle)
    uv run modal run -m apps.serve --text "open the browser and go to facebook"   # one-off test

Modal needs a class to keep the model loaded between calls; it stays thin and the work lives in
src/services/assistant.py.
"""
import modal

from apps.common import VOLUME_PATH, train_image, volume
from src.utils.config import load_yaml

CFG = load_yaml("configs/serve.yaml")
app = modal.App(CFG["app_name"])


@app.cls(image=train_image, gpu=CFG["modal"]["gpu"], volumes={VOLUME_PATH: volume},
         scaledown_window=CFG["modal"]["scaledown_window"], timeout=CFG["modal"]["timeout_seconds"])
class Assistant:
    @modal.enter()
    def load(self) -> None:
        from src.services.assistant import load_assistant

        self.assistant = load_assistant(CFG)

    @modal.method()
    def respond(self, messages: list[dict]) -> dict:
        from src.services.assistant import respond

        return respond(self.assistant, messages)


@app.local_entrypoint()
def main(text: str = "open the browser and go to facebook") -> None:
    print(Assistant().respond.remote([{"role": "user", "content": text}]))
