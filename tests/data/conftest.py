import pytest

from src.services.data.validate import build_context
from src.utils.config import load_yaml


@pytest.fixture(scope="session")
def data_cfg() -> dict:
    return load_yaml("configs/data.yaml")


@pytest.fixture(scope="session")
def ctx(data_cfg: dict) -> dict:
    return build_context(load_yaml(data_cfg["tools_config"]), data_cfg["validate"])


def make_row(**overrides) -> dict:
    row = {
        "id": "t-01", "category": "single", "tool_focus": "open_app", "style": "terse", "theme": "work",
        "utterance": "open the browser", "reply": "Opening the browser.", "needs_confirmation": False,
        "tool_calls": [{"name": "open_app", "arguments": {"app": "browser"}}],
    }
    return {**row, **overrides}
