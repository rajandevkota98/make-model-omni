"""The allowlist: tool name → handler. Anything not listed here, or with arguments that break the
tools.yaml schema, is rejected before a single command runs. There is no shell tool, ever."""
from jsonschema import Draft202012Validator

from client.executor.apps import open_app, open_url, web_search
from client.executor.media import media_control
from client.executor.system import lock_screen, set_brightness, set_volume, take_screenshot, toggle_setting, type_text
from client.executor.timer import set_timer
from client.executor.weather import get_weather

HANDLERS = {
    "open_app": open_app, "open_url": open_url, "web_search": web_search,
    "set_volume": set_volume, "set_brightness": set_brightness, "toggle_setting": toggle_setting,
    "media_control": media_control, "set_timer": set_timer, "get_weather": get_weather,
    "take_screenshot": take_screenshot, "type_text": type_text, "lock_screen": lock_screen,
}


def build_validators(tools_cfg: dict) -> dict:
    return {name: Draft202012Validator(spec["parameters"]) for name, spec in tools_cfg["tools"].items() if name in HANDLERS}


def confirm_tools(tools_cfg: dict) -> set[str]:
    return {name for name, spec in tools_cfg["tools"].items() if spec.get("confirm")}


def execute(call: dict, validators: dict, env: dict, handlers: dict = HANDLERS) -> dict:
    """Run one tool call → the tool result sent back to the model."""
    name, args = call.get("name"), call.get("arguments", {})
    if name not in validators or name not in handlers:
        return {"ok": False, "error": "unknown_tool"}
    if not isinstance(args, dict) or any(True for _ in validators[name].iter_errors(args)):
        return {"ok": False, "error": "invalid_arguments"}
    return handlers[name](args, env)
