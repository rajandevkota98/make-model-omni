from client.executor.registry import HANDLERS, build_validators, execute
from client.executor.shell import fill, run_first
from client.executor.weather import condition_for
from src.utils.config import load_yaml

TOOLS = load_yaml("configs/tools.yaml")
CLIENT = load_yaml("configs/client.yaml")


def dry_env() -> dict:
    lines = []
    return {"cfg": CLIENT, "dry_run": True, "log": lines.append, "speak": lambda text: None, "lines": lines}


def test_every_tool_has_a_handler():
    assert set(TOOLS["tools"]) == set(HANDLERS)


def test_unknown_tool_and_bad_arguments_rejected():
    validators = build_validators(TOOLS)
    env = dry_env()
    assert execute({"name": "run_shell", "arguments": {"cmd": "rm -rf ~"}}, validators, env)["error"] == "unknown_tool"
    assert execute({"name": "open_url", "arguments": {"url": "http://x.com"}}, validators, env)["error"] == "invalid_arguments"
    assert execute({"name": "open_app", "arguments": {"app": "bash"}}, validators, env)["error"] == "invalid_arguments"
    assert env["lines"] == []


def test_fill_keeps_text_as_one_argument():
    assert fill(["xdotool", "type", "--", "{text}"], {"text": "hi; rm -rf ~ {x}"}) == ["xdotool", "type", "--", "hi; rm -rf ~ {x}"]


def test_run_first_reports_missing_program():
    assert run_first([["definitely-not-installed-xyz"]], {}, dry_env(), "not_installed", "failed") == {"ok": False, "error": "not_installed"}


def test_dry_run_prints_instead_of_running():
    env = dry_env()
    assert run_first([["ls", "{path}"]], {"path": "/tmp"}, env, "missing", "failed") == {"ok": True}
    assert env["lines"] == ["[dry-run] ls /tmp"]


def test_toggle_setting_maps_state_to_value(monkeypatch):
    seen = []
    monkeypatch.setattr("client.executor.system.run_first", lambda cands, values, env, missing, failed: seen.append(values) or {"ok": True})
    HANDLERS["toggle_setting"]({"setting": "do_not_disturb", "state": "on"}, dry_env())
    assert seen == [{"value": "false"}]   # DND on = GNOME banners off


def test_weather_conditions_use_training_words():
    assert condition_for(0, 5, True, 40) == "sunny" and condition_for(0, 5, False, 40) == "clear"
    assert condition_for(3, 50, True, 40) == "windy" and condition_for(63, 50, True, 40) == "rainy"
