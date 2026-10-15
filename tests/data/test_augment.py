import random

import pytest
from conftest import make_row

from src.services.data.augment import augment_row, check_sample, spoken_temp
from src.utils.config import load_yaml


@pytest.fixture(scope="module")
def acfg() -> dict:
    return load_yaml("configs/augment.yaml")


def weather_row(**overrides) -> dict:
    return make_row(id="w-01", tool_focus="get_weather", utterance="weather in tokyo today", reply="Today's weather.",
                    tool_calls=[{"name": "get_weather", "arguments": {"day": "today", "location": "Tokyo"}}], **overrides)


def lock_row() -> dict:
    return make_row(id="c-01", category="confirm", needs_confirmation=True, utterance="lock it", reply="Lock the screen?",
                    tool_calls=[{"name": "lock_screen", "arguments": {}}])


def roles(item: dict) -> list[str]:
    return [m["role"] for m in item["messages"]]


def test_action_is_one_turn_with_reply_and_call(acfg, ctx):
    [item] = [s for s in augment_row(make_row(), acfg, "train") if s["kind"] == "action"]
    assert roles(item) == ["user", "assistant"]
    assert item["messages"][1]["content"] == "Opening the browser."
    assert item["messages"][1]["tool_calls"][0]["function"]["name"] == "open_app"
    assert check_sample(item, ctx) == []


def test_weather_waits_for_data(acfg, ctx):
    [item] = [s for s in augment_row(weather_row(), acfg, "train") if s["kind"] == "weather"]
    assert roles(item) == ["user", "assistant", "tool", "assistant"]
    assert item["messages"][1]["content"] == ""
    assert "Tokyo" in item["messages"][3]["content"] and "degrees" in item["messages"][3]["content"]
    assert check_sample(item, ctx) == []


def test_confirm_makes_yes_and_no(acfg, ctx):
    items = {s["kind"]: s for s in augment_row(lock_row(), acfg, "train")}
    assert set(items) == {"confirm_yes", "confirm_no"}
    assert items["confirm_yes"]["messages"][3]["tool_calls"][0]["function"]["name"] == "lock_screen"
    assert "tool_calls" not in items["confirm_no"]["messages"][3]
    assert all(check_sample(item, ctx) == [] for item in items.values())


def test_test_split_uses_held_out_phrases(acfg):
    yes_train = set(acfg["confirm"]["yes"]["train"])
    answers = {augment_row({**lock_row(), "id": f"c-{i}"}, acfg, "test")[0]["messages"][2]["content"] for i in range(30)}
    assert answers.isdisjoint(yes_train)


def test_error_copy_apologises(acfg, ctx):
    errors = [s for i in range(400) for s in augment_row(make_row(id=f"e-{i}"), acfg, "train") if s["kind"] == "tool_error"]
    assert 5 <= len(errors) <= 40  # ~5% of 400
    item = errors[0]
    assert roles(item) == ["user", "assistant", "tool", "assistant"]
    assert '"ok": false' in item["messages"][2]["content"]
    assert "browser" in item["messages"][3]["content"]
    assert check_sample(item, ctx) == []


def test_augment_is_deterministic(acfg):
    assert augment_row(weather_row(), acfg, "train") == augment_row(weather_row(), acfg, "train")


def test_gate_catches_confirm_tool_without_yes(ctx):
    item = {"messages": [{"role": "user", "content": "lock it"},
                         {"role": "assistant", "content": "Locking.", "tool_calls": [{"type": "function", "function": {"name": "lock_screen", "arguments": {}}}]}]}
    assert "confirm_tool_without_yes" in check_sample(item, ctx)


def test_spoken_temp():
    assert spoken_temp(-3) == "minus 3" and spoken_temp(18) == "18"
