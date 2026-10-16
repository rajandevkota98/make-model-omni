from jsonschema import Draft202012Validator

from src.utils.metrics import aggregate, calls_equal, gate, parse_turn, score_turn

SCHEMAS = {"set_volume": Draft202012Validator({"type": "object", "properties": {"action": {"enum": ["up", "down"]}}, "required": ["action"]}),
           "lock_screen": Draft202012Validator({"type": "object"})}
CONFIRM = {"lock_screen"}
CALL = '<tool_call>\n{"name": "set_volume", "arguments": {"action": "up"}}\n</tool_call><|im_end|>'


def test_parse_turn_reply_and_call():
    turn = parse_turn("Volume up.\n" + CALL)
    assert turn["reply"] == "Volume up." and turn["calls"][0]["name"] == "set_volume" and turn["json_ok"]


def test_parse_turn_broken_json():
    assert parse_turn("<tool_call>{oops</tool_call>")["json_ok"] is False
    assert parse_turn("<tool_call>{\"name\": \"x\"")["json_ok"] is False  # never closed


def test_calls_equal_ignores_case_and_space():
    a = [{"name": "get_weather", "arguments": {"location": "New  York"}}]
    b = [{"name": "get_weather", "arguments": {"location": "new york"}}]
    assert calls_equal(a, b)


def test_score_false_action_and_confirm_violation():
    lock = '<tool_call>\n{"name": "lock_screen", "arguments": {}}\n</tool_call>'
    score = score_turn(lock, "Lock the screen?<|im_end|>", "", "confirm_yes", SCHEMAS, CONFIRM)
    assert score["false_action"] and score["confirm_violation"] and not score["exact"]


def test_score_weather_faithfulness():
    prompt = '<tool_response>\n{"ok": true, "temp_c": -3, "condition": "snowy"}\n</tool_response>'
    good = score_turn("It's minus 3 degrees and snowy.", "x", prompt, "weather", SCHEMAS, CONFIRM)
    bad = score_turn("It's 20 degrees and sunny.", "x", prompt, "weather", SCHEMAS, CONFIRM)
    assert good["weather_faithful"] and not bad["weather_faithful"]


def test_aggregate_and_gate():
    scores = [score_turn("Up. " + CALL, "Volume up. " + CALL, "", "action", SCHEMAS, CONFIRM),
              score_turn("Hi there.", "Hello!", "", "no_tool", SCHEMAS, CONFIRM)]
    summary = aggregate(scores)
    assert summary["tool_exact"] == 1.0 and summary["false_action_rate"] == 0.0
    assert all(gate(summary).values())
