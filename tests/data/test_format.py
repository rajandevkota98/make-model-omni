import json

import pytest

from src.services.data.format import (IGNORE_INDEX, drop_repeats, format_conversation, labels_for, load_tokenizer,
                                      prompt_tools, strip_keys, turn_texts)
from src.utils.config import load_yaml

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def fcfg(data_cfg) -> dict:
    return data_cfg["format"]


@pytest.fixture(scope="module")
def tokenizer(fcfg):
    try:
        return load_tokenizer(fcfg)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached and no network")


@pytest.fixture(scope="module")
def tools(data_cfg, fcfg) -> list[dict]:
    return prompt_tools(load_yaml(data_cfg["tools_config"]), fcfg)


def weather_item() -> dict:
    call = {"name": "get_weather", "arguments": {"day": "today"}}
    return {"id": "w#weather", "kind": "weather", "messages": [
        {"role": "user", "content": "weather today"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
        {"role": "tool", "name": "get_weather", "content": json.dumps({"ok": True, "temp_c": 18})},
        {"role": "assistant", "content": "It's 18 degrees."}]}


def test_strip_keys_is_recursive():
    schema = {"type": "object", "if": 1, "properties": {"x": {"type": "string", "pattern": "^a"}}}
    assert strip_keys(schema, {"if", "pattern"}) == {"type": "object", "properties": {"x": {"type": "string"}}}


def test_prompt_hides_validation_keys(tools):
    text = json.dumps(tools)
    assert '"pattern"' not in text and '"if"' not in text and '"enum"' in text


def test_turn_matches_runtime_generation(tokenizer, tools):
    messages = [{"role": "system", "content": "sys"}, *weather_item()["messages"]]
    prompt, target = turn_texts(tokenizer, messages, tools, 2)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # same as add_generation_prompt at runtime
    assert target.startswith("<tool_call>") and target.endswith("<|im_end|>")


def test_one_example_per_assistant_turn(tokenizer, tools):
    examples = format_conversation(weather_item(), tokenizer, tools, "sys")
    assert [e["turn"] for e in examples] == [2, 4]
    first, second = (tokenizer.decode(e["input_ids"]) for e in examples)
    assert "<tool_call>" in first and "<tool_response>" not in first
    assert "<tool_response>" in second and second.endswith("It's 18 degrees.<|im_end|>")


def test_labels_mask_the_prompt(tokenizer, tools):
    example = format_conversation(weather_item(), tokenizer, tools, "sys")[1]
    labels = labels_for(example)
    n = example["prompt_len"]
    assert labels[:n] == [IGNORE_INDEX] * n
    assert tokenizer.decode(labels[n:]) == "It's 18 degrees.<|im_end|>"


def test_drop_repeats_keeps_first():
    examples = [{"id": "a", "input_ids": [1, 2]}, {"id": "b", "input_ids": [1, 2]}, {"id": "c", "input_ids": [3]}]
    assert [e["id"] for e in drop_repeats(examples)] == ["a", "c"]
