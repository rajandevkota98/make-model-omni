"""Stage 5 · augment: build multi-turn conversations in the standard `messages` format.

Runtime protocol the samples teach (the "hybrid" pattern):
  turn 1   assistant speaks + emits tool calls in one message (actions run immediately)
  turn 2   only when a result needs words: get_weather data, or a tool error.
           The runtime never asks the model again after plain {"ok": true} results.
  confirm  assistant asks first; the confirm tool runs only after the user's "yes".
"""
import json
import random
from collections import Counter
from pathlib import Path

from src.services.data.paths import split_path
from src.services.data.split import SPLITS
from src.services.data.validate import build_context, check_call
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, write_jsonl

TOOL_CATEGORIES = {"single", "multi_step"}
META_FIELDS = ("category", "tool_focus", "style", "theme")


# ── message builders ────────────────────────────────────────────────────────

def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str, calls: list[dict] | None = None) -> dict:
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [{"type": "function", "function": call} for call in calls]
    return message


def tool_result(call: dict, result: dict) -> dict:
    return {"role": "tool", "name": call["name"], "content": json.dumps(result)}


def bank(phrases: dict, split: str) -> list[str]:
    """Test uses held-out phrasings; train and dev share the train bank."""
    return phrases["test" if split == "test" else "train"]


def spoken(value: object) -> str:
    return str(value).replace("_", " ")


# ── weather ─────────────────────────────────────────────────────────────────

def fake_weather(call: dict, rng: random.Random, wcfg: dict) -> dict:
    condition = rng.choice(sorted(wcfg["conditions"]))
    low, high = wcfg["conditions"][condition]
    location = call["arguments"].get("location") or rng.choice(wcfg["default_locations"])
    return {"ok": True, "location": location, "day": call["arguments"]["day"], "temp_c": rng.randint(low, high), "condition": condition}


def spoken_temp(temp_c: int) -> str:
    return f"minus {abs(temp_c)}" if temp_c < 0 else str(temp_c)


def weather_sentence(call: dict, result: dict, rng: random.Random, wcfg: dict, split: str) -> str:
    day = result["day"]
    where = f" in {result['location']}" if call["arguments"].get("location") else ""
    text = rng.choice(bank(wcfg["templates"], split)).format(
        it_is="It's" if day == "today" else "It'll be", day=day, Day=day.capitalize(),
        where=where, condition=result["condition"], temp=spoken_temp(result["temp_c"]))
    return text[:1].upper() + text[1:]


# ── errors ──────────────────────────────────────────────────────────────────

def error_result(call: dict, rng: random.Random, ecfg: dict) -> dict:
    return {"ok": False, "error": rng.choice(ecfg["codes"][call["name"]])}


def apology(call: dict, result: dict, rng: random.Random, ecfg: dict, split: str) -> str:
    action = ecfg["actions"][call["name"]].format(**{k: spoken(v) for k, v in call["arguments"].items()})
    reason = ecfg["reasons"][result["error"]]
    text = rng.choice(bank(ecfg["templates"], split)).format(action=action, reason=reason, Reason=reason[:1].upper() + reason[1:])
    return text[:1].upper() + text[1:]


def can_fail(call: dict, ecfg: dict) -> bool:
    return call["name"] in ecfg["codes"]


# ── conversations ───────────────────────────────────────────────────────────

def tool_conversation(row: dict, rng: random.Random, acfg: dict, split: str, fail_index: int | None = None) -> list[dict]:
    calls = row["tool_calls"]
    has_weather = any(call["name"] == "get_weather" for call in calls)
    results, follow_up = [], []
    for index, call in enumerate(calls):
        if index == fail_index:
            result = error_result(call, rng, acfg["errors"])
            follow_up.append(apology(call, result, rng, acfg["errors"], split))
        elif call["name"] == "get_weather":
            result = fake_weather(call, rng, acfg["weather"])
            follow_up.append(weather_sentence(call, result, rng, acfg["weather"], split))
        else:
            result = {"ok": True}
        results.append(result)
    # weather can't be described before the data arrives, so turn 1 stays silent
    messages = [user(row["utterance"]), assistant("" if has_weather else row["reply"], calls)]
    if follow_up:
        messages += [tool_result(call, result) for call, result in zip(calls, results)]
        messages.append(assistant(" ".join(follow_up)))
    return messages


def confirm_conversation(row: dict, rng: random.Random, ccfg: dict, split: str, answer: str) -> list[dict]:
    calls = row["tool_calls"]
    opening = [user(row["utterance"]), assistant(row["reply"]), user(rng.choice(bank(ccfg[answer], split)))]
    if answer == "no":
        return opening + [assistant(rng.choice(bank(ccfg["cancel"], split)))]
    return opening + [assistant(rng.choice(bank(ccfg["done"][calls[0]["name"]], split)), calls)]


def sample(row: dict, split: str, kind: str, messages: list[dict]) -> dict:
    return {"id": f"{row['id']}#{kind}", "source_id": row["id"], "split": split, "kind": kind,
            **{field: row.get(field) for field in META_FIELDS}, "messages": messages}


def augment_row(row: dict, acfg: dict, split: str) -> list[dict]:
    rng = random.Random(f"{acfg['seed']}:{row['id']}")  # per-row seed: stable whatever the row order
    category = row["category"]
    if category == "confirm":
        return [sample(row, split, f"confirm_{answer}", confirm_conversation(row, rng, acfg["confirm"], split, answer))
                for answer in ("yes", "no")]
    if category not in TOOL_CATEGORIES:
        return [sample(row, split, category, [user(row["utterance"]), assistant(row["reply"])])]
    has_weather = any(call["name"] == "get_weather" for call in row["tool_calls"])
    samples = [sample(row, split, "weather" if has_weather else "action", tool_conversation(row, rng, acfg, split))]
    failable = [i for i, call in enumerate(row["tool_calls"]) if can_fail(call, acfg["errors"])]
    if failable and rng.random() < acfg["errors"]["rate"]:
        samples.append(sample(row, split, "tool_error", tool_conversation(row, rng, acfg, split, rng.choice(failable))))
    return samples


# ── gate ────────────────────────────────────────────────────────────────────

def check_sample(item: dict, ctx: dict) -> list[str]:
    """Structural rules every conversation must satisfy; empty list means OK."""
    messages, problems = item["messages"], []
    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        problems.append("bad_ends")
    for index, message in enumerate(messages):
        if message["role"] == "tool" and messages[index - 1]["role"] not in ("tool", "assistant"):
            problems.append("tool_without_call")
        if message["role"] != "assistant":
            continue
        calls = [tc["function"] for tc in message.get("tool_calls", [])]
        if not message["content"] and not calls:
            problems.append("empty_assistant")
        if message["content"] and ctx["reply_forbidden"].search(message["content"]):
            problems.append("reply_has_symbols")
        for call in calls:
            problems += check_call(call, ctx["schemas"])
            if call["name"] in ctx["confirm_tools"] and index < 3:
                problems.append("confirm_tool_without_yes")
        results = 0
        while index + 1 + results < len(messages) and messages[index + 1 + results]["role"] == "tool":
            results += 1
        if results not in (0, len(calls)):
            problems.append("tool_result_count")
    return problems


# ── stage ───────────────────────────────────────────────────────────────────

def augmented_path(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["augmented_dir"]) / f"{split}.jsonl"


def inputs(cfg: dict) -> list[Path]:
    return [split_path(cfg, name) for name in SPLITS] + [Path(cfg["augment_config"]), Path(cfg["tools_config"])]


def outputs(cfg: dict) -> list[Path]:
    return [augmented_path(cfg, name) for name in SPLITS]


def run(cfg: dict) -> dict:
    acfg = load_yaml(cfg["augment_config"])
    ctx = build_context(load_yaml(cfg["tools_config"]), cfg["validate"])
    rows_in, sizes, kinds, failures = 0, {}, {}, []
    for name in SPLITS:
        rows = read_jsonl(split_path(cfg, name))
        samples = [item for row in rows for item in augment_row(row, acfg, name)]
        failures += [(item["id"], problems) for item in samples if (problems := check_sample(item, ctx))]
        write_jsonl(augmented_path(cfg, name), samples)
        rows_in += len(rows)
        sizes[name] = len(samples)
        kinds[name] = dict(Counter(item["kind"] for item in samples).most_common())
    if failures:
        raise ValueError(f"{len(failures)} conversations failed the gate, e.g. {failures[:3]}")
    return {"in": rows_in, "out": sum(sizes.values()), "rejected": 0, "sizes": sizes, "kinds": kinds}
