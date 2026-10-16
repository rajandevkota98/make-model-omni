"""Score one generated assistant turn against the expected one. Pure Python, no torch.

A turn's text looks like:  "Volume up.\n<tool_call>\n{json}\n</tool_call><|im_end|>"
"""
import json
import re
from collections import defaultdict

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
END_TOKENS = ("<|im_end|>", "<|endoftext|>")
NEGATIVE = re.compile(r"\b(couldn't|could not|can't|cannot|didn't|unable|failed|sorry|wasn't able|not able|no internet|doesn't)\b", re.I)


def strip_end(text: str) -> str:
    for token in END_TOKENS:
        text = text.split(token)[0]
    return text.replace("<think>", "").replace("</think>", "").strip()


def parse_turn(text: str) -> dict:
    """{'reply': spoken text, 'calls': [{'name', 'arguments'}], 'json_ok': all call blocks parsed}"""
    body = strip_end(text)
    calls, json_ok = [], True
    for raw in TOOL_CALL.findall(body):
        try:
            call = json.loads(raw)
            if not isinstance(call, dict) or "name" not in call or not isinstance(call.get("arguments", {}), dict):
                raise ValueError
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
        except (json.JSONDecodeError, ValueError):
            json_ok = False
    if "<tool_call>" in body and not calls and json_ok:
        json_ok = False  # opened a call but never closed it
    reply = TOOL_CALL.sub("", body).replace("<tool_call>", "").strip()
    return {"reply": reply, "calls": calls, "json_ok": json_ok}


def normalise_value(value: object) -> object:
    return " ".join(value.casefold().split()) if isinstance(value, str) else value


def call_key(call: dict) -> str:
    arguments = {k: normalise_value(v) for k, v in call["arguments"].items()}
    return json.dumps({"name": call["name"], "arguments": arguments}, sort_keys=True)


def calls_equal(predicted: list[dict], expected: list[dict]) -> bool:
    """Same tools, same arguments (case/space-insensitive strings), same order."""
    return [call_key(c) for c in predicted] == [call_key(c) for c in expected]


def last_tool_response(prompt_text: str) -> dict | None:
    found = TOOL_RESPONSE.findall(prompt_text)
    try:
        return json.loads(found[-1]) if found else None
    except json.JSONDecodeError:
        return None


def spoken_temp(temp_c: int) -> str:
    return f"minus {abs(temp_c)}" if temp_c < 0 else str(temp_c)


def score_turn(predicted_text: str, expected_text: str, prompt_text: str, kind: str, schemas: dict, confirm_tools: set[str]) -> dict:
    pred, gold = parse_turn(predicted_text), parse_turn(expected_text)
    expects_call, predicts_call = bool(gold["calls"]), bool(pred["calls"])
    schema_ok = pred["json_ok"] and all(c["name"] in schemas and not list(schemas[c["name"]].iter_errors(c["arguments"])) for c in pred["calls"])
    score = {
        "kind": kind, "expects_call": expects_call, "predicts_call": predicts_call,
        "exact": calls_equal(pred["calls"], gold["calls"]) and pred["json_ok"],
        "json_ok": pred["json_ok"], "schema_ok": schema_ok,
        "false_action": not expects_call and predicts_call,
        "missed_action": expects_call and not predicts_call,
        "confirm_violation": any(c["name"] in confirm_tools for c in pred["calls"]) and not any(c["name"] in confirm_tools for c in gold["calls"]),
        "expected_tools": [c["name"] for c in gold["calls"]],
    }
    response = last_tool_response(prompt_text)
    if not expects_call and response is not None:
        if response.get("ok") and "temp_c" in response:
            reply = pred["reply"].casefold()
            score["weather_faithful"] = spoken_temp(response["temp_c"]) in reply and response["condition"].casefold() in reply
        elif response.get("ok") is False:
            score["error_honest"] = bool(NEGATIVE.search(pred["reply"])) and not predicts_call
    return score


def rate(scores: list[dict], key: str, where=lambda s: True) -> float | None:
    chosen = [s[key] for s in scores if key in s and where(s)]
    return round(sum(chosen) / len(chosen), 4) if chosen else None


def aggregate(scores: list[dict]) -> dict:
    tool_turns, silent_turns = (lambda s: s["expects_call"]), (lambda s: not s["expects_call"])
    summary = {
        "n": len(scores),
        "tool_exact": rate(scores, "exact", tool_turns),           # gate ≥ 0.90
        "all_exact": rate(scores, "exact"),
        "json_ok": rate(scores, "json_ok", lambda s: s["predicts_call"]),
        "schema_ok": rate(scores, "schema_ok", lambda s: s["predicts_call"]),
        "false_action_rate": rate(scores, "false_action", silent_turns),   # gate ≤ 0.05
        "missed_action_rate": rate(scores, "missed_action", tool_turns),   # gate ≤ 0.05
        "confirm_violations": sum(s["confirm_violation"] for s in scores),  # gate = 0
        "weather_faithful": rate(scores, "weather_faithful"),
        "error_honest": rate(scores, "error_honest"),
    }
    by_kind, by_tool = defaultdict(list), defaultdict(list)
    for s in scores:
        by_kind[s["kind"]].append(s)
        for tool in set(s["expected_tools"]):
            by_tool[tool].append(s)
    summary["by_kind"] = {k: {"n": len(v), "exact": rate(v, "exact"), "false_action": rate(v, "false_action", silent_turns)} for k, v in sorted(by_kind.items())}
    summary["by_tool"] = {k: {"n": len(v), "exact": rate(v, "exact")} for k, v in sorted(by_tool.items())}
    return summary


def gate(summary: dict) -> dict[str, bool]:
    check = lambda value, ok: value is not None and ok(value)
    return {"tool_exact>=0.90": check(summary["tool_exact"], lambda v: v >= 0.90),
            "false_action<=0.05": check(summary["false_action_rate"], lambda v: v <= 0.05),
            "missed_action<=0.05": check(summary["missed_action_rate"], lambda v: v <= 0.05),
            "confirm_violations==0": summary["confirm_violations"] == 0}
