"""One user command, end to end, following the runtime protocol the model was trained on:

  1. model turn 1: a short spoken reply + tool calls (actions run right away)
  2. only if a result needs words (weather data or an error) → tool results go back → model turn 2
  3. confirm tools (type_text, lock_screen) run only right after the model asked and the user answered

`respond(messages) -> {"reply", "tool_calls", ...}` and `execute(call) -> result` are passed in,
so this file has no Modal or desktop code and is easy to test.
"""
import json
from typing import Callable

Respond = Callable[[list[dict]], dict]
Execute = Callable[[dict], dict]


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str, calls: list[dict] | None = None) -> dict:
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [{"type": "function", "function": call} for call in calls]
    return message


def tool_message(call: dict, result: dict) -> dict:
    return {"role": "tool", "name": call["name"], "content": json.dumps(result)}


def is_question(message: dict | None) -> bool:
    return bool(message) and message["role"] == "assistant" and not message.get("tool_calls") and message["content"].rstrip().endswith("?")


def last_assistant(history: list[dict]) -> dict | None:
    return next((m for m in reversed(history) if m["role"] == "assistant"), None)


def needs_words(result: dict) -> bool:
    """Plain {"ok": true} needs no follow-up; data (weather) or a failure does."""
    return not result.get("ok") or len(result) > 1


def allowed_calls(calls: list[dict], history: list[dict], confirm: set[str]) -> tuple[list[dict], list[dict]]:
    """Split into (run, blocked): a confirm tool needs the previous assistant turn to have asked."""
    asked = is_question(last_assistant(history))
    run = [c for c in calls if c["name"] not in confirm or asked]
    return run, [c for c in calls if c not in run]


def handle(history: list[dict], text: str, respond: Respond, execute: Execute, confirm: set[str]) -> tuple[list[dict], list[dict]]:
    """→ (new history, events). Events: {"say": text} · {"call", "result"} · {"blocked"} · {"ignored"} · {"model"}"""
    events: list[dict] = []
    messages = [*history, user(text)]
    first = respond(messages)
    events.append({"model": first})
    calls, blocked = allowed_calls(first["tool_calls"], history, confirm)
    events += [{"blocked": call} for call in blocked]
    messages.append(assistant(first["reply"], calls))
    if first["reply"]:
        events.append({"say": first["reply"]})
    results = []
    for call in calls:
        result = execute(call)
        results.append(result)
        events.append({"call": call, "result": result})
    if any(needs_words(r) for r in results):
        messages += [tool_message(c, r) for c, r in zip(calls, results)]
        second = respond(messages)
        events.append({"model": second})
        events += [{"ignored": call} for call in second["tool_calls"]]   # turn 2 only talks
        messages.append(assistant(second["reply"]))
        if second["reply"]:
            events.append({"say": second["reply"]})
    return messages, events


def next_history(messages: list[dict], keep_after_question: bool) -> list[dict]:
    """Training conversations start fresh for each command, so only keep context while a question is open."""
    return messages if keep_after_question and is_question(last_assistant(messages)) else []
