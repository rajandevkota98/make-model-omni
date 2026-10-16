from client.conversation import allowed_calls, assistant, handle, needs_words, next_history

CONFIRM = {"lock_screen", "type_text"}
VOLUME = {"name": "set_volume", "arguments": {"action": "up"}}
WEATHER = {"name": "get_weather", "arguments": {"day": "today"}}
LOCK = {"name": "lock_screen", "arguments": {}}


def scripted(*turns: dict):
    """A fake server that answers with the given turns in order and records what it was sent."""
    sent, queue = [], list(turns)

    def respond(messages: list[dict]) -> dict:
        sent.append(list(messages))
        return {"ms": 1, "prompt_tokens": 1, **queue.pop(0)}

    return respond, sent


def test_action_runs_without_follow_up():
    respond, sent = scripted({"reply": "Volume up.", "tool_calls": [VOLUME]})
    ran = []
    messages, events = handle([], "louder", respond, lambda c: ran.append(c) or {"ok": True}, CONFIRM)
    assert ran == [VOLUME] and len(sent) == 1
    assert {"say": "Volume up."} in events and messages[-1]["tool_calls"][0]["function"] == VOLUME


def test_weather_result_goes_back_for_words():
    respond, sent = scripted({"reply": "", "tool_calls": [WEATHER]}, {"reply": "It's 18 degrees and sunny today.", "tool_calls": []})
    result = {"ok": True, "location": "London", "day": "today", "temp_c": 18, "condition": "sunny"}
    messages, events = handle([], "weather?", respond, lambda c: result, CONFIRM)
    assert len(sent) == 2 and sent[1][-1]["role"] == "tool"
    assert [e["say"] for e in events if "say" in e] == ["It's 18 degrees and sunny today."]


def test_error_goes_back_and_follow_up_calls_are_ignored():
    respond, _ = scripted({"reply": "Volume up.", "tool_calls": [VOLUME]}, {"reply": "Sorry, no audio device.", "tool_calls": [VOLUME]})
    ran = []
    _, events = handle([], "louder", respond, lambda c: ran.append(c) or {"ok": False, "error": "no_audio_device"}, CONFIRM)
    assert len(ran) == 1 and {"ignored": VOLUME} in events


def test_confirm_tool_blocked_without_question():
    respond, _ = scripted({"reply": "Locked.", "tool_calls": [LOCK]})
    ran = []
    _, events = handle([], "lock it", respond, lambda c: ran.append(c) or {"ok": True}, CONFIRM)
    assert ran == [] and {"blocked": LOCK} in events


def test_confirm_tool_runs_after_question_and_yes():
    history = [{"role": "user", "content": "lock my screen"}, assistant("Lock the screen now?")]
    respond, _ = scripted({"reply": "Locking.", "tool_calls": [LOCK]})
    ran = []
    handle(history, "yes", respond, lambda c: ran.append(c) or {"ok": True}, CONFIRM)
    assert ran == [LOCK]


def test_allowed_calls_keeps_normal_tools():
    run, blocked = allowed_calls([VOLUME, LOCK], [], CONFIRM)
    assert run == [VOLUME] and blocked == [LOCK]


def test_needs_words():
    assert not needs_words({"ok": True})
    assert needs_words({"ok": False, "error": "x"}) and needs_words({"ok": True, "temp_c": 3})


def test_history_kept_only_while_question_open():
    asked = [{"role": "user", "content": "type hi"}, assistant("Type hi?")]
    done = [{"role": "user", "content": "louder"}, assistant("Volume up.", [VOLUME])]
    assert next_history(asked, True) == asked
    assert next_history(done, True) == [] and next_history(asked, False) == []
