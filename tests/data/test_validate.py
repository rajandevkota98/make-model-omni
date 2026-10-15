from conftest import make_row

from src.services.data.validate import check_row, find_duplicate_ids


def call(name: str, **arguments) -> list[dict]:
    return [{"name": name, "arguments": arguments}]


def test_good_row_has_no_problems(ctx):
    assert check_row(make_row(), ctx) == []


def test_missing_field(ctx):
    row = make_row()
    del row["reply"]
    assert check_row(row, ctx) == ["field_missing:reply"]


def test_unknown_tool(ctx):
    assert check_row(make_row(tool_calls=call("rm_rf")), ctx) == ["tool_unknown:rm_rf"]


def test_enum_violation(ctx):
    problems = check_row(make_row(tool_calls=call("open_app", app="firefox")), ctx)
    assert problems and problems[0].startswith("args_invalid:open_app")


def test_level_only_with_set(ctx):
    assert check_row(make_row(tool_calls=call("set_volume", action="set", level=40)), ctx) == []
    assert check_row(make_row(tool_calls=call("set_volume", action="set")), ctx)
    assert check_row(make_row(tool_calls=call("set_volume", action="up", level=40)), ctx)


def test_url_must_be_https(ctx):
    assert check_row(make_row(tool_calls=call("open_url", url="http://x.com")), ctx)


def test_call_count_per_category(ctx):
    assert "call_count:clarify:1" in check_row(make_row(category="clarify"), ctx)


def test_confirm_rules(ctx):
    lock = call("lock_screen")
    assert check_row(make_row(category="confirm", needs_confirmation=True, tool_calls=lock, reply="Lock it?"), ctx) == []
    assert "confirm_tool_outside_confirm:single" in check_row(make_row(tool_calls=lock), ctx)
    assert "needs_confirmation_mismatch" in check_row(make_row(category="confirm", tool_calls=lock), ctx)


def test_reply_limits(ctx):
    long_reply = " ".join(["word"] * 13)
    assert "reply_too_long:single" in check_row(make_row(reply=long_reply), ctx)
    assert check_row(make_row(category="refuse", tool_calls=[], reply=long_reply), ctx) == []
    assert "reply_has_symbols" in check_row(make_row(reply="Opening https://x.com"), ctx)


def test_duplicate_ids():
    assert find_duplicate_ids([make_row(), make_row(), make_row(id="t-02")]) == {"t-01"}
