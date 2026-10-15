from conftest import make_row

from src.services.data.transform import dedupe, dedupe_key, normalise_row, sentence_case, title_case

TRANSFORM_CFG = {"title_case_args": {"get_weather": ["location"]}}


def test_title_case_keeps_acronyms():
    assert title_case("new york") == "New York"
    assert title_case("NYC") == "NYC"


def test_normalise_row():
    row = make_row(utterance="  open   the browser ", reply="opening it.",
                   tool_calls=[{"name": "get_weather", "arguments": {"day": "today", "location": "tokyo"}}])
    out = normalise_row(row, TRANSFORM_CFG)
    assert out["utterance"] == "open the browser"
    assert out["reply"] == "Opening it."
    assert out["tool_calls"][0]["arguments"]["location"] == "Tokyo"
    assert row["reply"] == "opening it."  # input row is not mutated


def test_dedupe_key():
    assert dedupe_key("Open YouTube!") == dedupe_key("open   youtube") == "open youtube"


def test_dedupe_keeps_first_copy():
    kept, dropped = dedupe([make_row(id="a"), make_row(id="b", utterance="Open the browser!")])
    assert [r["id"] for r in kept] == ["a"]
    assert dropped[0]["reasons"] == ["duplicate"]


def test_dedupe_drops_all_conflicting_copies():
    other = [{"name": "open_app", "arguments": {"app": "email"}}]
    kept, dropped = dedupe([make_row(id="a"), make_row(id="b", tool_calls=other)])
    assert kept == []
    assert {r["reasons"][0] for r in dropped} == {"conflicting_labels"}


def test_sentence_case():
    assert sentence_case("sorry, i can't. i'm able to open it! ok") == "Sorry, I can't. I'm able to open it! Ok"
    assert sentence_case("opening it in vim") == "Opening it in vim"
