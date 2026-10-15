"""Stage 3 · transform: normalise text, then drop duplicate commands and commands with conflicting labels."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.services.data.paths import stage_dir
from src.utils.io import read_jsonl, write_jsonl

NON_WORD = re.compile(r"[^a-z0-9 ]+")


def squash_spaces(text: str) -> str:
    return " ".join(text.split())


def capitalise_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def title_case(text: str) -> str:
    """'new york' → 'New York', but 'NYC' stays 'NYC' (unlike str.title)."""
    return " ".join(word[:1].upper() + word[1:] for word in text.split())


def normalise_call(call: dict, title_case_args: dict[str, list[str]]) -> dict:
    keys = set(title_case_args.get(call["name"], []))
    arguments = {
        key: title_case(squash_spaces(value)) if key in keys and isinstance(value, str) else value
        for key, value in call["arguments"].items()
    }
    return {"name": call["name"], "arguments": arguments}


def normalise_row(row: dict, transform_cfg: dict) -> dict:
    title_case_args = transform_cfg.get("title_case_args", {})
    return {
        **row,
        "utterance": squash_spaces(row["utterance"]),
        "reply": capitalise_first(squash_spaces(row["reply"])),
        "tool_calls": [normalise_call(call, title_case_args) for call in row["tool_calls"]],
    }


def dedupe_key(utterance: str) -> str:
    """'Open YouTube!' and 'open   youtube' → 'open youtube'."""
    return " ".join(NON_WORD.sub(" ", utterance.lower()).split())


def label_key(row: dict) -> str:
    """What the model is asked to produce for a command (case-insensitive)."""
    label = {"category": row["category"], "tool_calls": row["tool_calls"]}
    return json.dumps(label, sort_keys=True).lower()


def dedupe(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the first copy of each command; drop every copy of a command whose copies disagree."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[dedupe_key(row["utterance"])].append(row)
    kept, dropped = [], []
    for group in groups.values():
        if len({label_key(row) for row in group}) > 1:
            dropped += [{**row, "reasons": ["conflicting_labels"]} for row in group]
        else:
            kept.append(group[0])
            dropped += [{**row, "reasons": ["duplicate"]} for row in group[1:]]
    return kept, dropped


def inputs(cfg: dict) -> list[Path]:
    return [stage_dir(cfg, "validate") / "valid.jsonl"]


def outputs(cfg: dict) -> list[Path]:
    out = stage_dir(cfg, "transform")
    return [out / "clean.jsonl", out / "dropped.jsonl"]


def run(cfg: dict) -> dict:
    rows = [normalise_row(row, cfg["transform"]) for row in read_jsonl(inputs(cfg)[0])]
    kept, dropped = dedupe(rows)
    clean_path, dropped_path = outputs(cfg)
    write_jsonl(clean_path, kept)
    write_jsonl(dropped_path, dropped)
    reasons = Counter(row["reasons"][0] for row in dropped)
    return {"in": len(rows), "out": len(kept), "rejected": len(dropped), "reasons": dict(reasons)}
