"""Stage 2 · validate: judge every row against tools.yaml and the category rules. Never changes a row."""
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from src.services.data.paths import stage_dir
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, write_jsonl

TYPE_MAP = {"string": str, "array": list, "boolean": bool, "object": dict}


def build_context(tools_cfg: dict, validate_cfg: dict) -> dict:
    """Everything check_row needs, compiled once."""
    tools = tools_cfg["tools"]
    return {
        "schemas": {name: Draft202012Validator(spec["parameters"]) for name, spec in tools.items()},
        "confirm_tools": {name for name, spec in tools.items() if spec.get("confirm")},
        "fields": {field: TYPE_MAP[kind] for field, kind in validate_cfg["required_fields"].items()},
        "categories": validate_cfg["categories"],
        "reply_max_words": validate_cfg["reply_max_words"],
        "reply_forbidden": re.compile(validate_cfg["reply_forbidden_pattern"]),
    }


def check_fields(row: dict, fields: dict[str, type]) -> list[str]:
    problems = []
    for field, kind in fields.items():
        if field not in row:
            problems.append(f"field_missing:{field}")
        elif not isinstance(row[field], kind):
            problems.append(f"field_type:{field}")
        elif kind is str and not row[field].strip():
            problems.append(f"field_empty:{field}")
    return problems


def check_call(call: object, schemas: dict) -> list[str]:
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
        return ["call_malformed"]
    if call["name"] not in schemas:
        return [f"tool_unknown:{call['name']}"]
    return [f"args_invalid:{call['name']}:{err.message}" for err in schemas[call["name"]].iter_errors(call["arguments"])]


def check_category(row: dict, ctx: dict) -> list[str]:
    category, calls = row["category"], row["tool_calls"]
    rule = ctx["categories"].get(category)
    if rule is None:
        return [f"category_unknown:{category}"]
    problems = []
    if not rule["min_calls"] <= len(calls) <= rule["max_calls"]:
        problems.append(f"call_count:{category}:{len(calls)}")
    names = [c.get("name") for c in calls if isinstance(c, dict)]
    uses_confirm_tool = any(name in ctx["confirm_tools"] for name in names)
    if category == "confirm" and not all(name in ctx["confirm_tools"] for name in names):
        problems.append("confirm_without_confirm_tool")
    if category != "confirm" and uses_confirm_tool:
        problems.append(f"confirm_tool_outside_confirm:{category}")
    if row["needs_confirmation"] != (category == "confirm"):
        problems.append("needs_confirmation_mismatch")
    return problems


def check_reply(reply: str, category: str, ctx: dict) -> list[str]:
    limit = ctx["reply_max_words"].get(category, ctx["reply_max_words"]["default"])
    problems = []
    if len(reply.split()) > limit:
        problems.append(f"reply_too_long:{category}")
    if ctx["reply_forbidden"].search(reply):
        problems.append("reply_has_symbols")
    return problems


def check_row(row: dict, ctx: dict) -> list[str]:
    """All problems with one row; an empty list means the row is valid."""
    problems = check_fields(row, ctx["fields"])
    if problems:
        return problems
    for call in row["tool_calls"]:
        problems += check_call(call, ctx["schemas"])
    return problems + check_category(row, ctx) + check_reply(row["reply"], row["category"], ctx)


def find_duplicate_ids(rows: list[dict]) -> set[str]:
    counts = Counter(row.get("id") for row in rows)
    return {row_id for row_id, n in counts.items() if n > 1}


def reason_code(reason: str) -> str:
    """'args_invalid:set_volume:...' → 'args_invalid:set_volume' (drop free-text detail for counting)."""
    return ":".join(reason.split(":")[:2])


def inputs(cfg: dict) -> list[Path]:
    return [stage_dir(cfg, "ingest") / "ingested.jsonl", Path(cfg["tools_config"])]


def outputs(cfg: dict) -> list[Path]:
    out = stage_dir(cfg, "validate")
    return [out / "valid.jsonl", out / "rejected.jsonl"]


def run(cfg: dict) -> dict:
    rows = read_jsonl(inputs(cfg)[0])
    ctx = build_context(load_yaml(cfg["tools_config"]), cfg["validate"])
    duplicate_ids = find_duplicate_ids(rows)
    valid, rejected, reasons = [], [], Counter()
    for row in rows:
        problems = check_row(row, ctx) + (["id_duplicate"] if row.get("id") in duplicate_ids else [])
        if problems:
            rejected.append({**row, "reasons": problems})
            reasons.update({reason_code(p) for p in problems})
        else:
            valid.append(row)
    valid_path, rejected_path = outputs(cfg)
    write_jsonl(valid_path, valid)
    write_jsonl(rejected_path, rejected)
    return {"in": len(rows), "out": len(valid), "rejected": len(rejected), "reasons": dict(reasons.most_common())}
