"""Stage 5 · report: one JSON + one printed table summarising every stage and the final splits."""
from collections import Counter
from pathlib import Path

from src.services.data.paths import manifest_path, split_path
from src.services.data.split import SPLITS
from src.utils.io import read_json, read_jsonl, write_json

PIPELINE_STAGES = ("ingest", "validate", "transform", "split")


def tool_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(call["name"] for row in rows for call in row["tool_calls"]).most_common())


def split_profile(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "category": dict(Counter(row["category"] for row in rows).most_common()),
        "tool_calls": tool_counts(rows),
        "style": dict(Counter(row.get("style") for row in rows).most_common()),
    }


def build_report(manifests: dict[str, dict], splits: dict[str, list[dict]]) -> dict:
    return {
        "stages": {name: {k: m.get(k) for k in ("in", "out", "rejected", "reasons")} for name, m in manifests.items()},
        "splits": {name: split_profile(rows) for name, rows in splits.items()},
    }


def format_table(report: dict) -> str:
    lines = [f"{'stage':<10}{'in':>8}{'out':>8}{'dropped':>9}  top reasons"]
    for name, s in report["stages"].items():
        reasons = ", ".join(f"{r} {n}" for r, n in list((s.get("reasons") or {}).items())[:4])
        lines.append(f"{name:<10}{s['in']:>8}{s['out']:>8}{s['rejected']:>9}  {reasons}")
    categories = sorted({c for p in report["splits"].values() for c in p["category"]})
    lines += ["", f"{'split':<10}{'rows':>7}" + "".join(f"{c[:10]:>11}" for c in categories)]
    for name, p in report["splits"].items():
        lines.append(f"{name:<10}{p['rows']:>7}" + "".join(f"{p['category'].get(c, 0):>11}" for c in categories))
    return "\n".join(lines)


def inputs(cfg: dict) -> list[Path]:
    return [manifest_path(cfg, name) for name in PIPELINE_STAGES] + [split_path(cfg, name) for name in SPLITS]


def outputs(cfg: dict) -> list[Path]:
    return [Path(cfg["paths"]["reports_dir"]) / "data_report.json"]


def run(cfg: dict) -> dict:
    manifests = {name: read_json(manifest_path(cfg, name)) for name in PIPELINE_STAGES}
    splits = {name: read_jsonl(split_path(cfg, name)) for name in SPLITS}
    report = build_report(manifests, splits)
    write_json(outputs(cfg)[0], report)
    print(format_table(report))
    return {"in": sum(p["rows"] for p in report["splits"].values()), "out": 1, "rejected": 0}
