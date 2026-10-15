"""Stage 1 · ingest: read every raw JSONL file, keep going past broken lines, tag each row with its source."""
import json
from pathlib import Path

from src.services.data.paths import stage_dir
from src.utils.io import read_lines, write_jsonl


def parse_line(text: str, source: str, lineno: int) -> tuple[dict | None, dict | None]:
    """Return (row, None) on success or (None, rejection) when the line is not a JSON object."""
    try:
        row = json.loads(text)
    except json.JSONDecodeError as err:
        return None, {"_source": {"file": source, "line": lineno}, "reasons": [f"bad_json:{err.msg}"], "raw": text.strip()[:500]}
    if not isinstance(row, dict):
        return None, {"_source": {"file": source, "line": lineno}, "reasons": ["not_an_object"], "raw": text.strip()[:500]}
    return {**row, "_source": {"file": source, "line": lineno}}, None


def ingest_file(path: Path) -> tuple[list[dict], list[dict]]:
    rows, rejected = [], []
    for lineno, text in read_lines(path):
        row, rejection = parse_line(text, path.name, lineno)
        if row is not None:
            rows.append(row)
        else:
            rejected.append(rejection)
    return rows, rejected


def inputs(cfg: dict) -> list[Path]:
    return sorted(Path(cfg["paths"]["raw_dir"]).glob(cfg["ingest"]["pattern"]))


def outputs(cfg: dict) -> list[Path]:
    out = stage_dir(cfg, "ingest")
    return [out / "ingested.jsonl", out / "rejected.jsonl"]


def run(cfg: dict) -> dict:
    files = inputs(cfg)
    if not files:
        raise FileNotFoundError(f"no {cfg['ingest']['pattern']} files in {cfg['paths']['raw_dir']}")
    rows, rejected, per_file = [], [], {}
    for path in files:
        file_rows, file_rejected = ingest_file(path)
        rows += file_rows
        rejected += file_rejected
        per_file[path.name] = {"rows": len(file_rows), "rejected": len(file_rejected)}
    ingested_path, rejected_path = outputs(cfg)
    write_jsonl(ingested_path, rows)
    write_jsonl(rejected_path, rejected)
    return {"in": len(rows) + len(rejected), "out": len(rows), "rejected": len(rejected), "files": per_file}
