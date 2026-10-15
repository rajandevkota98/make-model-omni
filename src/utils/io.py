"""JSON / JSONL file helpers."""
import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (1-based line number, text) for every non-blank line."""
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                yield lineno, line


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for _, line in read_lines(path)]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
