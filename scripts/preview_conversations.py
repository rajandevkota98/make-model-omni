"""Print a few augmented conversations per kind, for reading by eye.

    uv run python -m scripts.preview_conversations --split train --per-kind 2
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.utils.io import read_jsonl


def render_message(message: dict) -> str:
    role = message["role"]
    if role == "tool":
        return f"    tool[{message['name']}]  {message['content']}"
    calls = " ".join(f"<{tc['function']['name']} {json.dumps(tc['function']['arguments'])}>" for tc in message.get("tool_calls", []))
    return f"    {role:<9}  {message['content']} {calls}".rstrip()


def render(item: dict) -> str:
    return "\n".join([f"  [{item['kind']}] {item['id']}"] + [render_message(m) for m in item["messages"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-kind", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dir", default="data/augmented")
    args = parser.parse_args()
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for item in read_jsonl(Path(args.dir) / f"{args.split}.jsonl"):
        by_kind[item["kind"]].append(item)
    rng = random.Random(args.seed)
    for kind in sorted(by_kind):
        for item in rng.sample(by_kind[kind], min(args.per_kind, len(by_kind[kind]))):
            print(render(item), end="\n\n")


if __name__ == "__main__":
    main()
