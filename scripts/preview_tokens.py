"""Show formatted examples the way the model sees them: prompt tail (ignored) + target (learned).

    uv run python -m scripts.preview_tokens --split train --per-kind 1
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.services.data.format import load_tokenizer
from src.utils.config import load_yaml
from src.utils.io import read_jsonl

DIM, BOLD, RESET = "\033[2m", "\033[1;32m", "\033[0m"


def show(example: dict, tokenizer, tail_chars: int) -> str:
    ids, n = example["input_ids"], example["prompt_len"]
    prompt, target = tokenizer.decode(ids[:n]), tokenizer.decode(ids[n:])
    head = f"[{example['kind']}] {example['id']}  tokens={len(ids)} learned={len(ids) - n}"
    return f"{head}\n{DIM}…{prompt[-tail_chars:]}{RESET}{BOLD}{target}{RESET}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-kind", type=int, default=1)
    parser.add_argument("--tail", type=int, default=260, help="characters of prompt to show")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    tokenizer = load_tokenizer(cfg["format"])
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for example in read_jsonl(Path(cfg["paths"]["formatted_dir"]) / f"{args.split}.jsonl"):
        by_kind[example["kind"]].append(example)
    rng = random.Random(args.seed)
    for kind in sorted(by_kind):
        for example in rng.sample(by_kind[kind], min(args.per_kind, len(by_kind[kind]))):
            print(show(example, tokenizer, args.tail), end="\n\n")


if __name__ == "__main__":
    main()
