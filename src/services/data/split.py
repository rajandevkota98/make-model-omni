"""Stage 4 · split: stratified train / dev / test. Once test.jsonl exists it is frozen and reused."""
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.services.data.paths import split_path, stage_dir
from src.services.data.transform import dedupe_key
from src.utils.io import read_jsonl, write_jsonl

SPLITS = ("train", "dev", "test")


def stratum(row: dict, keys: list[str]) -> str:
    return "|".join(str(row.get(key)) for key in keys)


def allocate(counts: dict[str, int], take: int) -> dict[str, int]:
    """Split `take` across groups in proportion to their size (largest-remainder method, exact total)."""
    total = sum(counts.values())
    if take > total:
        raise ValueError(f"cannot take {take} rows from {total}")
    exact = {group: n * take / total for group, n in counts.items()}
    quota = {group: int(share) for group, share in exact.items()}
    leftover = take - sum(quota.values())
    by_remainder = sorted(exact, key=lambda g: (exact[g] - quota[g], g), reverse=True)
    for group in by_remainder[:leftover]:
        quota[group] += 1
    return quota


def stratified_take(rows: list[dict], take: int, keys: list[str], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Return (taken, rest) with `taken` mirroring the stratum mix of `rows`."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        groups[stratum(row, keys)].append(row)
    quota = allocate({g: len(members) for g, members in groups.items()}, take)
    taken, rest = [], []
    for group in sorted(groups):
        members = groups[group][:]
        rng.shuffle(members)
        taken += members[: quota[group]]
        rest += members[quota[group]:]
    return taken, rest


def split_rows(rows: list[dict], split_cfg: dict, frozen_test: list[dict] | None = None) -> dict[str, list[dict]]:
    rng = random.Random(split_cfg["seed"])
    keys = split_cfg["stratify_on"]
    if frozen_test is None:
        test, rest = stratified_take(rows, split_cfg["test_size"], keys, rng)
    else:
        test_keys = {dedupe_key(row["utterance"]) for row in frozen_test}
        test, rest = frozen_test, [row for row in rows if dedupe_key(row["utterance"]) not in test_keys]
    dev, train = stratified_take(rest, split_cfg["dev_size"], keys, rng)
    return {"train": train, "dev": dev, "test": test}


def overlap(splits: dict[str, list[dict]]) -> int:
    """Number of commands that appear in more than one split (must be 0)."""
    seen = Counter()
    for rows in splits.values():
        seen.update({dedupe_key(row["utterance"]) for row in rows})
    return sum(1 for n in seen.values() if n > 1)


def inputs(cfg: dict) -> list[Path]:
    return [stage_dir(cfg, "transform") / "clean.jsonl"]


def outputs(cfg: dict) -> list[Path]:
    return [split_path(cfg, name) for name in SPLITS]


def run(cfg: dict) -> dict:
    rows = read_jsonl(inputs(cfg)[0])
    test_path = split_path(cfg, "test")
    reuse_test = test_path.exists() and not cfg["split"].get("refreeze_test", False)
    splits = split_rows(rows, cfg["split"], read_jsonl(test_path) if reuse_test else None)
    shared = overlap(splits)
    if shared:
        raise ValueError(f"{shared} commands appear in more than one split")
    for name in ("train", "dev") if reuse_test else SPLITS:
        write_jsonl(split_path(cfg, name), splits[name])
    sizes = {name: len(splits[name]) for name in SPLITS}
    return {"in": len(rows), "out": sum(sizes.values()), "rejected": 0, "sizes": sizes, "test_frozen_reused": reuse_test, "overlap": shared}
