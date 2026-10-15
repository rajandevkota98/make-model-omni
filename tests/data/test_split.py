from conftest import make_row

from src.services.data.split import allocate, overlap, split_rows

SPLIT_CFG = {"seed": 42, "test_size": 10, "dev_size": 5, "stratify_on": ["category"]}


def rows(n: int) -> list[dict]:
    categories = ["single"] * 3 + ["refuse"]
    return [make_row(id=f"r{i:03d}", utterance=f"command {i}", category=categories[i % 4]) for i in range(n)]


def test_allocate_is_exact_and_proportional():
    quota = allocate({"a": 75, "b": 25}, 10)
    assert quota == {"a": 8, "b": 2} or quota == {"a": 7, "b": 3}
    assert sum(allocate({"a": 3, "b": 3, "c": 3}, 4).values()) == 4


def test_split_sizes_and_no_overlap():
    splits = split_rows(rows(100), SPLIT_CFG)
    assert {k: len(v) for k, v in splits.items()} == {"train": 85, "dev": 5, "test": 10}
    assert overlap(splits) == 0


def test_split_is_stratified():
    test = split_rows(rows(100), SPLIT_CFG)["test"]
    assert sum(r["category"] == "refuse" for r in test) in (2, 3)


def test_split_is_deterministic():
    ids = lambda s: [r["id"] for r in s["test"]]
    assert ids(split_rows(rows(100), SPLIT_CFG)) == ids(split_rows(rows(100), SPLIT_CFG))


def test_frozen_test_is_reused_and_excluded():
    data = rows(100)
    frozen = data[:10]
    splits = split_rows(data, SPLIT_CFG, frozen_test=frozen)
    assert splits["test"] == frozen
    assert overlap(splits) == 0 and len(splits["train"]) + len(splits["dev"]) == 90
