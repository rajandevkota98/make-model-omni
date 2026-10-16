from collections import Counter

import numpy as np

from src.services.audio_data import iterate_audio, plan_epoch
from src.services.speech import write_shard


def make_rows(tmp_path, shards: int = 3, per: int = 5) -> list[dict]:
    rows = []
    for s in range(shards):
        index, offset, pieces = [], 0, []
        for i in range(per):
            n = 1600 * (i + 1)
            pieces.append(np.full(n, s * 100 + i, np.int16))
            index.append({"clip_id": f"{s}-{i}", "split": "train", "shard": s, "offset": offset, "n_samples": n, "text": "x"})
            offset += n
        write_shard(str(tmp_path), "train", s, np.concatenate(pieces), index)
        rows += [{**r, "audio_dir": str(tmp_path), "source": "t"} for r in index]
    return rows


def test_plan_covers_every_row_once_and_is_seeded(tmp_path):
    rows = make_rows(tmp_path)
    plan = plan_epoch(rows, 4, 2, seed=1, epoch=0)
    assert sorted(i for _, batch in plan for i in batch) == list(range(len(rows)))
    assert plan == plan_epoch(rows, 4, 2, seed=1, epoch=0) and plan != plan_epoch(rows, 4, 2, seed=1, epoch=1)


def test_batches_stay_inside_their_shard_group(tmp_path):
    rows = make_rows(tmp_path)
    for group, batch in plan_epoch(rows, 4, 2, seed=0, epoch=0):
        assert {(rows[i]["audio_dir"], rows[i]["split"], rows[i]["shard"]) for i in batch} <= set(group)


def test_iterate_audio_returns_the_right_clips(tmp_path):
    rows = make_rows(tmp_path)
    seen = Counter()
    for picked, waves in iterate_audio(rows, plan_epoch(rows, 3, 2, seed=0, epoch=0)):
        for row, wave in zip(picked, waves):
            shard, i = map(int, row["clip_id"].split("-"))
            assert len(wave) == row["n_samples"] and np.allclose(wave, (shard * 100 + i) / 32767)
            seen[row["clip_id"]] += 1
    assert len(seen) == len(rows) and set(seen.values()) == {1}
