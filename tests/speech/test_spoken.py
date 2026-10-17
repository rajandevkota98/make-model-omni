import random

import numpy as np

from src.services.data.spoken_format import heard, split_segments
from src.services.spoken_data import fill_slots, pick_clip, plan_epoch, replay_count

ECFG = {"frames_per_second": 50, "max_frames": 1500}


def pool_with(texts: dict[str, list[tuple[int, str | None]]]) -> dict:
    rows, audio, offset = [], [], 0
    for text, clips in texts.items():
        for n, noise in clips:
            rows.append({"text": text, "audio_dir": "d", "split": "s", "shard": 0, "offset": offset, "n_samples": n, "noise": noise})
            audio.append(np.ones(n, np.int16))
            offset += n
    by_text: dict = {}
    for r in rows:
        by_text.setdefault(r["text"], []).append(r)
    return {"rows": rows, "by_text": by_text, "audio": {("d", "s", 0): np.concatenate(audio)}}


def test_heard_replaces_only_user_turns():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}, {"role": "tool", "content": "{}"}]
    assert [m["content"] for m in heard(msgs, "<A>")] == ["s", "<A>", "{}"]


def test_split_segments_around_slots():
    assert split_segments([1, 9, 2, 3, 9, 4], 9) == [[1], [2, 3], [4]]
    assert split_segments([1, 2], 9) == [[1, 2]]


def test_fill_slots_expands_each_clip():
    pool = pool_with({"lock it": [(16000, None)], "yes": [(8000, None)]})
    example = {"segments": [[1], [2], [3]], "audio_texts": ["lock it", "yes"], "target": [7, 8], "kind": "confirm_yes", "id": "x"}
    clips = [pool["by_text"]["lock it"][0], pool["by_text"]["yes"][0]]
    item = fill_slots(example, clips, pool, 99, ECFG, 4)
    assert item["input_ids"] == [1] + [99] * 13 + [2] + [99] * 7 + [3, 7, 8]
    assert item["prompt_len"] == 23 and [len(w) for w in item["waveforms"]] == [16000, 8000]


def test_pick_clip_variant():
    pool = pool_with({"t": [(100, None), (100, "pink")]})
    rng = random.Random(0)
    assert all(pick_clip(pool, "t", rng, "clean")["noise"] is None for _ in range(10))
    assert all(pick_clip(pool, "t", rng, "noisy")["noise"] == "pink" for _ in range(10))


def test_replay_fraction_of_final_mix():
    n = replay_count(900, 0.10)
    assert n == 100 and abs(n / (900 + n) - 0.10) < 1e-9 and replay_count(900, 0.0) == 0


def test_plan_covers_each_source_once():
    plan = plan_epoch({"spoken": 10, "text": 3, "asr": 2}, 4, seed=0, epoch=0)
    for source, n in (("spoken", 10), ("text", 3), ("asr", 2)):
        assert sorted(i for s, idx in plan if s == source for i in idx) == list(range(n))
