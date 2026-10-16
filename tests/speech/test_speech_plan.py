from collections import Counter

import numpy as np

from src.services.data.speech_plan import plan_split, user_texts
from src.services.speech import load_clip, read_index, synth_shard, write_shard
from src.utils.config import load_yaml

SCFG = load_yaml("configs/speech.yaml")
TEXTS = Counter({"turn up the volume": 1, "open the browser": 1, "yes do it": 50})


def test_user_texts_counts_every_user_turn():
    convs = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "hi"}]}]
    assert user_texts(convs) == Counter({"hi": 2})


def test_train_voice_counts_and_frequent_texts():
    rows = plan_split("train", TEXTS, SCFG)
    per_text = Counter(r["text"] for r in rows)
    assert per_text["turn up the volume"] == SCFG["clips"]["train_per_text"]
    assert per_text["yes do it"] == len(SCFG["voices"]["train"])
    assert all(r["voice"] in SCFG["voices"]["train"] for r in rows)
    assert len({r["clip_id"] for r in rows}) == len(rows)


def test_test_split_uses_held_out_voice_clean_and_noisy():
    rows = plan_split("test", TEXTS, SCFG)
    assert all(r["voice"] in SCFG["voices"]["test"] for r in rows)
    for text in TEXTS:
        mine = [r for r in rows if r["text"] == text]
        assert sorted(r["noise"] is None for r in mine) == [False, True] and len({r["voice"] for r in mine}) == 1


def test_plan_is_reproducible():
    assert plan_split("dev", TEXTS, SCFG) == plan_split("dev", TEXTS, SCFG)


def fake_tts(engine, text, voice, speed):
    return (0.2 * np.sin(np.arange(int(24000 * 0.1 * len(text.split()) / speed)) / 5)).astype(np.float32)


def test_synth_shard_writes_sliceable_clips(tmp_path):
    rows = plan_split("train", TEXTS, SCFG)
    audio, index = synth_shard({}, rows, SCFG, fake_tts)
    assert audio.dtype == np.int16 and sum(r["n_samples"] for r in index) == len(audio)
    write_shard(str(tmp_path), "train", 0, audio, index)
    back = read_index(str(tmp_path), "train")
    assert len(back) == len(rows)
    clip = load_clip(str(tmp_path), {**back[1], "shard": 0})
    assert len(clip) == back[1]["n_samples"] and np.array_equal(clip, audio[back[1]["offset"]: back[1]["offset"] + len(clip)])
