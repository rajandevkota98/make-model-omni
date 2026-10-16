import io

import numpy as np
import soundfile as sf

from src.services.librispeech import convert, parse_transcripts
from src.services.speech import load_clip, read_index


def flac(seconds: float) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, (np.sin(np.arange(int(16000 * seconds)) / 9) * 8000).astype(np.int16), 16000, format="FLAC")
    return buffer.getvalue()


def test_parse_transcripts_lowercases():
    assert parse_transcripts("84-1-0000 GO DO YOU HEAR\n84-1-0001 YES\n") == {"84-1-0000": "go do you hear", "84-1-0001": "yes"}


def test_convert_shards_and_joins_text(tmp_path):
    members = [("LibriSpeech/dev/84/1/84-1-000%d.flac" % i, flac(0.5 + i * 0.1)) for i in range(5)]
    members += [("LibriSpeech/dev/84/1/84-1-0009.flac", flac(31)), ("LibriSpeech/dev/84/1/84-1.trans.txt",
                 b"\n".join(b"84-1-000%d WORD %d" % (i, i) for i in range(5)))]
    summary = convert(iter(members), str(tmp_path), "dev", 2, 16000, 30)
    assert summary["clips"] == 5 and summary["shards"] == 3 and summary["too_long"] == 1 and summary["missing_text"] == 0
    rows = read_index(str(tmp_path), "dev")
    assert [r["text"] for r in rows] == [f"word {i}" for i in range(5)]
    assert len(load_clip(str(tmp_path), rows[3])) == rows[3]["n_samples"] == int(16000 * 0.8)
