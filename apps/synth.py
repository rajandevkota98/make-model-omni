"""Step 4 on Modal: speak every planned clip with Kokoro, then check it with Whisper.

    uv run python -m pipelines.data --stage speech_plan   # local: decide the clips
    uv run modal run -m apps.synth --mode smoke            # ~20 clips → data/speech/samples/*.wav + speed
    uv run modal run -m apps.synth --mode full             # every shard not done yet (safe to re-run)
    uv run modal run -m apps.synth --mode qa               # round-trip WER per voice and noise type
"""
import io
import time
from pathlib import Path

import modal

from apps.common import VOLUME_PATH, tts_image, volume
from src.utils.config import load_yaml

CFG = load_yaml("configs/speech.yaml")
SPLITS = ("train", "dev", "test")
app = modal.App("make-model-omni-synth")
OPTIONS = dict(image=tts_image, gpu=CFG["modal"]["gpu"], volumes={VOLUME_PATH: volume}, timeout=CFG["modal"]["timeout_seconds"])


def volume_path(path: str) -> str:
    """'/vol/data/speech/plan' → 'data/speech/plan' (paths inside the Volume)."""
    return str(Path(path).relative_to(VOLUME_PATH))


def plan_rows(split: str, shard: int | None = None) -> list[dict]:
    from src.utils.io import read_jsonl

    rows = read_jsonl(Path(CFG["paths"]["volume_plan_dir"]) / f"{split}.jsonl")
    return rows if shard is None else [r for r in rows if r["shard"] == shard]


@app.cls(max_containers=CFG["modal"]["max_containers"], **OPTIONS)
class Synth:
    """Thin Modal wrapper: Kokoro stays loaded between shards; the work is in src/services/speech.py."""

    @modal.enter()
    def load(self) -> None:
        from src.services.tts import load_engine

        self.engine = load_engine(CFG["tts"])

    @modal.method()
    def shard(self, split: str, shard: int) -> dict:
        from src.services.speech import shard_done, synth_shard, write_shard
        from src.services.tts import synthesize

        audio_dir = CFG["paths"]["volume_audio_dir"]
        volume.reload()
        if shard_done(audio_dir, split, shard):
            return {"split": split, "shard": shard, "skipped": True}
        started = time.time()
        audio, index = synth_shard(self.engine, plan_rows(split, shard), CFG, synthesize)
        write_shard(audio_dir, split, shard, audio, index)
        volume.commit()
        return {"split": split, "shard": shard, "clips": len(index), "seconds_audio": round(len(audio) / CFG["audio"]["sample_rate"]),
                "seconds_wall": round(time.time() - started, 1), "problems": sum(bool(r["problems"]) for r in index)}

    @modal.method()
    def preview(self, rows: list[dict]) -> list[tuple[str, bytes, float]]:
        """Synthesise a few rows without saving shards → (clip_id, wav bytes, seconds per clip)."""
        import numpy as np
        from scipy.io import wavfile

        from src.services.speech import synth_shard
        from src.services.tts import synthesize

        started = time.time()
        audio, index = synth_shard(self.engine, rows, CFG, synthesize)
        per_clip = (time.time() - started) / max(1, len(rows))
        out = []
        for row in index:
            buffer = io.BytesIO()
            wavfile.write(buffer, CFG["audio"]["sample_rate"], np.asarray(audio[row["offset"]: row["offset"] + row["n_samples"]]))
            out.append((row["clip_id"], buffer.getvalue(), round(per_clip, 3)))
        return out


@app.function(**OPTIONS)
def qa(stamp: str) -> dict:
    """Whisper round trip on a sample of clips per voice (dev + test cover every voice)."""
    import jiwer

    from src.services.speech import load_clip, read_index
    from src.services.speech_qa import load_asr, normaliser, sample_rows, summarise, transcribe
    from src.utils.io import write_json

    volume.reload()
    q, audio_dir, rate = CFG["qa"], CFG["paths"]["volume_audio_dir"], CFG["audio"]["sample_rate"]
    rows = sample_rows(read_index(audio_dir, "train") + read_index(audio_dir, "test"), q["per_voice"])
    asr = load_asr(q["asr_model"])
    norm, results = normaliser(asr), []
    for i in range(0, len(rows), q["batch_size"]):
        batch = rows[i:i + q["batch_size"]]
        hyps = transcribe(asr, [load_clip(audio_dir, r) / 32767 for r in batch], rate)
        for row, hyp in zip(batch, hyps):
            ref, hyp_n = norm(row["text"]), norm(hyp)
            results.append({"clip_id": row["clip_id"], "voice": row["voice"], "noise": row["noise"], "snr_db": row["snr_db"],
                            "text": row["text"], "heard": hyp.strip(), "ref": ref, "hyp": hyp_n,
                            "wer": round(jiwer.wer(ref, hyp_n or "-"), 3) if ref else 0.0})
    summary = {**summarise(results, q["max_wer_clean"]), "asr_model": q["asr_model"]}
    write_json(Path(VOLUME_PATH) / "evals" / f"speech-qa-{stamp}.json", summary)
    volume.commit()
    return summary


def upload_plan() -> None:
    with volume.batch_upload(force=True) as batch:
        for split in SPLITS:
            batch.put_file(Path(CFG["paths"]["plan_dir"]) / f"{split}.jsonl", f"{volume_path(CFG['paths']['volume_plan_dir'])}/{split}.jsonl")


def local_rows(split: str) -> list[dict]:
    from src.utils.io import read_jsonl

    return read_jsonl(Path(CFG["paths"]["plan_dir"]) / f"{split}.jsonl")


def smoke() -> None:
    """A mix: every train voice, both test voices, clean and noisy, written as wavs to listen to."""
    rows, seen = [], set()
    for row in local_rows("train") + local_rows("test"):
        key = (row["voice"], row["noise"] is None)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    out_dir = Path(CFG["paths"]["samples_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = Synth().preview.remote(rows)
    for (clip_id, wav, per_clip), row in zip(clips, rows):
        (out_dir / f"{clip_id}.wav").write_bytes(wav)
        print(f"  {clip_id:<40} {row['noise'] or 'clean':<7} {row['text'][:60]}")
    total = sum(len(local_rows(s)) for s in SPLITS)
    print(f"\n{len(clips)} clips in {out_dir}/ · ~{clips[0][2]} s per clip → ~{clips[0][2] * total / 3600:.1f} GPU-hours for {total} clips")


def full() -> None:
    jobs = [(split, shard) for split in SPLITS for shard in sorted({r["shard"] for r in local_rows(split)})]
    print(f"{len(jobs)} shards → up to {CFG['modal']['max_containers']} containers")
    done = 0
    for result in Synth().shard.starmap(jobs, order_outputs=False):
        done += 1
        print(f"  [{done}/{len(jobs)}] {result}")


def show_qa(summary: dict) -> None:
    print(f"  WER all {summary['wer_all']} on {summary['n']} clips ({summary['asr_model']})")
    print("  clean WER by voice:", {v: s["wer"] for v, s in summary["by_voice_clean"].items()})
    print("  WER by noise:", {v: s["wer"] for v, s in summary["by_noise"].items()})
    print("  flagged voices:", summary["flagged_voices"] or "none")
    for r in summary["worst"][:8]:
        print(f"    {r['wer']:.2f} {r['voice']:<10} said: {r['text'][:50]!r}  heard: {r['heard'][:50]!r}")


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    upload_plan()
    if mode == "smoke":
        smoke()
    elif mode == "full":
        full()
    elif mode == "qa":
        show_qa(qa.remote(time.strftime("%Y%m%d-%H%M")))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
