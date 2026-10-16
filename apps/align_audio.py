"""Step 5 on Modal: stage 1b audio alignment (only the audio adapter trains).

    uv run modal run -m apps.align_audio --mode smoke           # 2 steps, real models
    uv run modal run -m apps.align_audio --mode overfit         # 10 clips → loss ≈ 0
    uv run modal run -m apps.align_audio --mode bench           # speed
    uv run modal run -m apps.align_audio --mode five_percent    # 16K clips, then dev WER
    uv run modal run --detach -m apps.align_audio --mode full   # real run, then test WER
    uv run modal run -m apps.align_audio --mode eval --adapter /vol/ckpt/<run>/best [--split dev] [--limit N]
"""
import json
import secrets as pysecrets
import time

import modal

from apps.common import VOLUME_PATH, audio_train_image, secrets, volume
from src.utils.config import deep_merge, load_yaml

CFG = load_yaml("configs/train/align_audio.yaml")
TRACKING = "configs/tracking.yaml"
app = modal.App("make-model-omni-align-audio")
OPTIONS = dict(image=audio_train_image, volumes={VOLUME_PATH: volume}, secrets=secrets, cpu=CFG["modal"]["cpu"],
               memory=CFG["modal"]["memory_mb"], timeout=CFG["modal"]["timeout_hours"] * 3600)


def config_for(mode: str, overrides: dict | None = None) -> dict:
    return deep_merge(deep_merge(CFG, CFG["presets"].get(mode, {})), overrides or {})


@app.function(gpu=CFG["modal"]["gpu"], retries=modal.Retries(max_retries=2, initial_delay=30.0), **OPTIONS)
def train(mode: str, run_id: str, overrides: dict | None = None) -> dict:
    from src.services.align_runs import run_training

    volume.reload()
    return run_training(config_for(mode, overrides), run_id, load_yaml(TRACKING), commit=volume.commit)


@app.function(gpu=CFG["modal"]["gpu"], **OPTIONS)
def evaluate(mode: str, adapter: str, split: str, name: str, limit: int) -> dict:
    from src.services.align_runs import run_evaluation

    volume.reload()
    summary = run_evaluation(config_for(mode), adapter, split, name, load_yaml(TRACKING), limit or None)
    volume.commit()
    return summary


def show(summary: dict) -> None:
    for key, value in summary["wer"].items():
        print(f"  WER {key:<22} {value['wer']:.4f}  (n={value['n']})")
    p = summary["probe"]
    print(f"  probe: {p['distinct']} distinct nearest tokens of {p['tokens']} ({p['distinct_ratio']}) · top {p['top'][:5]}")


@app.local_entrypoint()
def main(mode: str = "smoke", adapter: str = "", split: str = "test", limit: int = 0, gpu: str = "", overrides: str = "") -> None:
    stamp = f"{time.strftime('%Y%m%d-%H%M')}-{pysecrets.token_hex(2)}"
    train_fn = train.with_options(gpu=gpu) if gpu else train
    eval_fn = evaluate.with_options(gpu=gpu) if gpu else evaluate
    if mode == "eval":
        show(eval_fn.remote("full", adapter, split, f"align-eval-{split}-{stamp}", limit))
        return
    if mode not in CFG["presets"]:
        raise SystemExit(f"unknown mode {mode!r}")
    run_id = f"{CFG['run_name']}-{mode}-{stamp}"
    summary = train_fn.remote(mode, run_id, json.loads(overrides) if overrides else None)
    print(summary)
    after = {"smoke": ("dev", 16), "overfit": ("dev", 0), "five_percent": ("dev", 0), "full": ("test", 0)}
    if mode in after:
        eval_split, eval_limit = after[mode]
        eval_mode = "overfit" if mode == "overfit" else "full"
        show(eval_fn.remote(eval_mode, summary["best_adapter"], eval_split, f"{run_id}-{eval_split}", limit or eval_limit))
