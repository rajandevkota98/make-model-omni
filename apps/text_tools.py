"""Step 3 on Modal: baseline, smoke, overfit, full training and evaluation of text tool SFT.

    uv run modal run -m apps.text_tools --mode baseline          # untrained Qwen3 on test
    uv run modal run -m apps.text_tools --mode smoke             # 2 steps
    uv run modal run -m apps.text_tools --mode overfit           # 10 examples → loss ≈ 0
    uv run modal run --detach -m apps.text_tools --mode full     # real run, then test eval
    uv run modal run -m apps.text_tools --mode eval --adapter /vol/ckpt/<run>/best [--weights merged_bf16]
    add --gpu A100 (etc.) to override the GPU, --limit N to eval fewer examples,
    --overrides '{"train": {"micro_batch": 8}}' to change any config value for one run
"""
import json
import secrets as pysecrets
import time

import modal

from apps.common import VOLUME_PATH, app, secrets, train_image, volume
from src.utils.config import deep_merge, load_yaml

CONFIG_PATH = "configs/train/text_tools.yaml"
TRACKING_PATH = "configs/tracking.yaml"
CFG = load_yaml(CONFIG_PATH)
GPU_OPTIONS = dict(image=train_image, volumes={VOLUME_PATH: volume}, secrets=secrets,
                   timeout=CFG["modal"]["timeout_hours"] * 3600)


def config_for(mode: str, overrides: dict | None = None) -> dict:
    return deep_merge(deep_merge(CFG, CFG["presets"].get(mode, {})), overrides or {})


@app.function(gpu=CFG["modal"]["gpu"], retries=modal.Retries(max_retries=2, initial_delay=30.0), **GPU_OPTIONS)
def train(mode: str, run_id: str, overrides: dict | None = None) -> dict:
    from src.services.runs import run_training

    return run_training(config_for(mode, overrides), run_id, load_yaml(TRACKING_PATH), commit=volume.commit)


@app.function(gpu=CFG["modal"]["gpu"], **GPU_OPTIONS)
def evaluate(adapter: str, split: str, name: str, limit: int, weights: str = "qlora") -> dict:
    from src.services.runs import run_evaluation

    volume.reload()
    summary = run_evaluation(CFG, adapter or None, split, name, load_yaml(TRACKING_PATH), limit or None, weights)
    volume.commit()
    return summary


def show(summary: dict) -> None:
    keys = ("n", "tool_exact", "all_exact", "json_ok", "schema_ok", "false_action_rate", "missed_action_rate",
            "confirm_violations", "weather_faithful", "error_honest")
    print("\n".join(f"  {k:<20} {summary.get(k)}" for k in keys))
    print("  by kind:", {k: v["exact"] for k, v in summary["by_kind"].items()})
    print("  gate:", summary["gate"])


@app.local_entrypoint()
def main(mode: str = "smoke", adapter: str = "", split: str = "test", limit: int = 0, gpu: str = "", overrides: str = "",
         weights: str = "qlora") -> None:
    stamp = f"{time.strftime('%Y%m%d-%H%M')}-{pysecrets.token_hex(2)}"  # unique even for parallel launches
    train_fn = train.with_options(gpu=gpu) if gpu else train
    eval_fn = evaluate.with_options(gpu=gpu) if gpu else evaluate
    if mode == "baseline":
        show(eval_fn.remote("", split, f"baseline-{split}-{stamp}", limit))
    elif mode == "eval":
        show(eval_fn.remote(adapter, split, f"eval-{weights}-{split}-{stamp}", limit, weights))
    elif mode in CFG["presets"]:
        run_id = f"{CFG['run_name']}-{mode}-{stamp}{'-' + gpu.lower() if gpu else ''}"
        summary = train_fn.remote(mode, run_id, json.loads(overrides) if overrides else None)
        print(summary)
        if mode == "full":
            show(eval_fn.remote(summary["best_adapter"], split, f"{run_id}-{split}", limit))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
