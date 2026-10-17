"""Step 6 on Modal: spoken tool SFT (LoRA + audio adapter), then tool-call eval from held-out voices.

    uv run modal run -m apps.spoken_tools --mode baseline        # step 3 LoRA + step 5 ears, no joint training
    uv run modal run -m apps.spoken_tools --mode smoke           # 2 steps, real models
    uv run modal run -m apps.spoken_tools --mode overfit         # 10 examples → loss ≈ 0
    uv run modal run -m apps.spoken_tools --mode bench           # speed
    uv run modal run --detach -m apps.spoken_tools --mode full   # real run, then test eval (clean + noisy)
    uv run modal run -m apps.spoken_tools --mode eval --checkpoint /vol/ckpt/<run>/best [--split dev] [--limit N]

Long runs that must survive the laptop going offline: deploy once, then launch (train + test eval run
in one remote job; nothing depends on the local client):
    uv run modal deploy -m apps.spoken_tools
    uv run python -m apps.spoken_tools --mode full [--run-id <existing run to resume>]
"""
import json
import secrets as pysecrets
import time

import modal

from apps.common import VOLUME_PATH, audio_train_image, secrets, volume
from src.utils.config import deep_merge, load_yaml

CFG = load_yaml("configs/train/spoken_tools.yaml")
TRACKING = "configs/tracking.yaml"
app = modal.App("make-model-omni-spoken-tools")
OPTIONS = dict(image=audio_train_image, volumes={VOLUME_PATH: volume}, secrets=secrets, cpu=CFG["modal"]["cpu"],
               memory=CFG["modal"]["memory_mb"], timeout=CFG["modal"]["timeout_hours"] * 3600)


def config_for(mode: str, overrides: dict | None = None) -> dict:
    return deep_merge(deep_merge(CFG, CFG["presets"].get(mode, {})), overrides or {})


@app.function(gpu=CFG["modal"]["gpu"], retries=modal.Retries(max_retries=2, initial_delay=30.0), **OPTIONS)
def train(mode: str, run_id: str, overrides: dict | None = None) -> dict:
    from src.services.spoken_runs import run_training

    volume.reload()
    return run_training(config_for(mode, overrides), run_id, load_yaml(TRACKING), commit=volume.commit)


@app.function(gpu=CFG["modal"]["gpu"], **OPTIONS)
def evaluate(mode: str, lora: str, adapter: str, split: str, name: str, limit: int) -> dict:
    from src.services.spoken_runs import run_evaluation

    volume.reload()
    summary = run_evaluation(config_for(mode), lora, adapter, split, name, load_yaml(TRACKING), limit or None)
    volume.commit()
    return summary


@app.function(**{**OPTIONS, "cpu": 1, "memory": 2048})
def pipeline(mode: str, run_id: str, overrides: dict | None = None) -> dict:
    """Train, then evaluate the best checkpoint on test, all remotely."""
    summary = train.remote(mode, run_id, overrides)
    best = summary["best"]
    result = evaluate.remote("full", f"{best}/lora", best, "test", f"{run_id}-test", 0)
    return {"train": summary, "test": result}


def show(summary: dict) -> None:
    keys = ("tool_exact", "all_exact", "schema_ok", "false_action_rate", "missed_action_rate", "confirm_violations",
            "weather_faithful", "error_honest")
    for variant in CFG["eval"]["variants"]:
        r = summary[variant]
        print(f"  [{variant}] " + " · ".join(f"{k} {r.get(k)}" for k in keys))
        print(f"  [{variant}] gate: {r['gate']}")


@app.local_entrypoint()
def main(mode: str = "smoke", checkpoint: str = "", split: str = "test", limit: int = 0, gpu: str = "", overrides: str = "") -> None:
    stamp = f"{time.strftime('%Y%m%d-%H%M')}-{pysecrets.token_hex(2)}"
    train_fn = train.with_options(gpu=gpu) if gpu else train
    eval_fn = evaluate.with_options(gpu=gpu) if gpu else evaluate
    if mode == "baseline":
        init = CFG["init"]
        show(eval_fn.remote("full", init["lora"], init["adapter"], split, f"spoken-baseline-{split}-{stamp}", limit))
        return
    if mode == "eval":
        show(eval_fn.remote("full", f"{checkpoint}/lora", checkpoint, split, f"spoken-eval-{split}-{stamp}", limit))
        return
    if mode not in CFG["presets"]:
        raise SystemExit(f"unknown mode {mode!r}")
    run_id = f"{CFG['run_name']}-{mode}-{stamp}"
    summary = train_fn.remote(mode, run_id, json.loads(overrides) if overrides else None)
    print(summary)
    after = {"smoke": ("dev", 8), "overfit": ("dev", 0), "full": ("test", 0)}
    if mode in after:
        eval_split, eval_limit = after[mode]
        best = summary["best"]
        show(eval_fn.remote("overfit" if mode == "overfit" else "full", f"{best}/lora", best, eval_split,
                            f"{run_id}-{eval_split}", limit or eval_limit))


def launch() -> None:
    """Spawn `pipeline` on the deployed app and return immediately."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--overrides", default="")
    opts = parser.parse_args()
    run_id = opts.run_id or f"{CFG['run_name']}-{opts.mode}-{time.strftime('%Y%m%d-%H%M')}-{pysecrets.token_hex(2)}"
    fn = modal.Function.from_name(app.name, "pipeline")
    call = fn.spawn(opts.mode, run_id, json.loads(opts.overrides) if opts.overrides else None)
    print(f"spawned {run_id} · call {call.object_id}")


if __name__ == "__main__":
    launch()
