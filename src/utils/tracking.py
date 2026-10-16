"""Experiment tracking: a local JSONL log always, plus Comet ML when enabled and a key is present.

A run is a plain dict {"comet": experiment | None, "jsonl": Path | None}; every function
is a no-op for the parts that are missing, so training code never branches on tracking.
"""
import json
import os
import time
from pathlib import Path


def flatten(params: dict, prefix: str = "") -> dict:
    """{'lora': {'r': 32}} → {'lora.r': 32} so Comet shows nested config as columns."""
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def start_comet(ccfg: dict, run_name: str):
    if not ccfg.get("enabled") or not os.environ.get("COMET_API_KEY"):
        return None
    import comet_ml  # only where tracking is actually on

    experiment = comet_ml.start(project_name=ccfg["project"], workspace=ccfg.get("workspace") or os.environ.get("COMET_WORKSPACE"))
    experiment.set_name(run_name)
    experiment.add_tags(ccfg.get("tags", []))
    return experiment


def start_run(tracking_cfg: dict, run_name: str, params: dict, jsonl_path: Path | None = None) -> dict:
    run = {"comet": start_comet(tracking_cfg.get("comet", {}), run_name), "jsonl": jsonl_path}
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    log_params(run, params)
    return run


def append_jsonl(path: Path | None, record: dict) -> None:
    if path is not None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def log_params(run: dict, params: dict) -> None:
    flat = flatten(params)
    append_jsonl(run["jsonl"], {"type": "params", "time": time.time(), **flat})
    if run["comet"] is not None:
        run["comet"].log_parameters(flat)


def log_metrics(run: dict, metrics: dict[str, float], step: int, prefix: str = "") -> None:
    named = {f"{prefix}{k}": v for k, v in metrics.items()}
    append_jsonl(run["jsonl"], {"type": "metrics", "step": step, "time": time.time(), **named})
    if run["comet"] is not None:
        run["comet"].log_metrics(named, step=step)


def log_table(run: dict, name: str, rows: list[dict]) -> None:
    """Small tables such as sample generations or per-kind eval scores."""
    append_jsonl(run["jsonl"], {"type": "table", "name": name, "rows": rows})
    if run["comet"] is not None and rows:
        headers = list(rows[0])
        run["comet"].log_table(f"{name}.csv", tabular_data=[[row.get(h) for h in headers] for row in rows], headers=headers)


def end_run(run: dict) -> None:
    if run["comet"] is not None:
        run["comet"].end()
