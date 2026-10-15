"""Where each pipeline stage reads and writes. Every path comes from configs/data.yaml."""
from pathlib import Path


def stage_dir(cfg: dict, stage: str) -> Path:
    return Path(cfg["paths"]["artifacts_dir"]) / stage


def manifest_path(cfg: dict, stage: str) -> Path:
    return stage_dir(cfg, stage) / "manifest.json"


def split_path(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["splits_dir"]) / f"{split}.jsonl"
