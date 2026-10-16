"""Data pipeline runner: ingest → validate → transform → split → augment → format → speech_plan → report.

Each stage module exposes inputs(cfg), outputs(cfg) and run(cfg) -> manifest.
A stage is skipped when its inputs, its config section and its code are unchanged
since the last run (see manifest.json in data/artifacts/<stage>/).

    uv run python -m pipelines.data                    # run all stages
    uv run python -m pipelines.data --stage validate   # one stage
    uv run python -m pipelines.data --force            # ignore the cache
"""
import argparse
import inspect
from pathlib import Path
from types import ModuleType

from src.services.data import augment, format, ingest, report, speech_plan, split, transform, validate
from src.services.data.paths import manifest_path
from src.utils.config import load_yaml
from src.utils.hashing import fingerprint
from src.utils.io import read_json, write_json
from src.utils.logging import get_logger

STAGES: dict[str, ModuleType] = {
    "ingest": ingest,
    "validate": validate,
    "transform": transform,
    "split": split,
    "augment": augment,
    "format": format,
    "speech_plan": speech_plan,
    "report": report,
}
ALWAYS_RUN = {"report"}

log = get_logger("pipeline.data")


def stage_fingerprint(name: str, stage: ModuleType, cfg: dict) -> str:
    code = [Path(inspect.getsourcefile(stage))]
    return fingerprint(stage.inputs(cfg) + code, {"stage": cfg.get(name), "tools": cfg["tools_config"]})


def is_cached(name: str, stage: ModuleType, cfg: dict, fp: str) -> bool:
    path = manifest_path(cfg, name)
    if name in ALWAYS_RUN or not path.exists():
        return False
    return read_json(path).get("fingerprint") == fp and all(p.exists() for p in stage.outputs(cfg))


def run_stage(name: str, cfg: dict, force: bool = False) -> dict:
    stage = STAGES[name]
    fp = stage_fingerprint(name, stage, cfg)
    if not force and is_cached(name, stage, cfg, fp):
        log.info("%-9s skipped (inputs unchanged)", name)
        return read_json(manifest_path(cfg, name))
    manifest = {"stage": name, "fingerprint": fp, **stage.run(cfg), "outputs": [str(p) for p in stage.outputs(cfg)]}
    write_json(manifest_path(cfg, name), manifest)
    log.info("%-9s in=%-6s out=%-6s dropped=%s", name, manifest["in"], manifest["out"], manifest["rejected"])
    return manifest


def run_pipeline(cfg: dict, only: str | None = None, force: bool = False) -> list[dict]:
    names = [only] if only else list(STAGES)
    return [run_stage(name, cfg, force) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--stage", choices=list(STAGES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(load_yaml(args.config), args.stage, args.force)


if __name__ == "__main__":
    main()
