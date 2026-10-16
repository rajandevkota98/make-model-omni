"""Keep a local copy of the LibriSpeech source parquet files (~62 GB) in data/librispeech/raw/<subset>/.

They are fetched from the Modal Volume (kept there by apps.librispeech), which is ~5× faster from here
than Hugging Face or openslr. Safe to re-run: subsets already complete locally are skipped.

    uv run python -m scripts.download_librispeech             # every subset
    uv run python -m scripts.download_librispeech dev-clean
"""
import json
import subprocess
import sys
from pathlib import Path

from src.utils.config import load_yaml

VOLUME = "omni-lab"


def remote_files(remote_dir: str) -> dict[str, int]:
    out = subprocess.run(["modal", "volume", "ls", "--json", VOLUME, remote_dir], capture_output=True, text=True, check=True).stdout
    return {Path(f["filename"]).name: f["size"] for f in json.loads(out) if f["type"] == "file"}


def pull(subset: str, cfg: dict) -> bool:
    remote_dir = f"{cfg['paths']['volume_raw_dir'].removeprefix('/vol/')}/{subset}"
    local_dir = Path(cfg["paths"]["local_raw_dir"]) / subset
    local_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, size in sorted(remote_files(remote_dir).items()):
        target = local_dir / name
        if target.exists() and target.stat().st_size == size:
            continue
        ok &= subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{remote_dir}/{name}", str(target)]).returncode == 0
    print(f"{subset}: {sum(p.stat().st_size for p in local_dir.glob('*.parquet')) / 1e9:.1f} GB local")
    return ok


def main() -> None:
    cfg = load_yaml("configs/librispeech.yaml")
    failed = [s for s in (sys.argv[1:] or cfg["subsets"]) if not pull(s, cfg)]
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
