"""Publish the release to Hugging Face: the model repo (from the local mirror) and the demo Space.

    huggingface-cli login                                   # once; a token with write access
    uv run modal volume get --force omni-lab release/omni-assistant-0.6b data/release/
    uv run python -m scripts.publish_hf --dry-run           # list what would be uploaded
    uv run python -m scripts.publish_hf --model --space     # upload (public) from this laptop
    uv run modal run -m apps.release --mode publish         # same, from Modal (fast; needs the `huggingface` secret)

Both repos are created public. Re-running updates them (a new commit on the Hub).
"""
import argparse
from pathlib import Path

from src.services.hub import publish
from src.utils.config import load_yaml

CFG = load_yaml("configs/release.yaml")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--space", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--message", default="Release omni-assistant-0.6b")
    opts = parser.parse_args()
    model, space = (opts.model, opts.space) if opts.model or opts.space else (True, True)
    if model:
        folder = Path(CFG["local_dir"])
        if not (folder / "model.safetensors").exists():
            raise SystemExit(f"{folder} has no model.safetensors; pull it from the Volume first (see --help)")
        publish(f"{CFG['hf_user']}/{CFG['model_repo']}", folder, "model", opts.dry_run, opts.message)
    if space:
        publish(f"{CFG['hf_user']}/{CFG['space_repo']}", Path(CFG["space_files"]), "space", opts.dry_run, opts.message)


if __name__ == "__main__":
    main()
