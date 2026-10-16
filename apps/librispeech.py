"""Download LibriSpeech on Modal CPU containers and convert it to int16 shards.

    uv run modal run -m apps.librispeech                        # every parquet file not done yet (HF, fast)
    uv run modal run -m apps.librispeech --subset dev-clean
    uv run modal run -m apps.librispeech --source openslr       # original tar.gz, streamed (slow)

Each source parquet file is also kept on the Volume (data/librispeech/raw/) for the local copy.
"""
import modal

from apps.common import VOLUME_PATH, volume
from src.utils.config import load_yaml

CFG = load_yaml("configs/librispeech.yaml")
app = modal.App("make-model-omni-librispeech")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.5.4", "soundfile==0.14.0", "pyyaml==6.0.3", "pyarrow==25.0.1", "huggingface_hub==2.0.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .workdir("/root")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src", "apps")
)
FUNCTION = dict(image=image, cpu=CFG["modal"]["cpu"], memory=CFG["modal"]["memory_mb"], volumes={VOLUME_PATH: volume},
                timeout=CFG["modal"]["timeout_seconds"], retries=modal.Retries(max_retries=2, initial_delay=30.0))


@app.function(max_containers=CFG["modal"]["max_containers"], **FUNCTION)
def fetch_parquet(subset: str, filename: str, file_index: int) -> dict:
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from src.services.librispeech import convert_parquet, part_done_path

    audio_dir, h = CFG["paths"]["volume_audio_dir"], CFG["hf"]
    done = part_done_path(audio_dir, subset, file_index)
    if done.exists():
        return {"subset": subset, "file": filename, "skipped": True}
    local = hf_hub_download(h["repo"], filename, repo_type="dataset", revision=h["revision"], cache_dir="/tmp/hf")
    raw = Path(CFG["paths"]["volume_raw_dir"]) / subset / Path(filename).name
    raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, raw)
    summary = convert_parquet(local, audio_dir, subset, file_index, CFG["shards"]["size"], CFG["audio"]["sample_rate"], CFG["audio"]["max_seconds"])
    done.write_text(f"{summary}\n")
    volume.commit()
    return {**summary, "file": filename}


@app.function(**FUNCTION)
def fetch_tar(subset: str) -> dict:
    from src.services.librispeech import convert, mark_done, stream_members, subset_done

    audio_dir = CFG["paths"]["volume_audio_dir"]
    if subset_done(audio_dir, subset):
        return {"subset": subset, "skipped": True}
    errors = []
    for mirror in CFG["mirrors"]:
        try:
            summary = convert(stream_members(f"{mirror}/{subset}.tar.gz"), audio_dir, subset, CFG["shards"]["size"],
                              CFG["audio"]["sample_rate"], CFG["audio"]["max_seconds"])
            mark_done(audio_dir, subset, summary)
            volume.commit()
            return summary
        except OSError as error:   # network trouble: try the next mirror
            errors.append(f"{mirror}: {error}")
    raise RuntimeError("; ".join(errors))


def parquet_jobs(subsets: list[str]) -> list[tuple[str, str, int]]:
    from huggingface_hub import HfApi

    h = CFG["hf"]
    files = HfApi().list_repo_files(h["repo"], repo_type="dataset", revision=h["revision"])
    jobs = []
    for subset in subsets:
        folder = h["folders"][subset] + "/"
        jobs += [(subset, f, i) for i, f in enumerate(sorted(f for f in files if f.startswith(folder) and f.endswith(".parquet")))]
    return jobs


@app.local_entrypoint()
def main(subset: str = "", source: str = "") -> None:
    subsets = [subset] if subset else CFG["subsets"]
    if (source or CFG["source"]) == "openslr":
        results = fetch_tar.map(subsets, order_outputs=False)
    else:
        jobs = parquet_jobs(subsets)
        print(f"{len(jobs)} parquet files")
        results = fetch_parquet.starmap(jobs, order_outputs=False)
    for result in results:
        print(result)
