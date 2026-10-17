"""Build and check the Hugging Face release on Modal. Nothing here uploads to Hugging Face.

    uv run modal run -m apps.release --mode build    # merged weights + adapter + card → /vol/release/…
    uv run modal run -m apps.release --mode parity   # release code vs training pipeline (GPU)
    uv run modal run -m apps.release --mode cpu      # release code + Space app on CPU, like the free Space
    uv run modal run -m apps.release --mode publish  # upload model repo + Space (public), using the `huggingface` secret
"""
import json
from pathlib import Path

import modal

from apps.common import VOLUME_PATH, playground_image, volume
from src.utils.config import load_yaml

CFG = load_yaml("configs/release.yaml")
M = CFG["modal"]
app = modal.App("make-model-omni-release")
image = playground_image.add_local_dir(CFG["model_files"], f"/root/{CFG['model_files']}").add_local_dir(
    CFG["space_files"], f"/root/{CFG['space_files']}")
space_image = (modal.Image.debian_slim(python_version="3.12").apt_install("espeak-ng")
               .pip_install_from_requirements(f"{CFG['space_files']}/requirements.txt")
               .uv_pip_install("pyyaml==6.0.3")   # only for this Modal entrypoint's own imports, not the Space
               .workdir("/root").add_local_dir("configs", "/root/configs").add_local_python_source("src", "apps", "client")
               .add_local_dir(CFG["space_files"], "/root/space"))
TEXTS = ["turn the volume down a bit", "what's the weather tomorrow in paris", "delete all my files"]


@app.function(image=image, gpu=M["gpu"], volumes={VOLUME_PATH: volume}, memory=16384, timeout=1800)
def build() -> dict:
    from src.services.release import build_repo

    volume.reload()
    summary = build_repo(CFG)
    volume.commit()
    return summary


@app.function(image=image, gpu=M["gpu"], volumes={VOLUME_PATH: volume}, memory=16384, timeout=1800)
def parity(clips: list[bytes]) -> dict:
    from src.services.release import parity as run_parity

    volume.reload()
    return run_parity(CFG, clips, TEXTS)


@app.function(image=space_image, cpu=M["space_cpu"], memory=M["space_memory_mb"], volumes={VOLUME_PATH: volume}, timeout=3600)
def cpu(clips: list[bytes]) -> dict:
    """The Space's own code and requirements, on Space-sized CPU, loading the built repo from the Volume."""
    import os
    import sys
    import time

    import numpy as np

    os.environ["OMNI_MODEL"] = CFG["out_dir"]
    sys.path.insert(0, CFG["out_dir"])
    sys.path.insert(0, "/root/space")
    import app as space_app

    started = time.time()
    state = space_app.load_all()
    out = {"load_s": round(time.time() - started, 1), "turns": []}
    space_app.build_ui(state)   # constructs the Gradio Blocks without launching
    for pcm in clips[:3]:
        wave = np.frombuffer(pcm, np.int16).astype(np.float32) / 32767
        t = time.time()
        turn = space_app.run_turn(state, [], (16000, (wave * 32767).astype(np.int16)), "")
        out["turns"].append({"s": round(time.time() - t, 2), "reply": turn["reply"], "calls": turn["tool_calls"],
                             "speech_s": turn["speech_seconds"], "tts_s": turn["tts_s"]})
    return out


@app.function(image=image, volumes={VOLUME_PATH: volume}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def publish() -> list[str]:
    """Model first (the Space downloads it at startup), then the Space."""
    from src.services.hub import publish as upload

    volume.reload()
    message = "Release omni-assistant-0.6b"
    return [upload(f"{CFG['hf_user']}/{CFG['model_repo']}", Path(CFG["out_dir"]), "model", False, message),
            upload(f"{CFG['hf_user']}/{CFG['space_repo']}", Path(f"/root/{CFG['space_files']}"), "space", False, message)]


@app.local_entrypoint()
def main(mode: str = "build") -> None:
    from client.audio_io import wav_to_pcm

    clips = [wav_to_pcm(str(p)) for p in sorted(Path(CFG["parity_clips"]).glob("test-*.wav"))]
    if mode == "build":
        summary = build.remote()
        print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))
        print("\n".join(f"  {size / 1e6:9.1f} MB  {name}" for name, size in summary["files"].items()))
    elif mode == "parity":
        result = parity.remote(clips)
        for row in result["rows"]:
            print(("✓ " if row["match"] else "✗ ") + row["release"].replace("\n", " ")[:120])
            if not row["match"]:
                print("  ref: " + row["reference"].replace("\n", " ")[:120])
        print(f"{result['matches']}/{result['n']} identical · {result['timing']}")
    elif mode == "publish":
        print("\n".join(publish.remote()))
    elif mode == "cpu":
        print(json.dumps(cpu.remote(clips), indent=2))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
