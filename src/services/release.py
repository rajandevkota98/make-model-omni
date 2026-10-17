"""Build the Hugging Face model repo from the step 6 checkpoint, then check it.

  build   LoRA merged into the dequantized NF4 base (step 7's served weights) saved as a plain Qwen3
          checkpoint, + the tokenizer with <|audio_pad|>, the audio adapter, omni_config.json, model card, code
  parity  the standalone omni_assistant.py vs the training-code pipeline on the same clips (GPU)
  cpu     omni_assistant.py on CPU only, as the free Space runs it: load time and seconds per turn
"""
import json
import shutil
import sys
import time
from pathlib import Path

from src.utils.config import load_yaml
from src.utils.logging import get_logger

log = get_logger("release")
LORA_CARD = """# Unmerged LoRA (r=32, alpha=64, all linear layers)

The spoken-tool LoRA before merging. It was trained on {base} loaded in 4-bit NF4 (double quant, bf16 compute);
apply it to that 4-bit base for training-identical results. The model at the repo root already has it merged.
"""


def omni_config(cfg: dict, rcfg: dict, placeholder_id: int) -> dict:
    """Everything omni_assistant.py needs that is not in the Qwen3 / tokenizer files."""
    from src.services.data.format import prompt_tools

    data_cfg = load_yaml("configs/data.yaml")
    enc, ad = cfg["encoder"], cfg["adapter"]
    return {"placeholder": cfg["prompt"]["placeholder"], "placeholder_id": placeholder_id,
            "encoder": {k: enc[k] for k in ("id", "revision", "dim", "frames_per_second", "max_frames")},
            "adapter": {k: ad[k] for k in ("type", "frame_stack", "hidden", "out_dim")},
            "max_seconds": cfg["data"]["max_seconds"], "system_prompt": data_cfg["format"]["system_prompt"],
            "tools": prompt_tools(load_yaml(data_cfg["tools_config"]), data_cfg["format"]),
            "stop_ids": cfg["tokens"]["stop_ids"], "max_new_tokens": rcfg["max_new_tokens"],
            "base_models": {"llm": f"{cfg['model']['id']}@{cfg['model']['revision']}", "encoder": f"{enc['id']}@{enc['revision']}"}}


def build_repo(rcfg: dict) -> dict:
    from src.components.llm import load_tokenizer, merge_lora
    from src.components.tokens import add_placeholder
    from src.services.spoken_runs import full_config

    cfg = full_config(load_yaml(rcfg["spoken_config"]))
    ckpt, out = Path(rcfg["checkpoint"]), Path(rcfg["out_dir"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    merged = merge_lora(cfg["model"], str(ckpt / "lora"), from_nf4=True)
    if getattr(merged.config, "quantization_config", None) is not None:
        del merged.config.quantization_config   # weights are plain bf16 now; loaders must not expect bnb
    tokenizer = load_tokenizer(cfg["model"])
    placeholder_id = add_placeholder(tokenizer, cfg["prompt"]["placeholder"], merged.get_input_embeddings().weight.shape[0])
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    shutil.copy(ckpt / "adapter.safetensors", out / "audio_adapter.safetensors")
    shutil.copytree(ckpt / "lora", out / "lora")
    (out / "lora" / "README.md").write_text(LORA_CARD.format(base=f"{cfg['model']['id']}@{cfg['model']['revision']}"))
    (out / "omni_config.json").write_text(json.dumps(omni_config(cfg, rcfg, placeholder_id), indent=2) + "\n")
    for path in Path(rcfg["model_files"]).iterdir():
        shutil.copy(path, out / path.name)
    files = {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()}
    log.info("built %s: %d files, %.2f GB", out, len(files), sum(files.values()) / 1e9)
    return {"out_dir": str(out), "files": files, "placeholder_id": placeholder_id}


def import_release(out_dir: str):
    sys.path.insert(0, out_dir)
    import omni_assistant

    return omni_assistant


def reference_turns(rcfg: dict, clips: list[bytes], texts: list[str]) -> list[str]:
    """The training-code pipeline (playground path) on the same inputs."""
    from src.services.playground import AUDIO, load_playground, run_turn

    pg = load_playground({"spoken_config": rcfg["spoken_config"], "checkpoint": rcfg["checkpoint"],
                          "weights": rcfg["reference_weights"], "generate": {"max_new_tokens": rcfg["max_new_tokens"]},
                          "speech": {"tts_config": "configs/speech.yaml", "voice": "af_heart", "speed": 1.0}})
    audio = [run_turn(pg, "tools", [{"role": "user", "content": AUDIO}], [pcm])["raw"] for pcm in clips]
    return audio + [run_turn(pg, "tools", [{"role": "user", "content": t}], [])["raw"] for t in texts]


def release_turns(out_dir: str, clips: list[bytes], texts: list[str], device: str) -> tuple[list[dict], dict]:
    import numpy as np

    om = import_release(out_dir)
    started = time.time()
    assistant = om.load(out_dir, device=device)
    timing = {"load_s": round(time.time() - started, 1), "turn_s": []}
    results = []
    for pcm in clips:
        t = time.time()
        results.append(om.respond(assistant, [{"role": "user", "content": om.AUDIO}],
                                  [np.frombuffer(pcm, np.int16).astype(np.float32) / 32767]))
        timing["turn_s"].append(round(time.time() - t, 2))
    for text in texts:
        t = time.time()
        results.append(om.respond(assistant, [{"role": "user", "content": text}]))
        timing["turn_s"].append(round(time.time() - t, 2))
    return results, timing


def parity(rcfg: dict, clips: list[bytes], texts: list[str]) -> dict:
    """Same greedy decoding on the same weights → the replies should match exactly."""
    import torch

    from src.utils.metrics import strip_end

    reference = [strip_end(r) for r in reference_turns(rcfg, clips, texts)]
    torch.cuda.empty_cache()
    release, timing = release_turns(rcfg["out_dir"], clips, texts, "cuda")
    rows = [{"reference": ref, "release": rel["raw"].strip(), "match": ref == rel["raw"].strip()} for ref, rel in zip(reference, release)]
    return {"matches": sum(r["match"] for r in rows), "n": len(rows), "rows": rows, "timing": timing}
