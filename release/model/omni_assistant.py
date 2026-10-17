"""omni-assistant: Qwen3-0.6B that hears speech directly and answers with a short reply + tool calls.

Standalone inference (torch + transformers; no bitsandbytes, CPU or GPU):

    from omni_assistant import AUDIO, load, respond
    assistant = load("<this repo id or a local folder>")
    result = respond(assistant, [{"role": "user", "content": AUDIO}], [waveform])   # float32, 16 kHz mono
    result = respond(assistant, [{"role": "user", "content": "turn the volume down"}])
    # → {"reply": "Turning it down.", "tool_calls": [{"name": "set_volume", "arguments": {"action": "down"}}], ...}

How it works: Whisper-small's encoder turns the audio into 50 frames per second; 4 neighbouring frames
are glued together (12.5 per second) and a small trained MLP maps each one to a vector the size of a
Qwen3 word embedding. Those vectors replace the <|audio_pad|> slots in the chat prompt, and Qwen3
(with the fine-tuned LoRA merged in) writes the reply. Nothing here runs the tools; that is up to you.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

AUDIO = "audio"            # a user message with exactly this content is filled with the next waveform
SAMPLE_RATE = 16000
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class MLPAdapter(nn.Module):
    """Stacked Whisper frames → Qwen3-sized vectors, RMS-normalised to the scale of word embeddings."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.norm = nn.RMSNorm(out_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(frames))


# ── loading ─────────────────────────────────────────────────────────────────

def resolve(source: str) -> Path:
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(source))


def load(source: str, device: str | None = None, dtype: torch.dtype | None = None) -> dict:
    """Everything a turn needs. Default: bf16 on CUDA, float32 on CPU."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    root = resolve(source)
    cfg = json.loads((root / "omni_config.json").read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    enc, ad = cfg["encoder"], cfg["adapter"]
    llm = AutoModelForCausalLM.from_pretrained(root, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    whisper = WhisperModel.from_pretrained(enc["id"], revision=enc["revision"], dtype=dtype).get_encoder().to(device).eval()
    adapter = MLPAdapter(enc["dim"] * ad["frame_stack"], ad["hidden"], ad["out_dim"])
    adapter.load_state_dict(load_file(str(root / "audio_adapter.safetensors")))
    return {"cfg": cfg, "device": device, "dtype": dtype, "llm": llm, "adapter": adapter.to(device, torch.float32).eval(),
            "tokenizer": AutoTokenizer.from_pretrained(root, padding_side="left"), "whisper": whisper,
            "extractor": WhisperFeatureExtractor.from_pretrained(enc["id"], revision=enc["revision"])}


# ── audio → vectors ─────────────────────────────────────────────────────────

def encoder_frames(n_samples: int, enc: dict) -> int:
    """Whisper reads a padded 30 s window; keep only the frames that cover real audio."""
    return min(enc["max_frames"], math.ceil(n_samples * enc["frames_per_second"] / SAMPLE_RATE))


def audio_tokens(n_samples: int, cfg: dict) -> int:
    return math.ceil(encoder_frames(n_samples, cfg["encoder"]) / cfg["adapter"]["frame_stack"])


def frame_stack(frames: torch.Tensor, k: int) -> torch.Tensor:
    """[T, D] → [ceil(T / k), k·D]; the last group is zero-padded."""
    t, d = frames.shape
    return nn.functional.pad(frames, (0, 0, 0, (-t) % k)).reshape(-1, k * d)


@torch.no_grad()
def hear(assistant: dict, waveforms: list[np.ndarray]) -> torch.Tensor:
    """Waveforms → all audio vectors, in order: [total audio tokens, Qwen3 hidden size]."""
    cfg, device = assistant["cfg"], assistant["device"]
    clips = [np.asarray(w, np.float32)[: cfg["max_seconds"] * SAMPLE_RATE] for w in waveforms]
    mel = assistant["extractor"](clips, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    hidden = assistant["whisper"](mel.to(device, assistant["dtype"])).last_hidden_state
    stacked = torch.cat([frame_stack(hidden[i, : encoder_frames(len(c), cfg["encoder"])], cfg["adapter"]["frame_stack"])
                         for i, c in enumerate(clips)])
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        return assistant["adapter"](stacked.float())


# ── prompt → reply ──────────────────────────────────────────────────────────

def prompt_ids(assistant: dict, messages: list[dict], waveforms: list[np.ndarray]) -> list[int]:
    """Chat template exactly as in training (system prompt, tools=, no thinking); one slot per audio token."""
    cfg, tok = assistant["cfg"], assistant["tokenizer"]
    pid = tok.convert_tokens_to_ids(cfg["placeholder"])
    chat = [{"role": "system", "content": cfg["system_prompt"]},
            *({**m, "content": cfg["placeholder"]} if m["role"] == "user" and m["content"] == AUDIO else m for m in messages)]
    text = tok.apply_chat_template(chat, tools=cfg["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if ids.count(pid) != len(waveforms):
        raise ValueError(f"{ids.count(pid)} audio messages but {len(waveforms)} waveforms")
    out = []
    for token in ids:
        out += [pid] * audio_tokens(len(waveforms.pop(0)), cfg) if token == pid else [token]
    return out


def parse_turn(text: str) -> dict:
    """Generated text → spoken reply + tool calls (each <tool_call> block holds one JSON object)."""
    calls, json_ok = [], True
    for raw in TOOL_CALL.findall(text):
        try:
            call = json.loads(raw)
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
        except (json.JSONDecodeError, KeyError, TypeError):
            json_ok = False
    reply = TOOL_CALL.sub("", text).replace("<tool_call>", "").strip()
    return {"reply": reply, "tool_calls": calls, "json_ok": json_ok and ("<tool_call>" not in text or bool(calls))}


@torch.no_grad()
def generate(assistant: dict, messages: list[dict], waveforms: list[np.ndarray] | None = None) -> str:
    waveforms = list(waveforms or [])
    llm, tok, cfg, device = assistant["llm"], assistant["tokenizer"], assistant["cfg"], assistant["device"]
    ids = torch.tensor([prompt_ids(assistant, messages, list(waveforms))], device=device)
    embeds = llm.get_input_embeddings()(ids)
    if waveforms:
        mask = (ids == tok.convert_tokens_to_ids(cfg["placeholder"])).unsqueeze(-1)
        embeds = embeds.masked_scatter(mask, hear(assistant, waveforms).to(embeds.dtype))
    out = llm.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), max_new_tokens=cfg["max_new_tokens"],
                       do_sample=False, eos_token_id=cfg["stop_ids"], pad_token_id=tok.pad_token_id, use_cache=True)
    return tok.decode(out[0], skip_special_tokens=True)


def respond(assistant: dict, messages: list[dict], waveforms: list[np.ndarray] | None = None) -> dict:
    """messages: the chat without the system prompt. Tool results go back as {"role": "tool", "content": json}."""
    raw = generate(assistant, messages, waveforms)
    return {**parse_turn(raw), "raw": raw}


def assistant_message(result: dict) -> dict:
    """A turn's result as the assistant message to append before the next user turn."""
    message = {"role": "assistant", "content": result["reply"]}
    if result["tool_calls"]:
        message["tool_calls"] = [{"type": "function", "function": c} for c in result["tool_calls"]]
    return message
