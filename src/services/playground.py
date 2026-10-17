"""Playground turns: typed or spoken user messages → what the model says and which tools it would call.

Nothing is executed. With speak=True the reply text also comes back as Kokoro audio. Two modes:
  tools       the assistant prompt (system + tools), exactly as in training; any user turn can be audio
  transcribe  the step 5 prompt, "Transcribe the audio." — shows what the ears heard
"""
import time

import numpy as np

from src.utils.config import load_yaml
from src.utils.metrics import parse_turn

AUDIO = "audio"   # a user message whose content is this marker is filled with the next waveform


def load_playground(pcfg: dict) -> dict:
    from src.services.data.format import prompt_tools
    from src.services.spoken_runs import full_config, load_parts

    cfg = full_config(load_yaml(pcfg["spoken_config"]))
    cfg["eval"]["batch_size"] = 1
    cfg["eval"]["max_new_tokens"] = pcfg["generate"]["max_new_tokens"]
    parts = load_parts(cfg, f"{pcfg['checkpoint']}/lora", pcfg["checkpoint"], train=False)
    data_cfg = load_yaml("configs/data.yaml")
    return {"cfg": cfg, "parts": parts, "tools": prompt_tools(load_yaml(data_cfg["tools_config"]), data_cfg["format"]),
            "system_prompt": data_cfg["format"]["system_prompt"], "voice": load_voice(pcfg["speech"])}


def load_voice(speech_cfg: dict) -> dict:
    from src.services.tts import load_engine

    tts_cfg = load_yaml(speech_cfg["tts_config"])["tts"]
    return {**speech_cfg, "engine": load_engine(tts_cfg), "sample_rate": tts_cfg["sample_rate"]}


def speak_reply(voice: dict, text: str) -> dict:
    """Reply text → int16 PCM at the TTS rate. Empty text → no audio."""
    from src.services.tts import synthesize
    from src.utils.audio import to_int16

    started = time.time()
    audio = synthesize(voice["engine"], text, voice["voice"], voice["speed"]) if text.strip() else np.zeros(0, np.float32)
    return {"speech": to_int16(audio).tobytes(), "speech_rate": voice["sample_rate"],
            "speech_ms": round((time.time() - started) * 1000), "speech_seconds": round(len(audio) / voice["sample_rate"], 2)}


def pcm_to_float(pcm: bytes) -> np.ndarray:
    """16 kHz mono int16 PCM → float32 in [-1, 1]."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767


def tools_item(pg: dict, messages: list[dict], waves: list[np.ndarray]) -> dict:
    from src.components.merge import audio_tokens
    from src.services.data.format import encode, render
    from src.services.data.spoken_format import split_segments

    parts, cfg = pg["parts"], pg["cfg"]
    placeholder = cfg["prompt"]["placeholder"]
    chat = [{"role": "system", "content": pg["system_prompt"]},
            *({**m, "content": placeholder} if m["role"] == "user" and m["content"] == AUDIO else m for m in messages)]
    segments = split_segments(encode(parts["tokenizer"], render(parts["tokenizer"], chat, pg["tools"], True)), parts["placeholder_id"])
    if len(segments) != len(waves) + 1:
        raise ValueError(f"{len(segments) - 1} audio messages but {len(waves)} recordings")
    ids = list(segments[0])
    for wave, segment in zip(waves, segments[1:]):
        ids += [parts["placeholder_id"]] * audio_tokens(len(wave), cfg["encoder"], cfg["adapter"]["frame_stack"]) + list(segment)
    return {"input_ids": ids, "prompt_len": len(ids), "waveforms": waves}


def transcribe_item(pg: dict, wave: np.ndarray) -> dict:
    from src.components.merge import audio_tokens

    p, cfg = pg["parts"]["asr_prompt"], pg["cfg"]
    ids = p["before"] + [p["placeholder_id"]] * audio_tokens(len(wave), cfg["encoder"], cfg["adapter"]["frame_stack"]) + p["after"]
    return {"input_ids": ids, "prompt_len": len(ids), "waveforms": [wave]}


def run_turn(pg: dict, mode: str, messages: list[dict], audio: list[bytes], speak: bool = False) -> dict:
    """messages: chat without the system prompt; user content == "audio" takes the next recording."""
    from src.services.spoken_runs import generate_items

    started = time.time()
    waves = [pcm_to_float(a) for a in audio]
    item = transcribe_item(pg, waves[0]) if mode == "transcribe" else tools_item(pg, messages, waves)
    raw = generate_items(pg["parts"], [item], pg["cfg"])[0]
    turn = parse_turn(raw)
    result = {"raw": raw.strip(), "reply": turn["reply"], "tool_calls": turn["calls"], "json_ok": turn["json_ok"],
              "audio_seconds": [round(len(w) / 16000, 2) for w in waves], "prompt_tokens": item["prompt_len"],
              "ms": round((time.time() - started) * 1000)}
    return {**result, **speak_reply(pg["voice"], turn["reply"])} if speak and mode == "tools" else result
