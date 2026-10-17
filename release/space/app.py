"""omni-assistant demo Space: speak (or type) a desktop command; the model hears the audio directly and
answers with a short spoken reply + the tool calls it would make. Nothing is executed here.

Runs on the free CPU tier: Qwen3-0.6B (LoRA merged) + Whisper-small encoder + audio adapter in float32,
and Kokoro-82M reading the reply aloud.
"""
import os
import sys
import time
from math import gcd

import gradio as gr
import numpy as np
import torch

MODEL = os.environ.get("OMNI_MODEL", "Rajan98/omni-assistant-0.6b")
VOICE = os.environ.get("OMNI_VOICE", "af_heart")
TTS_REPO, TTS_RATE = "hexgrad/Kokoro-82M", 24000
EXAMPLES = ["open the browser and go to wikipedia", "turn the volume down a bit", "what's the weather tomorrow in paris",
            "set a timer for ten minutes for the pasta", "type hello world", "delete all my files"]
torch.set_num_threads(max(1, os.cpu_count() or 1))


# ── loading ─────────────────────────────────────────────────────────────────

def import_model_code(source: str):
    from huggingface_hub import snapshot_download

    root = source if os.path.isdir(source) else snapshot_download(source)
    sys.path.insert(0, root)
    import omni_assistant

    return omni_assistant, root


def load_all() -> dict:
    from kokoro import KPipeline

    om, root = import_model_code(MODEL)
    return {"om": om, "assistant": om.load(root, device="cpu"), "tts": KPipeline(lang_code=VOICE[0], repo_id=TTS_REPO)}


# ── one turn ────────────────────────────────────────────────────────────────

def to_16k(audio: tuple[int, np.ndarray]) -> np.ndarray:
    """Gradio (rate, int16 or float array, mono or stereo) → float32 mono at 16 kHz."""
    from scipy.signal import resample_poly

    rate, data = audio
    wave = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        wave /= np.iinfo(data.dtype).max
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    g = gcd(int(rate), 16000)
    return wave if rate == 16000 else resample_poly(wave, 16000 // g, int(rate) // g).astype(np.float32)


def speak(state: dict, text: str) -> np.ndarray:
    chunks = [r.audio.numpy() for r in state["tts"](text, voice=VOICE) if r.audio is not None] if text.strip() else []
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def is_open_question(result: dict) -> bool:
    return not result["tool_calls"] and result["reply"].rstrip().endswith("?")


def run_turn(state: dict, memory: list, audio, text: str) -> dict:
    """memory: [(message, waveform or None)] kept only while the assistant is waiting for an answer."""
    om = state["om"]
    if audio is not None:
        message, wave = {"role": "user", "content": om.AUDIO}, to_16k(audio)
    elif text and text.strip():
        message, wave = {"role": "user", "content": text.strip()}, None
    else:
        raise gr.Error("Record something or type a command first.")
    turns = [*memory, (message, wave)]
    started = time.time()
    result = om.respond(state["assistant"], [m for m, _ in turns], [w for _, w in turns if w is not None])
    model_s = time.time() - started
    started = time.time()
    speech = speak(state, result["reply"])
    return {**result, "model_s": round(model_s, 2), "tts_s": round(time.time() - started, 2), "speech": speech,
            "speech_seconds": round(len(speech) / TTS_RATE, 2),
            "memory": [*turns, (om.assistant_message(result), None)] if is_open_question(result) else []}


# ── UI ──────────────────────────────────────────────────────────────────────

INTRO = """# 🎙️ omni-assistant · a 0.6B model that hears desktop commands
Speak (or type) something like *"open the browser and go to wikipedia"*. The model hears the **audio directly**
(no speech-to-text step), writes a short reply and the **tool calls** it would make on a Linux desktop. Nothing is run here.
Qwen3-0.6B + Whisper-small encoder + a trained audio adapter, fine-tuned for 12 desktop tools · replies read by Kokoro-82M.
Free CPU: expect a few seconds per turn. [Model card](https://huggingface.co/{model})"""


def build_ui(state: dict) -> gr.Blocks:
    def on_send(audio, text, memory, chat):
        turn = run_turn(state, memory, audio, text)
        said = "🎤 (voice message)" if audio is not None else text.strip()
        calls = turn["tool_calls"] or "no tool call"
        chat = [*chat, {"role": "user", "content": said}, {"role": "assistant", "content": turn["reply"] or "(no spoken reply)"}]
        timing = f"model {turn['model_s']} s · voice {turn['tts_s']} s" + (" · waiting for your answer" if turn["memory"] else "")
        voice = (TTS_RATE, turn["speech"]) if len(turn["speech"]) else None
        return chat, calls, voice, timing, turn["memory"], None, ""

    with gr.Blocks(title="omni-assistant") as demo:
        gr.Markdown(INTRO.format(model=MODEL))
        memory = gr.State([])
        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak a command")
                box = gr.Textbox(label="…or type it", placeholder="turn the volume down a bit")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    reset = gr.Button("New conversation")
                gr.Examples(EXAMPLES, inputs=box)
            with gr.Column():
                chat = gr.Chatbot(label="Conversation", height=260)
                calls = gr.JSON(label="Tool calls it would make")
                voice = gr.Audio(label="Spoken reply", autoplay=True, interactive=False)
                timing = gr.Markdown()
        outputs = [chat, calls, voice, timing, memory, mic, box]
        send.click(on_send, [mic, box, memory, chat], outputs)
        box.submit(on_send, [mic, box, memory, chat], outputs)
        reset.click(lambda: ([], None, None, "", [], None, ""), None, outputs)
    return demo


if __name__ == "__main__":
    build_ui(load_all()).queue(max_size=8).launch()
