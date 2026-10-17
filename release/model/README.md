---
license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: audio-text-to-text
base_model:
  - Qwen/Qwen3-0.6B
  - openai/whisper-small
tags: [speech, audio, tool-calling, function-calling, voice-assistant, qwen3, whisper, lora, linux]
---

# omni-assistant-0.6b

A small voice assistant model for a Linux desktop. You say *"open the browser and go to Wikipedia"*; the model
**hears the audio directly**, with no separate speech-to-text step, and answers with a short reply plus a
Qwen3-style `<tool_call>`. Your own executor runs the call.

- **In:** speech (16 kHz) and/or text, in a multi-turn chat
- **Out:** a short reply meant to be spoken, plus zero or more tool calls from 12 desktop tools
- **Size:** Qwen3-0.6B (merged LoRA) + a frozen Whisper-small encoder (loaded from `openai/whisper-small`) +
  an 8.4M-parameter audio adapter, about 0.7B parameters in all
- **Not included:** speech output. The project's desktop client reads replies aloud with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
  a separate TTS model; this model only writes text.

Code, training pipeline and the desktop voice client: [github.com/rajandevkota98/make-model-omni](https://github.com/rajandevkota98/make-model-omni)

## How it works

```
speech ─► Whisper-small encoder (frozen) ─► 50 frames/s ─► stack 4 ─► MLP adapter (trained) ─┐
                                                                                            ├─► Qwen3-0.6B + merged LoRA ─► reply + <tool_call>
text / chat template (system prompt + tools) ─► embeddings, with <|audio_pad|> slots ───────┘
```

Each `<|audio_pad|>` slot (12.5 per second of audio) is replaced by one adapter vector. The adapter ends in an
RMSNorm whose gain matches the length of Qwen3's word embeddings, so audio arrives at the same scale as text.
Nothing was trained from scratch: Qwen3 and Whisper are pretrained, the adapter is the only new layer.

| Stage | What trained | Data |
|---|---|---|
| Typed tool SFT | LoRA r=32 on all linear layers (QLoRA, 4-bit base) | ~18K synthetic desktop commands + built multi-turn samples |
| Audio alignment | audio adapter only (Qwen3, Whisper frozen) | LibriSpeech 960 h + TTS-voiced commands, "Transcribe the audio." |
| Spoken tool SFT | LoRA + adapter together | commands voiced by 8 Kokoro voices with noise, 10% typed replay, 5% transcription replay |

Release weights: the LoRA merged into the **dequantized NF4 base it was trained on**, saved as plain bf16
Qwen3. Merging into the original bf16 weights instead cost about 1.3 points on noisy audio. The unmerged LoRA
is in `lora/` (it targets the 4-bit NF4 base).

## Use

```python
from omni_assistant import AUDIO, load, respond      # omni_assistant.py ships in this repo
assistant = load("Rajan98/omni-assistant-0.6b")   # bf16 on GPU, float32 on CPU

result = respond(assistant, [{"role": "user", "content": AUDIO}], [waveform])   # float32 mono, 16 kHz
print(result["reply"], result["tool_calls"])
# e.g. Opening wikipedia.  [{'name': 'open_url', 'arguments': {'url': 'https://www.wikipedia.org'}}]
```

Typed input works the same way (`{"role": "user", "content": "turn the volume down"}`). Tool results go back as
`{"role": "tool", "content": "<json>"}` and the model summarises them (weather, errors). `load` downloads this
repo with `huggingface_hub`; it needs `torch` and `transformers` only, no bitsandbytes. Text-only use also works
with plain `AutoModelForCausalLM`, because the root of this repo is a normal Qwen3 checkpoint.

### Tools it was trained on

`open_app` · `open_url` · `web_search` · `set_volume` · `set_brightness` · `toggle_setting` · `media_control` ·
`set_timer` · `get_weather` · `take_screenshot` · `type_text` (asks first) · `lock_screen` (asks first).
The exact schemas are in `omni_config.json`; the system prompt and `tools=` list must match it, as `omni_assistant.py` does.
It also learned to ask when a request is unclear, to decline unsafe or unsupported requests, and to answer small talk without a tool.

## Evaluation

Frozen test set of 1,274 assistant turns, never trained on. Spoken tests use two **held-out voices**
(af_sarah, bm_fable), each clean and with added noise. `tool_exact` = turns that should call tools whose calls
match exactly (names and arguments).

| Release weights (merged, bf16) | tool_exact | all turns exact | false action | missed action | confirm violations | valid JSON |
|---|---|---|---|---|---|---|
| Typed | 81.2% | 87.8% | 4.0% | 2.0% | 0 | 100% |
| Spoken, clean | 72.1% | 82.7% | 4.2% | 7.3% | 0 | 100% |
| Spoken, noisy | 69.4% | 80.8% | 5.3% | 6.4% | 2 | 100% |

Before spoken fine-tuning (typed LoRA + aligned adapter only): spoken tool_exact 17.5% / 19.6%.
Audio alignment alone reached 7.0% WER on LibriSpeech test-clean (measured on the adapter before spoken fine-tuning).

**The project's own targets were not all met:** typed tool_exact target was ≥ 90% and spoken was "within 5
points of typed" (≥ 76.9%). Most misses are free-text arguments:
- search queries, site names and timer labels the model mishears or rewords ("toaster oven" → "toasteries"); `web_search` 28%, `open_url` 42%, `set_timer` 44% exact when spoken;
- about one in ten spoken action commands gets a clarifying question instead of an action;
- it sometimes takes "don't bother" as a yes to a confirmation. **Keep a client-side rule that confirm tools run only after an explicit yes.**

## Limitations and safe use

- Trained on synthetic speech (Kokoro voices) plus LibriSpeech read speech. Real microphones, accents and rooms will be harder than these numbers suggest.
- English only. Short commands; clips are cut at 30 s.
- The model only proposes calls. Run them through an allow-list with fixed commands per tool, never a shell. Confirm `type_text` and `lock_screen` with the user.

## Training data

- Desktop commands: ~20K synthetic rows generated with Google Gemini, cleaned and split by the author. **Not released.**
- Speech: the commands voiced with Kokoro-82M (Apache-2.0) in 8 training voices, with noise; LibriSpeech ASR corpus (CC BY 4.0, Panayotov et al., 2015).

## Licence and credits

Released under Apache-2.0. Built on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0) and the
[Whisper-small](https://huggingface.co/openai/whisper-small) encoder (MIT); spoken replies in the client by
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0). Training used LibriSpeech (CC BY 4.0).
Trained by Rajan Devkota as a hands-on project on multimodal adapters (LLaVA-style audio adapter + QLoRA).
