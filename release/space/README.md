---
title: omni-assistant
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: Speak a desktop command; a 0.6B model hears it and picks a tool
models:
  - Rajan98/omni-assistant-0.6b
  - Qwen/Qwen3-0.6B
  - openai/whisper-small
  - hexgrad/Kokoro-82M
---

# omni-assistant demo

Speak or type a Linux-desktop command. The model hears the audio directly (Whisper-small encoder → trained
audio adapter → Qwen3-0.6B with a merged LoRA), writes a short reply and the tool calls it would make.
Kokoro-82M reads the reply aloud. **Nothing is executed in this Space.** The real assistant runs the tool
calls on your own desktop through an allow-listed executor.

Model, scores and limits: [Rajan98/omni-assistant-0.6b](https://huggingface.co/Rajan98/omni-assistant-0.6b)
