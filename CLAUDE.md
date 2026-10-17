# make-model-omni

Turn **Qwen3-0.6B** into a small omni model that **hears, sees, talks and acts**, built as
**omni-assistant**, a voice assistant for a Linux desktop. You say *"open the browser and go to
Facebook"*. The model hears the audio directly, emits a Qwen3 `<tool_call>`, a local executor runs
it, and the model speaks a short confirmation.

The goal is to learn multimodal internals hands-on. Explanations should be simple ("teach me like a kid").

## Code rules (must follow)

- **Functional Python, not OOP.** Write plain functions with explicit arguments and config passed
  in. Avoid classes, inheritance, stateful objects and singletons.
  - **Exception:** PyTorch needs `nn.Module` (adapters). Keep those classes thin, with
    `__init__` plus `forward` only. Put the logic in pure functions (`pixel_shuffle`,
    `frame_stack`, `merge_embeddings`, ...) that the module calls.
  - Use `dataclass(frozen=True)` or `TypedDict` only as plain data holders, with no methods.
- **Config-driven and modular.** Keep files small. Every swappable piece (adapter type, encoder,
  codec, TTS voice) is selected by a key in `configs/*.yaml` through a registry dict of functions.
- **Layout:** `src/{components,services,utils}`.
  - `components/`: model pieces.
  - `services/`: data, TTS, codec, storage, hub.
  - `utils/`: config, logging, freeze, metrics.
  - `apps/`: thin Modal entrypoints only.
  - `client/`: the laptop-side CPU code.
- Type hints everywhere. Unit tests in `tests/` use tiny tensors and run on CPU.
- Use `uv` for dependencies (`pyproject.toml`), Python 3.12.

## Compute

- **All GPU work runs on Modal** (budget: $279 credit). Use the Volume `omni-lab` (`/vol/data`,
  `/vol/ckpt`, `/vol/evals`).
- The dev machine has **no GPU** and about 1–4 GB of free RAM. Never run model inference or training
  locally. Locally, only run unit tests and CPU data scripts. Run smoke tests on a cheap Modal GPU
  (A10G).

## Architecture (locked decisions)

- **Thinker, LLaVA style:**
  - Eyes: frozen **SigLIP2-base** (`google/siglip2-base-patch16-512`, `hidden_states[-2]`), then an
    image adapter (pixel shuffle r=4, then an MLP with hidden size 2048, 64 tokens per image).
  - Ears: frozen **Whisper-small encoder** (decoder dropped, 30 s padding sliced off), then an audio
    adapter (frame stack k=4, then an MLP).
  - Both adapters feed **Qwen3-0.6B**.
  - Never shrink the 151,936-row embedding table when adding placeholder tokens.
- **Training is transfer learning, not distillation.** There is no CPT, and **no model is trained from
  scratch**: pretrained models are frozen or fine-tuned (LoRA); only the small adapters are new layers.
  - Stage 1a: image alignment. Stage 1b: audio alignment. Only the adapters train; Qwen3 is frozen
    in 4-bit.
  - Stage 2 (general SFT) and stage 3 (tool SFT) use **QLoRA**: NF4 with double quantization, bf16
    compute, LoRA r=32, α=64, dropout 0.05 on q, k, v, o, gate, up and down, LR 2e-4. One shared
    LoRA.
- **Inference:** merge the LoRA into bf16, then serve Qwen3 in **8-bit**. Adapters, encoders and the
  TTS stay **bf16**. Re-run evals after merging and after quantizing.
- **Speech out:** the pretrained **Kokoro-82M** TTS (Apache-2.0, one fixed voice) speaks the reply
  text. No Talker is trained (decided 2026-10-16: nothing from scratch). It speaks only the reply text,
  never JSON.
- Target size is about 1B params. HF release comes later, after licence checks: adapters-only repo
  first, then a full `trust_remote_code` repo.

## Tools (source of truth: Gemini data → `configs/tools.yaml`)

```
open_app        app: browser|email|calendar|music_player|text_editor|calculator|file_manager|settings|camera|terminal
open_url        url: https://…
web_search      query
set_volume      action: set|up|down|mute|unmute, level 0–100 (only with set)
set_brightness  action: set|up|down, level 0–100 (only with set)
toggle_setting  setting: wifi|bluetooth|dark_mode|night_light|do_not_disturb, state: on|off
media_control   action: play|pause|next|previous
set_timer       seconds 1–86400, label?
get_weather     day: today|tomorrow, location?   (executor: Open-Meteo, result returned as <tool_response>)
take_screenshot area: full|window|selection
type_text       text            (confirm: true)
lock_screen     —               (confirm: true)
```

**No shell tool, ever.** The executor only allows listed tools, and each app maps to a fixed
command. The desktop is X11, so typing uses `xdotool`.

## Data

- Synthetic commands come from Gemini: 20,000 rows. Raw files go in `data/synthetic/raw/`
  (gitignored).
  - Fields: `id`, `category`, `tool_focus`, `style`, `theme`, `utterance`, `tool_calls`,
    `needs_confirmation`, `reply`.
  - Categories: single 10K, and 2K each of multi_step, no_tool, clarify, refuse and confirm.
- Cleaning, in `scripts/validate_commands.py`:
  - Drop the 311 duplicates and the 8 commands that have conflicting labels.
  - Allow replies up to 25 words for refuse and no_tool, and 12 words for everything else.
  - Normalise case.
- Splits: train ~18,150, dev 500, test 1,000. Stratify on category × tool and dedupe before
  splitting. **The test set is frozen and never trained on.**
- Multi-turn samples built in code: confirm with yes/no follow-ups, weather with a `tool_response`
  and a spoken summary, and tool-error turns.
- Chat format: the Qwen3 template with `enable_thinking=False`, native `<tool_call>` JSON and a
  compact tool list in the system prompt. Loss is on assistant tokens only.
- Other datasets: LLaVA-Pretrain and LLaVA-665K subset (eyes), LibriSpeech-960 and SLURP (ears and
  real voices), TTS-voiced commands (3 voices plus noise; the test set uses a held-out voice).

## Roadmap (each step has a gate; don't skip ahead)

1. Clean and split the data (local).
2. Build the multi-turn samples (local).
3. Text-only QLoRA on A10G. Gate: tool-call exact match ≥ 90% on test, beating the untrained
   baseline.
4. TTS the commands.
5. Stage 1b audio alignment.
6. Spoken tool SFT (stage 3) with 15% replay. Gate: within 5 points of the text score.
7. Merge, 8-bit, re-eval. Gate: no more than a 1-point drop.
8. Speech out with pretrained Kokoro (no training). Gate: round-trip WER < 10%, real-time factor < 1.
9. Real-time client (P6). Gate: voice-end to action under 1.0 s.
10. Screen awareness (P7).
11. Release (P8).

Reference pages:
- `docs/omni-adapter-lab.html`: the guide.
- `docs/data-blueprint.html`: the data review and run sheet.

## Conventions

- Canonical audio: 16 kHz mono PCM s16le, 20 ms frames.
- Log every eval as JSON in `/vol/evals/`. Compare 4-bit+LoRA, merged bf16 and merged 8-bit.
- Never put secrets in the repo. The HF token lives in a Modal secret.
