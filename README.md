# make-model-omni

Qwen3-0.6B taught to drive a Linux desktop with tool calls: **omni-assistant**. Today it takes typed
commands. Hearing, seeing and its own voice come in the later roadmap steps (see `CLAUDE.md`).

## POC: try it

```bash
uv sync
uv run modal deploy -m apps.serve           # once: model server on Modal (L4, scales to zero)
uv run python -m client.main --dry-run      # safe: prints the desktop commands instead of running them
uv run python -m client.main                # real: opens apps, changes volume, ...
```

Then type commands like:

```
open the browser and go to facebook
what's the weather in pokhara tomorrow
turn the volume down and pause the music
set a timer for ten minutes for tea
lock my screen      → it asks first, answer "yes"
```

- The first command after a pause takes about 30–60 s while the GPU container starts. After that,
  replies take about 0.5–2 s.
- Typing text needs `sudo apt install xdotool`.
- The weather uses Open-Meteo. Set your city in `configs/client.yaml` (`weather.default_location`).
- Commands for each tool are in `configs/client.yaml`. Only the 12 tools in `configs/tools.yaml` can
  run, and their arguments are checked against the schema first. There is no shell tool.

## How a command flows

```
you type ─► client/main.py ─► Modal: apps/serve.py (Qwen3 + LoRA) ─► reply + <tool_call>s
                 │                                                        │
                 └─ client/executor (allowlist + schema check) ◄──────────┘
                      └─ result: weather data or an error goes back to the model for a spoken follow-up
```

## Step 3 results (frozen test set, 1,274 turns)

| | untrained | 4-bit + LoRA | merged bf16 (served) |
|---|---|---|---|
| tool-call exact match | 7.3% | 81.9% | 81.4% |
| false actions | 3.7% | 5.4% | 4.9% |
| missed actions | 77.1% | 0.7% | 1.6% |
| confirm violations | 13 | 0 | 0 |
