"""Serve one assistant turn: chat messages in, spoken reply + tool calls out.

The prompt is built exactly like the training data (same system prompt, same `tools=` list, same
template flags), so the model sees at runtime what it saw while learning. The server is stateless:
the client sends the whole conversation every turn.
"""
import time

from src.components.generate import generate_batch
from src.services.data.format import encode, prompt_tools, render
from src.utils.config import load_yaml
from src.utils.metrics import parse_turn


def load_weights(mcfg: dict, adapter: str, weights: str):
    """qlora: 4-bit base + LoRA as evaluated · merged_bf16: LoRA folded into a bf16 base."""
    import torch
    from transformers import AutoModelForCausalLM

    from src.components.llm import load_adapter, load_base

    if weights == "qlora":
        model = load_adapter(load_base(mcfg), adapter)
    elif weights == "merged_bf16":
        base = AutoModelForCausalLM.from_pretrained(mcfg["id"], revision=mcfg["revision"], dtype=torch.bfloat16,
                                                    device_map={"": 0}, attn_implementation="sdpa")
        model = load_adapter(base, adapter).merge_and_unload()
    else:
        raise ValueError(f"unknown weights {weights!r}")
    return model.eval()


def load_assistant(serve_cfg: dict) -> dict:
    """Everything a turn needs, loaded once per container."""
    from src.components.llm import load_tokenizer

    train_cfg = load_yaml(serve_cfg["model"]["config"])
    data_cfg = load_yaml("configs/data.yaml")
    return {
        "model": load_weights(train_cfg["model"], serve_cfg["model"]["adapter"], serve_cfg["model"]["weights"]),
        "tokenizer": load_tokenizer(train_cfg["model"], padding_side="left"),
        "tools": prompt_tools(load_yaml(data_cfg["tools_config"]), data_cfg["format"]),
        "system_prompt": data_cfg["format"]["system_prompt"],
        "stop_ids": train_cfg["data"]["stop_ids"],
        "max_new_tokens": serve_cfg["generate"]["max_new_tokens"],
    }


def respond(assistant: dict, messages: list[dict]) -> dict:
    """One assistant turn for `messages` (user / assistant / tool, without the system prompt)."""
    started = time.time()
    tokenizer = assistant["tokenizer"]
    chat = [{"role": "system", "content": assistant["system_prompt"]}, *messages]
    prompt_ids = encode(tokenizer, render(tokenizer, chat, assistant["tools"], True))
    raw = generate_batch(assistant["model"], tokenizer, [prompt_ids], assistant["max_new_tokens"], assistant["stop_ids"])[0]
    turn = parse_turn(raw)
    return {"reply": turn["reply"], "tool_calls": turn["calls"], "json_ok": turn["json_ok"], "raw": raw,
            "prompt_tokens": len(prompt_ids), "ms": round((time.time() - started) * 1000)}
