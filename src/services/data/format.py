"""Stage 7 · format: conversations → token ids + how many leading tokens are prompt (not learned).

One example per assistant turn, built exactly the way the runtime builds it:
  prompt = chat_template(history, add_generation_prompt=True)   → tokenized on its own
  target = what the template appends when that turn is the last  → tokenized on its own
Why not one sequence per conversation: Qwen3 renders earlier assistant turns WITHOUT the empty
<think></think> block, but the runtime generates every turn right AFTER it. Training on the single
sequence would teach a format the model never sees at inference.
"""
import statistics
from collections import Counter
from pathlib import Path

from src.services.data.augment import augmented_path
from src.services.data.split import SPLITS
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, write_jsonl

IGNORE_INDEX = -100


def load_tokenizer(fcfg: dict):
    from transformers import AutoTokenizer  # heavy import, only when the stage runs

    return AutoTokenizer.from_pretrained(fcfg["tokenizer"], revision=fcfg["revision"])


def strip_keys(schema: object, keys: set[str]) -> object:
    """Drop schema keys the prompt doesn't need (e.g. if/then/else; the description says it in words)."""
    if isinstance(schema, dict):
        return {k: strip_keys(v, keys) for k, v in schema.items() if k not in keys}
    if isinstance(schema, list):
        return [strip_keys(v, keys) for v in schema]
    return schema


def prompt_tools(tools_cfg: dict, fcfg: dict) -> list[dict]:
    """tools.yaml → the `tools=` list Qwen's chat template expects."""
    hidden = set(fcfg.get("hide_schema_keys", []))
    return [{"type": "function", "function": {"name": name, "description": spec["description"],
                                              "parameters": strip_keys(spec["parameters"], hidden)}}
            for name, spec in tools_cfg["tools"].items()]


def render(tokenizer, messages: list[dict], tools: list[dict], generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=generation_prompt, enable_thinking=False)


def turn_texts(tokenizer, messages: list[dict], tools: list[dict], index: int) -> tuple[str, str]:
    """(prompt, target) text for the assistant message at `index`."""
    prompt = render(tokenizer, messages[:index], tools, True)
    full = render(tokenizer, messages[: index + 1], tools, False)
    if not full.startswith(prompt):
        raise ValueError("chat template: turn is not a continuation of its prompt")
    return prompt, full[len(prompt):].rstrip("\n")  # learn up to and including <|im_end|>


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def format_conversation(item: dict, tokenizer, tools: list[dict], system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}, *item["messages"]]
    examples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt, target = turn_texts(tokenizer, messages, tools, index)
        prompt_ids, target_ids = encode(tokenizer, prompt), encode(tokenizer, target)
        examples.append({"id": f"{item['id']}@{index}", "conversation_id": item["id"], "kind": item["kind"],
                         "turn": index, "input_ids": prompt_ids + target_ids, "prompt_len": len(prompt_ids)})
    return examples


def drop_repeats(examples: list[dict]) -> list[dict]:
    """Identical token sequences (e.g. the shared first turn of confirm_yes / confirm_no) are kept once."""
    seen: set[tuple[int, ...]] = set()
    kept = []
    for example in examples:
        key = tuple(example["input_ids"])
        if key not in seen:
            seen.add(key)
            kept.append(example)
    return kept


def labels_for(example: dict) -> list[int]:
    """Prompt tokens are ignored by the loss; only the assistant's turn is learned."""
    ids, n = example["input_ids"], example["prompt_len"]
    return [IGNORE_INDEX] * n + ids[n:]


def length_stats(examples: list[dict]) -> dict:
    lengths = sorted(len(e["input_ids"]) for e in examples)
    if not lengths:
        return {}
    return {"n": len(lengths), "p50": lengths[len(lengths) // 2], "p95": lengths[int(len(lengths) * 0.95)],
            "max": lengths[-1], "mean": round(statistics.fmean(lengths), 1),
            "total_tokens": sum(lengths), "learned_tokens": sum(len(e["input_ids"]) - e["prompt_len"] for e in examples)}


# ── stage ───────────────────────────────────────────────────────────────────

def formatted_path(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["formatted_dir"]) / f"{split}.jsonl"


def inputs(cfg: dict) -> list[Path]:
    return [augmented_path(cfg, name) for name in SPLITS] + [Path(cfg["tools_config"])]


def outputs(cfg: dict) -> list[Path]:
    return [formatted_path(cfg, name) for name in SPLITS]


def run(cfg: dict) -> dict:
    fcfg = cfg["format"]
    tokenizer = load_tokenizer(fcfg)
    tools = prompt_tools(load_yaml(cfg["tools_config"]), fcfg)
    system_tokens = len(encode(tokenizer, render(tokenizer, [{"role": "system", "content": fcfg["system_prompt"]}], tools, False)))
    rows_in, stats, dropped = 0, {}, Counter()
    for name in SPLITS:
        conversations = read_jsonl(augmented_path(cfg, name))
        examples = [e for item in conversations for e in format_conversation(item, tokenizer, tools, fcfg["system_prompt"])]
        unique = drop_repeats(examples)
        kept = [e for e in unique if len(e["input_ids"]) <= fcfg["max_length"]]
        dropped[f"repeated_turn:{name}"] = len(examples) - len(unique)
        dropped[f"too_long:{name}"] = len(unique) - len(kept)
        write_jsonl(formatted_path(cfg, name), kept)
        rows_in += len(conversations)
        stats[name] = length_stats(kept)
    return {"in": rows_in, "out": sum(s.get("n", 0) for s in stats.values()), "rejected": sum(dropped.values()),
            "reasons": {k: v for k, v in dropped.items() if v},
            "system_prompt_tokens": system_tokens, "tokenizer": fcfg["tokenizer"], "revision": fcfg["revision"], "lengths": stats}
