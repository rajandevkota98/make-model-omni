"""Stage 7c · spoken_format: the step 3 examples again, but every user turn is heard instead of read.

Same system prompt, same tools, same assistant targets as `format`. Each user message is replaced by
one audio placeholder, and the prompt is stored as text segments around those slots:

    segments[0]  <audio for audio_texts[0]>  segments[1]  <audio for audio_texts[1]>  segments[2] ...

At training time each slot is filled with a clip of that text (a random voice per epoch) and expanded
to as many placeholder tokens as the clip needs. Tool results (weather, errors) stay as text.
"""
from pathlib import Path

from src.components.tokens import add_placeholder
from src.services.data.augment import augmented_path
from src.services.data.format import encode, load_tokenizer, prompt_tools, turn_texts
from src.services.data.split import SPLITS
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, write_jsonl

EMBEDDING_ROWS = 151936


def heard(messages: list[dict], placeholder: str) -> list[dict]:
    return [{**m, "content": placeholder} if m["role"] == "user" else m for m in messages]


def split_segments(ids: list[int], placeholder_id: int) -> list[list[int]]:
    segments, current = [], []
    for token in ids:
        if token == placeholder_id:
            segments.append(current)
            current = []
        else:
            current.append(token)
    return segments + [current]


def spoken_examples(item: dict, tokenizer, tools: list[dict], system_prompt: str, placeholder: str, placeholder_id: int) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}, *item["messages"]]
    spoken = heard(messages, placeholder)
    examples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt, target = turn_texts(tokenizer, spoken, tools, index)
        texts = [m["content"] for m in messages[:index] if m["role"] == "user"]
        segments = split_segments(encode(tokenizer, prompt), placeholder_id)
        if len(segments) != len(texts) + 1:
            raise ValueError(f"{item['id']}: {len(texts)} user turns but {len(segments) - 1} audio slots")
        examples.append({"id": f"{item['id']}@{index}", "conversation_id": item["id"], "kind": item["kind"], "turn": index,
                         "segments": segments, "audio_texts": texts, "target": encode(tokenizer, target)})
    return examples


def drop_repeats(examples: list[dict]) -> list[dict]:
    seen, kept = set(), []
    for e in examples:
        key = (tuple(map(tuple, e["segments"])), tuple(e["audio_texts"]), tuple(e["target"]))
        if key not in seen:
            seen.add(key)
            kept.append(e)
    return kept


# ── stage ───────────────────────────────────────────────────────────────────

def spoken_path(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["spoken_dir"]) / f"{split}.jsonl"


def inputs(cfg: dict) -> list[Path]:
    return [augmented_path(cfg, name) for name in SPLITS] + [Path(cfg["tools_config"])]


def outputs(cfg: dict) -> list[Path]:
    return [spoken_path(cfg, name) for name in SPLITS]


def run(cfg: dict) -> dict:
    fcfg, scfg = cfg["format"], cfg["spoken_format"]
    tokenizer = load_tokenizer(fcfg)
    placeholder_id = add_placeholder(tokenizer, scfg["placeholder"], EMBEDDING_ROWS)
    tools = prompt_tools(load_yaml(cfg["tools_config"]), fcfg)
    stats, rows_in, dropped = {}, 0, 0
    for name in SPLITS:
        conversations = read_jsonl(augmented_path(cfg, name))
        examples = [e for item in conversations
                    for e in spoken_examples(item, tokenizer, tools, fcfg["system_prompt"], scfg["placeholder"], placeholder_id)]
        unique = drop_repeats(examples)
        write_jsonl(spoken_path(cfg, name), unique)
        rows_in += len(conversations)
        dropped += len(examples) - len(unique)
        stats[name] = {"examples": len(unique), "audio_slots": sum(len(e["audio_texts"]) for e in unique)}
    return {"in": rows_in, "out": sum(s["examples"] for s in stats.values()), "rejected": dropped,
            "placeholder_id": placeholder_id, "splits": stats}
