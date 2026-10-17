"""Step 6 forward / loss / optimizer step for the full Thinker: audio → Whisper → adapter → Qwen3 + LoRA.

Same loss as step 3 (assistant tokens only, split into tool-call vs reply), but the prompt embeddings
have the adapter's audio vectors in the placeholder slots. Typed replay batches have no audio and go
through the exact same path (zero placeholders).
"""
import math

import torch
import torch.nn.functional as F

from src.components.align_step import adapter_tokens
from src.components.llm import decoder_and_head
from src.components.merge import IGNORE_INDEX, merge_embeddings
from src.components.train_step import add_stats, empty_stats, stats_to_metrics
from src.utils.schedule import lr_at


def batch_embeds(model, adapter, whisper: dict, batch: dict, placeholder_id: int, cfg: dict) -> torch.Tensor:
    decoder, _ = decoder_and_head(model)
    ids = batch["input_ids"].to("cuda")
    text = decoder.embed_tokens(ids)
    if not batch["waveforms"]:
        return text
    return merge_embeddings(text, ids, placeholder_id, adapter_tokens(adapter, whisper, batch["waveforms"], cfg))


def token_losses(model, adapter, whisper: dict, batch: dict, placeholder_id: int, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    decoder, head = decoder_and_head(model)
    embeds = batch_embeds(model, adapter, whisper, batch, placeholder_id, cfg)
    hidden = decoder(inputs_embeds=embeds, attention_mask=batch["attention_mask"].to("cuda")).last_hidden_state
    labels, types = batch["labels"][:, 1:].to("cuda"), batch["types"][:, 1:].to("cuda")
    learned = labels != IGNORE_INDEX
    logits = head(hidden[:, :-1][learned]).float()
    return F.cross_entropy(logits, labels[learned], reduction="none"), types[learned]


def train_step(model, adapter, whisper: dict, optimizer, micro_batches: list[dict], step: int, total_steps: int,
               placeholder_id: int, cfg: dict) -> dict:
    """Both param groups (LoRA, adapter) follow the same schedule, each from its own peak LR."""
    o = cfg["optim"]
    scale = lr_at(step, total_steps, 1.0, o["warmup_ratio"], o["min_lr_ratio"])
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * scale
    total = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in micro_batches)
    stats = empty_stats()
    for batch in micro_batches:
        losses, types = token_losses(model, adapter, whisper, batch, placeholder_id, cfg)
        (losses.sum() / total).backward()
        stats = add_stats(stats, losses, types)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, o["grad_clip"]).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {**stats_to_metrics(stats), "lr_lora": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm if math.isfinite(grad_norm) else float("nan")}


@torch.no_grad()
def evaluate_loss(model, adapter, whisper: dict, batches: list[dict], placeholder_id: int, cfg: dict) -> dict:
    """Dev loss overall, per source (spoken / text) and per conversation kind for spoken."""
    model.eval()
    adapter.eval()
    groups: dict[str, dict] = {}
    for batch in batches:
        losses, types = token_losses(model, adapter, whisper, batch, placeholder_id, cfg)
        rows = torch.nonzero(batch["labels"][:, 1:] != IGNORE_INDEX)[:, 0].tolist()
        for name in {f"{i['source']}" for i in batch["items"]} | {"all"}:
            mask = torch.tensor([name == "all" or batch["items"][r]["source"] == name for r in rows], device=losses.device)
            groups[name] = add_stats(groups.get(name, empty_stats()), losses[mask], types[mask])
    model.train()
    adapter.train()
    metrics = stats_to_metrics(groups.pop("all"))
    metrics.update({f"loss_{name}": stats_to_metrics(s)["loss"] for name, s in sorted(groups.items())})
    return metrics

