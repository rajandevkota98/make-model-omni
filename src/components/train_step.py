"""Loss and one optimizer step. Pure functions over (model, batch) — no trainer object.

How the loss works, step by step:
  1. The transformer reads the whole sequence once and gives one hidden vector per position.
  2. Position t predicts token t+1 ("next-token prediction"), so labels are shifted left by one.
  3. Only positions whose label is not IGNORE_INDEX (the assistant's turn) are kept — about 2% here.
  4. Only those few vectors go through lm_head (1024 → 151,936 vocab). Doing all positions would
     need ~10 GB just for logits; doing only the learned ones needs ~0.1 GB.
  5. Cross-entropy per token = −log(probability the model gave the right next token).
     Loss 0.0 = always certain and right; loss 2.3 ≈ choosing among ~10 equally likely tokens.
  6. The batch loss is the mean over learned tokens, also reported separately for tool-call
     tokens and spoken-reply tokens so you can see which part the model has learned.
"""
import math

import torch
import torch.nn.functional as F

from src.components.batching import REPLY, TOOL_CALL
from src.components.llm import decoder_and_head
from src.services.data.format import IGNORE_INDEX
from src.utils.schedule import lr_at


def token_losses(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token cross-entropy for learned positions only, and each token's type."""
    decoder, head = decoder_and_head(model)
    hidden = decoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels, types = batch["labels"][:, 1:], batch["types"][:, 1:]   # token t+1 is the answer for position t
    learned = labels != IGNORE_INDEX
    logits = head(hidden[:, :-1][learned]).float()
    return F.cross_entropy(logits, labels[learned], reduction="none"), types[learned]


def empty_stats() -> dict[str, float]:
    return {"sum": 0.0, "n": 0, "tool_sum": 0.0, "tool_n": 0, "reply_sum": 0.0, "reply_n": 0}


def add_stats(stats: dict, losses: torch.Tensor, types: torch.Tensor) -> dict:
    detached = losses.detach()
    tool, reply = types == TOOL_CALL, types == REPLY
    return {"sum": stats["sum"] + detached.sum().item(), "n": stats["n"] + detached.numel(),
            "tool_sum": stats["tool_sum"] + detached[tool].sum().item(), "tool_n": stats["tool_n"] + int(tool.sum()),
            "reply_sum": stats["reply_sum"] + detached[reply].sum().item(), "reply_n": stats["reply_n"] + int(reply.sum())}


def stats_to_metrics(stats: dict) -> dict[str, float]:
    mean = lambda s, n: s / n if n else float("nan")
    return {"loss": mean(stats["sum"], stats["n"]), "loss_tool_call": mean(stats["tool_sum"], stats["tool_n"]),
            "loss_reply": mean(stats["reply_sum"], stats["reply_n"]), "learned_tokens": stats["n"]}


def learned_count(batch: dict) -> int:
    return int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_step(model, optimizer, micro_batches: list[dict], step: int, total_steps: int, ocfg: dict) -> dict:
    """Gradient accumulation over micro-batches, normalised by learned tokens in the WHOLE step."""
    lr = lr_at(step, total_steps, ocfg["lr"], ocfg["warmup_ratio"], ocfg["min_lr_ratio"])
    set_lr(optimizer, lr)
    total_learned = sum(learned_count(batch) for batch in micro_batches)
    stats = empty_stats()
    for batch in micro_batches:
        losses, types = token_losses(model, batch)
        (losses.sum() / total_learned).backward()
        stats = add_stats(stats, losses, types)
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, ocfg["grad_clip"]).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {**stats_to_metrics(stats), "lr": lr, "grad_norm": grad_norm if math.isfinite(grad_norm) else float("nan")}


@torch.no_grad()
def evaluate_loss(model, batches: list[tuple[list[str], dict]]) -> dict[str, float]:
    """Dev loss overall and per conversation kind. `batches` = [(kinds, collated batch), ...]."""
    model.eval()
    overall, per_kind = empty_stats(), {}
    for kinds, batch in batches:
        losses, types = token_losses(model, batch)
        rows = torch.nonzero(batch["labels"][:, 1:] != IGNORE_INDEX)[:, 0]   # which example each learned token came from
        overall = add_stats(overall, losses, types)
        for kind in set(kinds):
            mask = torch.tensor([kinds[r] == kind for r in rows.tolist()], device=losses.device)
            per_kind[kind] = add_stats(per_kind.get(kind, empty_stats()), losses[mask], types[mask])
    model.train()
    metrics = stats_to_metrics(overall)
    metrics.update({f"loss_{kind}": stats_to_metrics(s)["loss"] for kind, s in sorted(per_kind.items())})
    return metrics
