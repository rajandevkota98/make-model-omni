"""Stage 1b forward, loss and optimizer step: audio → Whisper → adapter → frozen Qwen3 → transcript loss.

Only the adapter has gradients. The loss still flows backward THROUGH the frozen Qwen layers to reach
the adapter's output vectors: Qwen acts as a fixed judge telling the adapter "these vectors didn't
read like the right words yet".
"""
import math

import torch
import torch.nn.functional as F

from src.components.adapters.ops import frame_stack
from src.components.encoders import audio_features
from src.components.llm import decoder_and_head
from src.components.merge import IGNORE_INDEX, merge_embeddings
from src.utils.schedule import lr_at


def adapter_tokens(adapter, whisper: dict, waveforms: list, cfg: dict) -> torch.Tensor:
    """All clips' audio tokens, concatenated in batch order: [sum of tokens, Qwen dim]."""
    k = cfg["adapter"]["frame_stack"]
    stacked = torch.cat([frame_stack(f, k) for f in audio_features(whisper, waveforms, cfg["encoder"])])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return adapter(stacked.float())


def embed_batch(llm, adapter, whisper: dict, batch: dict, placeholder_id: int, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """→ (merged input embeddings, attention mask, audio tokens)."""
    decoder, _ = decoder_and_head(llm)
    input_ids = batch["input_ids"].to("cuda")
    audio = adapter_tokens(adapter, whisper, batch["waveforms"], cfg)
    text = decoder.embed_tokens(input_ids)
    return merge_embeddings(text, input_ids, placeholder_id, audio), batch["attention_mask"].to("cuda"), audio


def norm_ratio(audio: torch.Tensor, llm, batch: dict, placeholder_id: int) -> float:
    """Mean length of audio vectors ÷ mean length of real text embeddings. Far from ~1 = scale mismatch."""
    decoder, _ = decoder_and_head(llm)
    ids = batch["input_ids"].to("cuda")
    text = decoder.embed_tokens(ids[(ids != placeholder_id) & (batch["attention_mask"].to("cuda") == 1)])
    return float(audio.float().norm(dim=-1).mean() / text.float().norm(dim=-1).mean())


def token_losses(llm, adapter, whisper: dict, batch: dict, placeholder_id: int, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token cross-entropy on the transcript tokens only, plus the audio tokens (for stats)."""
    decoder, head = decoder_and_head(llm)
    embeds, mask, audio = embed_batch(llm, adapter, whisper, batch, placeholder_id, cfg)
    hidden = decoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    labels = batch["labels"][:, 1:].to("cuda")
    learned = labels != IGNORE_INDEX
    logits = head(hidden[:, :-1][learned]).float()
    return F.cross_entropy(logits, labels[learned], reduction="none"), audio


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_step(llm, adapter, whisper: dict, optimizer, micro_batches: list[dict], step: int, total_steps: int,
               placeholder_id: int, cfg: dict) -> dict:
    o = cfg["optim"]
    lr = lr_at(step, total_steps, o["lr"], o["warmup_ratio"], o["min_lr_ratio"])
    set_lr(optimizer, lr)
    total = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in micro_batches)
    loss_sum, n, ratios = 0.0, 0, []
    for batch in micro_batches:
        losses, audio = token_losses(llm, adapter, whisper, batch, placeholder_id, cfg)
        (losses.sum() / total).backward()
        loss_sum, n = loss_sum + losses.detach().sum().item(), n + losses.numel()
        ratios.append(norm_ratio(audio.detach(), llm, batch, placeholder_id))
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), o["grad_clip"]).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": loss_sum / max(n, 1), "lr": lr, "grad_norm": grad_norm if math.isfinite(grad_norm) else float("nan"),
            "norm_ratio": sum(ratios) / len(ratios), "learned_tokens": n}


@torch.no_grad()
def evaluate_loss(llm, adapter, whisper: dict, batches: list[dict], placeholder_id: int, cfg: dict) -> dict:
    """Dev loss overall and per source (librispeech / kokoro)."""
    adapter.eval()
    sums: dict[str, list[float]] = {}
    for batch in batches:
        losses, _ = token_losses(llm, adapter, whisper, batch, placeholder_id, cfg)
        rows = torch.nonzero(batch["labels"][:, 1:] != IGNORE_INDEX)[:, 0].tolist()
        for loss, r in zip(losses.tolist(), rows):
            for key in ("all", batch["rows"][r]["source"]):
                s = sums.setdefault(key, [0.0, 0])
                s[0], s[1] = s[0] + loss, s[1] + 1
    adapter.train()
    return {"loss" if key == "all" else f"loss_{key}": total / max(count, 1) for key, (total, count) in sums.items()}
