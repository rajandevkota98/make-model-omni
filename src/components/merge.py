"""Build audio prompts and swap placeholder embeddings for adapter outputs.

  prompt:  <|im_start|>user\n [P P P ... P] \nTranscribe the audio.<|im_end|>\n<|im_start|>assistant\n...
                              └ one P per audio token ┘
  merge:   embed(prompt) with every P row replaced, in order, by the adapter's vectors.
"""
import math

import torch

from src.components.encoders import encoder_frames

IGNORE_INDEX = -100


def audio_tokens(n_samples: int, ecfg: dict, k: int) -> int:
    return math.ceil(encoder_frames(n_samples, ecfg) / k)


def split_prompt(prompt_ids: list[int], placeholder_id: int) -> tuple[list[int], list[int]]:
    """Template ids around the single placeholder → (before, after)."""
    if prompt_ids.count(placeholder_id) != 1:
        raise ValueError("prompt must contain the placeholder exactly once")
    i = prompt_ids.index(placeholder_id)
    return prompt_ids[:i], prompt_ids[i + 1:]


def build_example(before: list[int], after: list[int], placeholder_id: int, n_audio: int, target_ids: list[int]) -> dict:
    """Token ids with n_audio placeholders; labels learn only the target."""
    prompt = before + [placeholder_id] * n_audio + after
    return {"input_ids": prompt + target_ids, "labels": [IGNORE_INDEX] * len(prompt) + target_ids, "prompt_len": len(prompt)}


def merge_embeddings(text_embeds: torch.Tensor, input_ids: torch.Tensor, placeholder_id: int, audio: torch.Tensor) -> torch.Tensor:
    """[B, L, D] text embeddings + [N, D] audio vectors (batch order) → [B, L, D] with placeholders filled."""
    mask = input_ids == placeholder_id
    if int(mask.sum()) != audio.shape[0]:
        raise ValueError(f"{int(mask.sum())} placeholders but {audio.shape[0]} audio vectors")
    return text_embeds.masked_scatter(mask.unsqueeze(-1), audio.to(text_embeds.dtype))
