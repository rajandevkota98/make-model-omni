"""MLP adapter: stacked encoder frames → vectors the size of Qwen's token embeddings.

The output goes through RMSNorm with a gain that starts at Qwen's average embedding length, so audio
tokens arrive at the same scale as word tokens. Without it the outputs grew ~180× larger than words
and drowned out everything Qwen's layers tried to add at those positions (see step 5 notes).

Thin nn.Module (PyTorch needs one for parameters); the frame logic is in ops.py.
"""
import math

import torch
from torch import nn


class MLPAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, target_norm: float | None) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.norm = nn.RMSNorm(out_dim) if target_norm else nn.Identity()
        if target_norm:
            nn.init.constant_(self.norm.weight, target_norm / math.sqrt(out_dim))  # RMS 1 → length target_norm

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(frames))


def build_mlp(acfg: dict, encoder_dim: int, target_norm: float | None = None) -> MLPAdapter:
    return MLPAdapter(encoder_dim * acfg["frame_stack"], acfg["hidden"], acfg["out_dim"], target_norm if acfg.get("out_norm") else None)
