"""Pure tensor ops the adapters use.

frame_stack: glue k neighbouring audio frames into one wider frame.
  Whisper gives 50 frames per second (one every 20 ms). With k=4, four frames of 768 numbers become one
  frame of 3072 numbers: 12.5 tokens per second for Qwen to read, 4× shorter, and nothing thrown away.
"""
import torch
import torch.nn.functional as F


def frame_stack(frames: torch.Tensor, k: int) -> torch.Tensor:
    """[T, D] → [ceil(T / k), k·D]; the last group is zero-padded."""
    t, d = frames.shape
    padded = F.pad(frames, (0, 0, 0, (-t) % k))
    return padded.reshape(-1, k * d)
