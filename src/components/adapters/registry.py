"""Adapter type (configs: adapter.type) → builder. Add a new adapter by adding one entry."""
from typing import Callable

from torch import nn

from src.components.adapters.mlp import build_mlp

ADAPTERS: dict[str, Callable[..., nn.Module]] = {
    "mlp": build_mlp,
}


def build_adapter(acfg: dict, encoder_dim: int, target_norm: float | None = None) -> nn.Module:
    """target_norm: mean length of the LLM's token embeddings (used when acfg.out_norm is on)."""
    return ADAPTERS[acfg["type"]](acfg, encoder_dim, target_norm)


def save_adapter(adapter: nn.Module, path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").contiguous() for k, v in adapter.state_dict().items()}, str(path))


def load_adapter_weights(adapter: nn.Module, path) -> nn.Module:
    from safetensors.torch import load_file

    adapter.load_state_dict(load_file(str(path)))
    return adapter
