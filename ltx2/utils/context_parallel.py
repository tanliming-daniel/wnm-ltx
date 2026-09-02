from __future__ import annotations

from typing import Any

import torch


def is_cp_enabled() -> bool:
    return False


def get_cp_world_size() -> int:
    return 1


def get_cp_rank() -> int:
    return 0


def scatter_sequence(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x


def gather_sequence(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x


def gather_for_loss(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x


def apply_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    attention_function: Any,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return attention_function(q, k, v, heads, mask)


def pad_to_cp_divisible(x: torch.Tensor, dim: int = 1) -> tuple[torch.Tensor, int]:
    return x, x.shape[dim]


def unpad_from_cp(x: torch.Tensor, orig_len: int, dim: int = 1) -> torch.Tensor:
    if x.shape[dim] <= orig_len:
        return x
    index = [slice(None)] * x.ndim
    index[dim] = slice(0, orig_len)
    return x[tuple(index)]
