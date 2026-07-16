# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed under the DINOv3 License Agreement.

import torch
from torch import Tensor, nn


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True, device=device)
        self.proj = nn.Linear(dim, dim, bias=True, device=device)

    @staticmethod
    def _apply_rope(q: Tensor, k: Tensor, rope: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        q, k = q.to(sin.dtype), k.to(sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        q = torch.cat(
            (q[:, :, :prefix], q[:, :, prefix:] * cos + _rotate_half(q[:, :, prefix:]) * sin),
            dim=-2,
        )
        k = torch.cat(
            (k[:, :, :prefix], k[:, :, prefix:] * cos + _rotate_half(k[:, :, prefix:]) * sin),
            dim=-2,
        )
        return q.to(q_dtype), k.to(k_dtype)

    def forward(self, value: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(attended)
