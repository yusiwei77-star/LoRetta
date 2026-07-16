# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed under the DINOv3 License Agreement.

import math

import torch
from torch import Tensor, nn


class RopePositionEmbedding(nn.Module):
    """Fixed axial RoPE used by LoRetta's inference-only ViT."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("RoPE embedding dimension must be divisible by four times num_heads")
        self.head_dim = embed_dim // num_heads
        self.dtype = dtype
        periods = 100.0 ** (
            2
            * torch.arange(self.head_dim // 4, device=device, dtype=dtype)
            / (self.head_dim // 2)
        )
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        options = {"device": self.periods.device, "dtype": self.dtype}
        coords_h = torch.arange(0.5, H, **options) / H
        coords_w = torch.arange(0.5, W, **options) / W
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)
