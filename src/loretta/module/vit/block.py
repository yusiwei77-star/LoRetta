# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed under the DINOv3 License Agreement.

from torch import Tensor, nn

from .attention import SelfAttention


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, device=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, device=device)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, device=device)

    def forward(self, value: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(value)))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, device=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, device=device)
        self.attn = SelfAttention(dim, num_heads, device=device)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, device=device)
        self.mlp = MLP(dim, dim * 4, device=device)

    def forward(
        self,
        value: Tensor,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        value = value + self.attn(self.norm1(value), rope=rope)
        return value + self.mlp(self.norm2(value))
