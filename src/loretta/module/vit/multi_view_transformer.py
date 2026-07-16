# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed under the DINOv3 License Agreement.

from einops import rearrange
import torch
from torch import Tensor, nn

from .block import SelfAttentionBlock
from .rope import RopePositionEmbedding


class MultiViewTransformer(nn.Module):
    """Fixed ViT-Base multi-view Transformer used by LoRetta's affine locator."""

    def __init__(self, *, in_dim: int, out_dim: int, device=None):
        super().__init__()
        dim, depth, num_heads = 768, 12, 12
        self.projector = nn.Linear(in_dim, dim, device=device)
        self.output_projector = nn.Linear(dim, out_dim, device=device)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(dim, num_heads, device=device) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6, device=device)
        self.rope_embed = RopePositionEmbedding(
            dim, num_heads=num_heads, device=device, dtype=torch.bfloat16
        )

    def forward(self, descriptors: Tensor) -> Tensor:
        batch_size, views, height, width, _ = descriptors.shape
        tokens = self.projector(descriptors.flatten(1, -2))
        rope = self.rope_embed(H=height, W=width)
        for index, block in enumerate(self.blocks):
            if index % 2 == 1:
                tokens = rearrange(
                    tokens,
                    "B (V H W) D -> (B V) (H W) D",
                    V=views,
                    H=height,
                    W=width,
                )
                tokens = block(tokens, rope)
                tokens = rearrange(
                    tokens,
                    "(B V) (H W) D -> B (V H W) D",
                    V=views,
                    H=height,
                    W=width,
                )
            else:
                tokens = block(tokens, None)
        return self.output_projector(self.norm(tokens)).reshape(
            batch_size, views, height, width, -1
        )
