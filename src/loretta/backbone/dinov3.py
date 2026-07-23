"""Pinned DINOv3 feature extractor for LoRetta inference."""

from functools import partial
from types import MethodType
from typing import Callable

import torch
from einops import rearrange
from torch import nn

from ..third_party.dinov3 import dinov3_vitl16


def _imagenet(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        (0.485, 0.456, 0.406), device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        (0.229, 0.224, 0.225), device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    return (image - mean) / std


def _wrap_forward(
    forward: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    patch_size: int,
    enable_amp: bool,
):
    def wrapped(self: nn.Module, image: torch.Tensor) -> list[torch.Tensor]:
        amp_enabled = enable_amp and image.device.type == "cuda"
        with (
            torch.autocast(image.device.type, torch.bfloat16, enabled=amp_enabled),
            torch.no_grad(),
        ):
            batch, channels, height, width = image.shape
            if channels != 3:
                raise ValueError(f"Expected RGB input, got {tuple(image.shape)}")
            out_height, out_width = height // patch_size, width // patch_size
            outputs = forward(_imagenet(image))
            return [
                rearrange(output, "B (H W) D -> B H W D", H=out_height, W=out_width)
                for output in outputs
            ]

    return wrapped


def build_dinov3_feature_extractor(
    *,
    enable_amp: bool,
    layer_indices: tuple[int, ...] = (11, 17),
) -> nn.Module:
    """Build the bundled DINOv3 module while preserving its native state-dict keys."""
    model: nn.Module = dinov3_vitl16()
    layers = [
        index if index > 0 else len(model.blocks) + index
        for index in layer_indices
    ]
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.forward = MethodType(
        _wrap_forward(
            partial(model.get_intermediate_layers, n=layers),
            patch_size=16,
            enable_amp=enable_amp,
        ),
        model,
    )
    return model
