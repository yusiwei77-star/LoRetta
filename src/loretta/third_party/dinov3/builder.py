# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Local builder for the exact DINOv3 ViT-L/16 architecture used by LoRetta."""

from .models import DinoVisionTransformer

DINOV3_UPSTREAM_COMMIT = "adc254450203739c8149213a7a69d8d905b4fcfa"  # pragma: allowlist secret


def dinov3_vitl16() -> DinoVisionTransformer:
    """Build the pinned DINOv3 ViT-L/16 architecture without network access."""
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=False,
    )
    model.init_weights()
    return model
