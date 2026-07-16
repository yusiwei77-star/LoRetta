from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import pickle

import torch
from torch import nn


LEGACY_PREFIX_RENAMES = (
    ("f.", "dinov3_feature_extractor."),
    (
        "matcher.matcher.mv_vit.",
        "affine_locator.multi_view_transformer.",
    ),
    (
        "matcher.matcher.decoder.flow.",
        "affine_locator.match_embedding_decoder.warp_head.",
    ),
    (
        "matcher.matcher.decoder.conf.",
        "affine_locator.match_embedding_decoder.matchability_head.",
    ),
    (
        "matcher.matcher.decoder.",
        "affine_locator.match_embedding_decoder.",
    ),
    ("matcher.matcher.null_pos_emb", "affine_locator.no_match_embedding"),
    ("matcher.matcher.null_logit", "affine_locator.no_match_logit"),
    ("matcher.matcher.omega", "affine_locator.position_frequencies"),
    ("matcher.matcher.scale", "affine_locator.position_scale"),
    ("matcher.matcher.temp", "affine_locator.temperature"),
    ("cnn_backbone.backbone.", "vgg_feature_pyramid."),
    ("refiner.refiner4.", "dense_register.refiner0."),
    ("refiner.refiner3.", "dense_register.refiner1."),
    ("refiner.refiner2.", "dense_register.refiner2."),
)


def load_safe_checkpoint_payload(path: Path):
    """Load tensor-only checkpoints without permitting arbitrary Python objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError as error:
        raise RuntimeError(
            "Checkpoint requires unsafe Python deserialization. Convert a trusted legacy "
            "checkpoint with scripts/migrate_checkpoint.py before inference."
        ) from error


def extract_state_dict(payload) -> Mapping[str, torch.Tensor]:
    state_dict = payload.get("state_dict", payload) if isinstance(payload, Mapping) else None
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint must be a state dict or contain a non-empty 'state_dict'")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("The selected checkpoint state_dict contains non-tensor values")
    return state_dict


def canonicalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Remove wrappers and migrate the legacy LoRetta schema to paper-aligned names."""
    canonical = OrderedDict()
    for original_key, value in state_dict.items():
        key = original_key
        wrapper_removed = True
        while wrapper_removed:
            wrapper_removed = False
            for wrapper_prefix in ("model.", "module."):
                if key.startswith(wrapper_prefix):
                    key = key[len(wrapper_prefix) :]
                    wrapper_removed = True
                    break
        for legacy_prefix, canonical_prefix in LEGACY_PREFIX_RENAMES:
            if key.startswith(legacy_prefix):
                key = canonical_prefix + key[len(legacy_prefix) :]
                break
        key = key.replace(".displacement_proj.", ".residual_embedding.")
        if key in canonical:
            raise ValueError(f"Duplicate parameter after checkpoint migration: {key}")
        canonical[key] = value
    return canonical


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> int:
    """Strictly load a canonical state dict or a trusted legacy Lightning checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    canonical = canonicalize_state_dict(
        extract_state_dict(load_safe_checkpoint_payload(path))
    )
    expected = model.state_dict()
    missing = sorted(set(expected) - set(canonical))
    unexpected = sorted(set(canonical) - set(expected))
    shape_errors = sorted(
        key
        for key in set(expected) & set(canonical)
        if tuple(expected[key].shape) != tuple(canonical[key].shape)
    )
    if missing or unexpected or shape_errors:
        details = []
        if missing:
            details.append(f"missing={missing[:10]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:10]}")
        if shape_errors:
            details.append(f"shape_mismatch={shape_errors[:10]}")
        raise RuntimeError("Checkpoint is incompatible with LoRetta: " + "; ".join(details))

    model.load_state_dict(canonical, strict=True)
    return len(canonical)
