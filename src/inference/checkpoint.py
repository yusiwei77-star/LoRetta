from pathlib import Path

import torch
from torch import nn


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> int:
    """Strictly load the published tensor-only LoRetta state dict."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state_dict, dict) or not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError(
            "Checkpoint must be the published canonical LoRetta tensor state dict"
        )

    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    shape_errors = sorted(
        key
        for key in set(expected) & set(state_dict)
        if tuple(expected[key].shape) != tuple(state_dict[key].shape)
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

    model.load_state_dict(state_dict, strict=True)
    return len(state_dict)
