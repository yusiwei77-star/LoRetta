from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn


MODEL_REPO_ID = "BRoss123/LoRetta"
CHECKPOINT_FILENAME = "loretta.pth"
MODEL_REVISION = "b26b60f781ea5d7dc483e347afc00fd410fe514b"


def resolve_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Return an explicit checkpoint or download the published LoRetta weights."""
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    try:
        return Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=CHECKPOINT_FILENAME,
                revision=MODEL_REVISION,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not download {MODEL_REPO_ID}/{CHECKPOINT_FILENAME}. "
            "Check the network connection and Hugging Face repository access, "
            "or pass --checkpoint with a local file."
        ) from error


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
