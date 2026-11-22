import hashlib
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
from torch import nn

MODEL_REPO_ID = "BRoss123/LoRetta"
CHECKPOINT_FILENAME = "loretta.pth"
MODEL_REVISION = "a37451603dc8f2575de51c19dc44ae671d6fcbe3"  # pragma: allowlist secret
CHECKPOINT_SHA256 = "9aca8bc46f1e1f653ae7ab31354d45461a462b8142af90fcc07dc6f1f1572589"  # pragma: allowlist secret
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def _verify_checkpoint_hash(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Downloaded checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, "
            f"found {actual}. Remove the cached file and download it again."
        )


def resolve_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Return an explicit checkpoint or download the published LoRetta weights."""
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    try:
        path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=CHECKPOINT_FILENAME,
                revision=MODEL_REVISION,
            )
        )
    except GatedRepoError as error:
        raise RuntimeError(
            f"Access to {MODEL_REPO_ID} is gated. Open the model page, accept the "
            "DINOv3 license and responsible-use notice, then run `hf auth login`. "
            "Alternatively, pass --checkpoint with an authorized local file."
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"Could not download {MODEL_REPO_ID}/{CHECKPOINT_FILENAME}. "
            "Check the network connection and Hugging Face authentication, "
            "or pass --checkpoint with a local file."
        ) from error
    _verify_checkpoint_hash(path)
    return path


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
