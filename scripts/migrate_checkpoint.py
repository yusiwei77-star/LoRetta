"""Convert a trusted legacy LoRetta Lightning checkpoint to a canonical state dict."""

import argparse
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint import (  # noqa: E402
    canonicalize_state_dict,
    extract_state_dict,
)


def load_trusted_legacy_checkpoint(path: Path):
    """Load a user-confirmed trusted checkpoint for one-time schema migration."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def migrate_checkpoint(source: Path, destination: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_name(destination.name + ".new")
    canonical = canonicalize_state_dict(
        extract_state_dict(load_trusted_legacy_checkpoint(source))
    )
    if len(canonical) != 807:
        raise RuntimeError(f"Expected 807 LoRetta state entries, found {len(canonical)}")
    try:
        torch.save(canonical, temporary_output)
        reloaded = torch.load(
            temporary_output, map_location="cpu", weights_only=True, mmap=True
        )
        if list(reloaded) != list(canonical):
            raise RuntimeError("Migrated checkpoint key order changed during serialization")
        for key, value in canonical.items():
            if not torch.equal(reloaded[key], value):
                raise RuntimeError(f"Migrated tensor differs from source: {key}")
        os.replace(temporary_output, destination)
    finally:
        temporary_output.unlink(missing_ok=True)
    return len(canonical)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a trusted legacy LoRetta checkpoint to canonical paper names"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = migrate_checkpoint(args.source, args.destination)
    print(f"Migrated {count} state entries to {args.destination}")


if __name__ == "__main__":
    main()
