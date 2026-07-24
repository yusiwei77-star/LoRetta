import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from httpx import Request, Response
from huggingface_hub.errors import GatedRepoError
from torch import nn

from loretta.inference.checkpoint import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    MODEL_REPO_ID,
    MODEL_REVISION,
    load_checkpoint,
    resolve_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def test_explicit_checkpoint_does_not_download(self):
        with patch("loretta.inference.checkpoint.hf_hub_download") as download:
            self.assertEqual(resolve_checkpoint("local.pth"), Path("local.pth"))
            download.assert_not_called()

    @patch("loretta.inference.checkpoint._verify_checkpoint_hash")
    @patch("loretta.inference.checkpoint.hf_hub_download", return_value="cached.pth")
    def test_default_checkpoint_downloads_from_hugging_face(self, download, verify):
        self.assertEqual(resolve_checkpoint(), Path("cached.pth"))
        download.assert_called_once_with(
            repo_id=MODEL_REPO_ID,
            filename=CHECKPOINT_FILENAME,
            revision=MODEL_REVISION,
        )
        verify.assert_called_once_with(Path("cached.pth"))

    @patch(
        "loretta.inference.checkpoint.hf_hub_download",
        side_effect=OSError("offline"),
    )
    def test_download_error_has_local_checkpoint_fallback(self, _download):
        with self.assertRaisesRegex(RuntimeError, "--checkpoint"):
            resolve_checkpoint()

    @patch("loretta.inference.checkpoint.hf_hub_download")
    def test_gated_download_explains_license_and_authentication(self, download):
        response = Response(403, request=Request("GET", "https://huggingface.co"))
        download.side_effect = GatedRepoError("gated", response=response)
        with self.assertRaisesRegex(RuntimeError, "hf auth login") as raised:
            resolve_checkpoint()
        self.assertIn("model page", str(raised.exception))
        self.assertIn("DINOv3 license", str(raised.exception))

    def test_checkpoint_hash_matches_published_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"canonical checkpoint")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("loretta.inference.checkpoint.CHECKPOINT_SHA256", expected):
                from loretta.inference.checkpoint import _verify_checkpoint_hash

                _verify_checkpoint_hash(path)

    def test_checkpoint_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"tampered")
            from loretta.inference.checkpoint import _verify_checkpoint_hash

            with self.assertRaisesRegex(RuntimeError, CHECKPOINT_SHA256):
                _verify_checkpoint_hash(path)

    def test_loads_canonical_tensor_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 1)
            path = Path(directory) / "model.pth"
            torch.save(model.state_dict(), path)
            self.assertEqual(load_checkpoint(model, path), 2)

    def test_rejects_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 1)
            path = Path(directory) / "bad.pth"
            torch.save({"weight": torch.zeros(3, 3)}, path)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                load_checkpoint(model, path)

    def test_rejects_wrapped_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 1)
            path = Path(directory) / "wrapped.pth"
            torch.save({"state_dict": model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "canonical LoRetta"):
                load_checkpoint(model, path)
