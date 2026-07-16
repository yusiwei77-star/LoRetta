import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.inference.checkpoint import load_checkpoint


class CheckpointTests(unittest.TestCase):
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
