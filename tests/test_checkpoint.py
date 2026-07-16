from collections import OrderedDict
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.inference.checkpoint import canonicalize_state_dict, load_checkpoint


class LegacyMetadata:
    pass


class CheckpointTests(unittest.TestCase):
    def test_loads_lightning_model_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 1)
            state = OrderedDict(
                (f"model.{key}", value.clone())
                for key, value in model.state_dict().items()
            )
            path = Path(directory) / "model.ckpt"
            torch.save({"state_dict": state}, path)
            self.assertEqual(load_checkpoint(model, path), 2)

    def test_rejects_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 1)
            path = Path(directory) / "bad.ckpt"
            torch.save({"state_dict": {"model.weight": torch.zeros(3, 3)}}, path)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                load_checkpoint(model, path)

    def test_migrates_legacy_loretta_prefixes(self):
        legacy = OrderedDict(
            [
                ("model.f.norm.weight", torch.ones(2)),
                ("model.matcher.matcher.null_logit", torch.zeros(())),
                ("model.matcher.matcher.decoder.flow.bias", torch.zeros(2)),
                ("model.matcher.matcher.decoder.conf.bias", torch.zeros(1)),
                (
                    "model.refiner.refiner4.displacement_proj.0.bias",
                    torch.zeros(2),
                ),
                ("model.refiner.refiner4.out_conv.bias", torch.zeros(3)),
            ]
        )
        canonical = canonicalize_state_dict(legacy)
        self.assertEqual(
            list(canonical),
            [
                "dinov3_feature_extractor.norm.weight",
                "affine_locator.no_match_logit",
                "affine_locator.match_embedding_decoder.warp_head.bias",
                "affine_locator.match_embedding_decoder.matchability_head.bias",
                "dense_register.refiner0.residual_embedding.0.bias",
                "dense_register.refiner0.out_conv.bias",
            ],
        )
        for original, migrated in zip(legacy.values(), canonical.values()):
            self.assertTrue(torch.equal(original, migrated))

    def test_removes_nested_wrapper_prefixes(self):
        state = {"module.model.weight": torch.ones(2)}
        self.assertEqual(list(canonicalize_state_dict(state)), ["weight"])

    def test_rejects_unsafe_legacy_metadata_during_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.ckpt"
            torch.save(
                {
                    "state_dict": {"model.weight": torch.ones(1)},
                    "metadata": LegacyMetadata(),
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "migrate_checkpoint.py"):
                load_checkpoint(nn.Linear(1, 1), path)
