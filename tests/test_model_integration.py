import os
import unittest
from pathlib import Path

import torch

from src.config import get_inference_config
from src.inference.checkpoint import load_checkpoint
from src.inference.image import image_to_tensor, load_rgb_image
from src.loretta import LoRetta


CHECKPOINT = os.environ.get("LORETTA_CHECKPOINT")
DEVICE = os.environ.get(
    "LORETTA_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(CHECKPOINT, "set LORETTA_CHECKPOINT to run model integration tests")
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(DEVICE)
        cls.model = LoRetta(get_inference_config(enable_amp=cls.device.type == "cuda"))

    @classmethod
    def tearDownClass(cls):
        del cls.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_checkpoint_has_exactly_the_expected_807_keys(self):
        self.assertEqual(len(self.model.state_dict()), 807)
        self.assertEqual(load_checkpoint(self.model, CHECKPOINT), 807)

    def test_default_pair_output_contract(self):
        self.assertEqual(load_checkpoint(self.model, CHECKPOINT), 807)
        self.model.eval().to(self.device)
        reference, _ = image_to_tensor(
            load_rgb_image(PROJECT_ROOT / "assets" / "reference.jpg"), self.device
        )
        sensed, _ = image_to_tensor(
            load_rgb_image(PROJECT_ROOT / "assets" / "sensed.jpg"), self.device
        )
        batch = {"reference_image": reference, "sensed_image": sensed}
        with torch.inference_mode():
            returned = self.model(batch)

        self.assertIs(returned, batch)
        self.assertEqual(
            set(batch),
            {"reference_image", "sensed_image", "warp_1", "matchability_1"},
        )
        self.assertEqual(tuple(batch["warp_1"].shape), (1, 2, 512, 512))
        self.assertEqual(tuple(batch["matchability_1"].shape), (1, 1, 512, 512))
        self.assertTrue(torch.isfinite(batch["warp_1"]).all())
        self.assertTrue(torch.isfinite(batch["matchability_1"]).all())
