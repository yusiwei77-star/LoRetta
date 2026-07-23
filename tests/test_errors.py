import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loretta.inference import load_rgb_image, resolve_device, run_demo


class ErrorHandlingTests(unittest.TestCase):
    def test_missing_image_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Image does not exist"):
            load_rgb_image("definitely_missing_image.png")

    def test_non_image_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-an-image.png"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Could not decode image"):
                load_rgb_image(path)

    @patch("loretta.inference.image.torch.cuda.is_available", return_value=False)
    def test_requested_cuda_must_exist(self, _is_available):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            resolve_device("cuda")

    def test_run_demo_validates_geometry_options_before_loading_files(self):
        with self.assertRaisesRegex(ValueError, "matchability_threshold"):
            run_demo(
                reference_path="missing-reference.png",
                sensed_path="missing-sensed.png",
                checkpoint_path="missing.pth",
                output_dir="outputs",
                matchability_threshold=1.1,
            )
