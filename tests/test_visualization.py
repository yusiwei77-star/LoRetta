import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from loretta.inference.visualization import clear_geometry_outputs, save_dense_outputs


class VisualizationTests(unittest.TestCase):
    def test_clear_geometry_outputs_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in (
                "ransac_inliers.png",
                "homography_registered.png",
                "dense_registered.png",
            ):
                (output / name).touch()
            clear_geometry_outputs(output)
            self.assertFalse((output / "ransac_inliers.png").exists())
            self.assertFalse((output / "homography_registered.png").exists())
            self.assertTrue((output / "dense_registered.png").exists())

    def test_dense_registration_preserves_original_sensed_pixels(self):
        sensed = np.arange(32 * 64 * 3, dtype=np.uint8).reshape(32, 64, 3)
        coordinates = (np.arange(16, dtype=np.float32) + 0.5) * 2.0 / 16.0 - 1.0
        x, y = np.meshgrid(coordinates, coordinates)
        warp = np.stack((x, y), axis=-1)
        matchability = np.ones((16, 16), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_dense_outputs(
                sensed_image=sensed,
                warp=warp,
                matchability=matchability,
                reference_size=(64, 32),
                output_dir=output,
            )
            with Image.open(output / "dense_registered.png") as image:
                registered = np.asarray(image)
            with Image.open(output / "matchability.png") as image:
                self.assertEqual(image.size, (64, 32))

        np.testing.assert_array_equal(registered, sensed)
