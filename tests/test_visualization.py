import tempfile
import unittest
from pathlib import Path

from src.inference.visualization import clear_geometry_outputs


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
