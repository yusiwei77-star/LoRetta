import unittest
from unittest.mock import patch

import numpy as np

from loretta.inference import estimate_homography, normalized_warp_to_pixels


class GeometryTests(unittest.TestCase):
    def test_normalized_warp_to_pixels_uses_pixel_centers(self):
        warp = np.asarray(
            [[[-1 + 1 / 2, -1 + 1 / 2], [1 - 1 / 2, 1 - 1 / 2]]],
            dtype=np.float32,
        )
        pixels = normalized_warp_to_pixels(warp, (2, 2))
        np.testing.assert_allclose(pixels, [[[0, 0], [1, 1]]], atol=1e-6)

    def test_normalized_warp_to_pixels_uses_width_height_order(self):
        warp = np.asarray([[[1.0, 1.0]]], dtype=np.float32)
        pixels = normalized_warp_to_pixels(warp, (8, 4))
        np.testing.assert_allclose(pixels, [[[7.5, 3.5]]], atol=1e-6)

    def test_estimate_homography_sensed_to_reference(self):
        height = width = 32
        rows, columns = np.mgrid[:height, :width]
        warp_pixels = np.stack((columns + 5, rows + 3), axis=-1).astype(np.float32)
        matchability = np.ones((height, width), dtype=np.float32)
        result = estimate_homography(
            warp_pixels,
            matchability,
            reference_size=(width, height),
            sensed_size=(width, height),
            matchability_threshold=0.5,
            max_keypoints=128,
            ransac_threshold=0.1,
        )
        self.assertTrue(result["success"])
        matrix = result["matrix_sensed_to_reference"]
        matrix = matrix / matrix[2, 2]
        np.testing.assert_allclose(matrix[:2, 2], [-5, -3], atol=1e-3)

    def test_estimate_homography_reports_too_few_points(self):
        result = estimate_homography(
            np.zeros((2, 2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            reference_size=(2, 2),
            sensed_size=(2, 2),
            matchability_threshold=0.5,
            max_keypoints=10,
            ransac_threshold=3.0,
        )
        self.assertEqual(
            result,
            {
                "success": False,
                "reason": "fewer_than_four_correspondences",
                "num_candidates": 0,
            },
        )

    @patch("loretta.inference.geometry.np.random.choice", wraps=np.random.choice)
    def test_correspondences_are_sampled_without_replacement(self, choice):
        height = width = 32
        rows, columns = np.mgrid[:height, :width]
        warp_pixels = np.stack((columns, rows), axis=-1).astype(np.float32)
        matchability = np.ones((height, width), dtype=np.float32)
        result = estimate_homography(
            warp_pixels,
            matchability,
            reference_size=(width, height),
            sensed_size=(width, height),
            matchability_threshold=0.5,
            max_keypoints=64,
            ransac_threshold=0.1,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["num_candidates"], 64)
        self.assertEqual(choice.call_args.args[1], 64)
        self.assertFalse(choice.call_args.kwargs["replace"])
