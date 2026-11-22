import unittest

import torch

from loretta.module.affine_fitter import MatchabilityWeightedAffineFitter


class AffineFitterTests(unittest.TestCase):
    def test_batched_fit_keeps_insufficient_items_at_identity(self):
        height, width = 4, 6
        rows, columns = torch.meshgrid(
            torch.linspace(-1 + 1 / height, 1 - 1 / height, height),
            torch.linspace(-1 + 1 / width, 1 - 1 / width, width),
            indexing="ij",
        )
        identity_warp = torch.stack((columns, rows), dim=0)
        coarse_warp = identity_warp.unsqueeze(0).expand(2, -1, -1, -1).clone()
        matchability = torch.ones(2, 1, height, width)
        matchability[1] = 0

        affine_priors, succeeded = MatchabilityWeightedAffineFitter().fit(
            coarse_warp,
            matchability,
            image_hw=(32, 60),
        )

        self.assertTrue(succeeded)
        self.assertEqual(tuple(affine_priors.shape), (2, 3, 3))
        torch.testing.assert_close(affine_priors[1], torch.eye(3), atol=0, rtol=0)
        torch.testing.assert_close(affine_priors[0], torch.eye(3), atol=1e-4, rtol=1e-4)
