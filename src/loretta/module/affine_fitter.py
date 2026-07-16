"""Matchability-weighted affine fitting for LoRetta's affine locator."""

import torch


class MatchabilityWeightedAffineFitter:
    """Fit reference-to-sensed affine priors with batched weighted least squares."""

    matchability_threshold = 0.3
    minimum_correspondences = 10

    def fit(
        self,
        coarse_warp: torch.Tensor,
        coarse_matchability: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, bool]:
        batch_size = coarse_warp.shape[0]
        image_height, image_width = int(image_hw[0]), int(image_hw[1])
        coarse_height, coarse_width = coarse_warp.shape[2:]
        device, dtype = coarse_warp.device, coarse_warp.dtype

        row_indices, column_indices = torch.meshgrid(
            torch.arange(coarse_height, device=device, dtype=dtype),
            torch.arange(coarse_width, device=device, dtype=dtype),
            indexing="ij",
        )
        reference_x = (column_indices + 0.5) * (float(image_width) / coarse_width)
        reference_y = (row_indices + 0.5) * (float(image_height) / coarse_height)
        reference_x = reference_x.unsqueeze(0).expand(batch_size, -1, -1)
        reference_y = reference_y.unsqueeze(0).expand(batch_size, -1, -1)

        scale_x, scale_y = 2.0 / image_width, 2.0 / image_height
        sensed_x = (coarse_warp[:, 0] + (1.0 - scale_x / 2.0)) / scale_x
        sensed_y = (coarse_warp[:, 1] + (1.0 - scale_y / 2.0)) / scale_y

        point_count = coarse_height * coarse_width
        reference_points = torch.stack(
            (
                reference_x.reshape(batch_size, point_count),
                reference_y.reshape(batch_size, point_count),
            ),
            dim=-1,
        )
        sensed_points = torch.stack(
            (sensed_x.reshape(batch_size, point_count), sensed_y.reshape(batch_size, point_count)),
            dim=-1,
        )
        weights = coarse_matchability.reshape(batch_size, point_count)
        weights = torch.where(
            weights > self.matchability_threshold, weights, torch.zeros_like(weights)
        )

        valid_correspondences = (weights > 0).sum(dim=1).to(torch.long)
        sufficient = valid_correspondences >= self.minimum_correspondences
        identity = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).expand(batch_size, -1, -1)
        if not bool(sufficient.any()):
            return identity.clone(), False

        affine_priors, fit_succeeded = self._weighted_least_squares(
            reference_points, sensed_points, weights
        )
        if bool((~sufficient).any()):
            affine_priors = torch.where(
                sufficient.view(batch_size, 1, 1), affine_priors, identity
            )
        return affine_priors, fit_succeeded

    @staticmethod
    def _weighted_least_squares(
        reference_points: torch.Tensor,
        sensed_points: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        batch_size, point_count = reference_points.shape[:2]
        device, dtype = reference_points.device, reference_points.dtype
        reference_x, reference_y = reference_points[:, :, 0], reference_points[:, :, 1]
        sensed_x, sensed_y = sensed_points[:, :, 0], sensed_points[:, :, 1]

        design = torch.zeros((batch_size, 2 * point_count, 6), dtype=dtype, device=device)
        targets = torch.zeros((batch_size, 2 * point_count), dtype=dtype, device=device)
        design[:, 0::2, 0] = reference_x
        design[:, 0::2, 1] = reference_y
        design[:, 0::2, 2] = 1.0
        targets[:, 0::2] = sensed_x
        design[:, 1::2, 3] = reference_x
        design[:, 1::2, 4] = reference_y
        design[:, 1::2, 5] = 1.0
        targets[:, 1::2] = sensed_y

        sqrt_weights = torch.sqrt(torch.clamp(weights, min=1e-8))
        row_weights = torch.zeros((batch_size, 2 * point_count), dtype=dtype, device=device)
        row_weights[:, 0::2] = sqrt_weights
        row_weights[:, 1::2] = sqrt_weights
        weighted_design = design * row_weights.unsqueeze(-1)
        weighted_targets = targets * row_weights

        normal_matrix = torch.bmm(weighted_design.transpose(1, 2), weighted_design)
        normal_target = torch.bmm(
            weighted_design.transpose(1, 2), weighted_targets.unsqueeze(-1)
        ).squeeze(-1)
        normal_matrix = normal_matrix + 1e-4 * torch.eye(
            6, device=device, dtype=normal_matrix.dtype
        ).unsqueeze(0)

        fit_succeeded = True
        try:
            solution = torch.linalg.solve(normal_matrix.float(), normal_target.float()).to(dtype)
        except RuntimeError:
            fit_succeeded = False
            solution = torch.zeros(batch_size, 6, dtype=dtype, device=device)
            solution[:, 0] = 1.0
            solution[:, 4] = 1.0

        affine_priors = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
        affine_priors[:, :2, :] = solution.reshape(batch_size, 2, 3)
        affine_priors[:, 2, 2] = 1.0
        return affine_priors, fit_succeeded
