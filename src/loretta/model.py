import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import VGG19FeaturePyramid, build_dinov3_feature_extractor
from .module import AffineLocator, DenseRegister


def _to_homogeneous(affine_prior: torch.Tensor) -> torch.Tensor:
    bottom_row = torch.tensor(
        (0.0, 0.0, 1.0), device=affine_prior.device, dtype=affine_prior.dtype
    ).view(1, 1, 3).expand(affine_prior.shape[0], -1, -1)
    return torch.cat((affine_prior, bottom_row), dim=1)


def _invert_affine(affine_prior: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(_to_homogeneous(affine_prior).float()).to(
        affine_prior.dtype
    )[:, :2]


class LoRetta(nn.Module):
    """Localization-and-registration model with the paper's inference interface."""

    def __init__(self, config: dict):
        super().__init__()
        extractor_config = config["dinov3_feature_extractor"]
        self.dinov3_feature_extractor = build_dinov3_feature_extractor(
            enable_amp=extractor_config["enable_amp"],
            layer_indices=tuple(extractor_config["layer_indices"]),
        )
        self.affine_locator = AffineLocator(config["affine_locator"])
        self.vgg_feature_pyramid = VGG19FeaturePyramid()
        self.dense_register = DenseRegister(config["dense_register"])

    @staticmethod
    def _prepare_dinov3_images(
        reference_image: torch.Tensor,
        sensed_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(
            (0.430, 0.411, 0.296), device=reference_image.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            (0.213, 0.156, 0.143), device=reference_image.device
        ).view(1, 3, 1, 1)
        return (
            torch.clamp(reference_image * std + mean, 0.0, 1.0),
            torch.clamp(sensed_image * std + mean, 0.0, 1.0),
        )

    @staticmethod
    def _build_affine_warp(
        affine_prior: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, device, dtype = (
            affine_prior.shape[0],
            affine_prior.device,
            affine_prior.dtype,
        )
        normalized_y = torch.linspace(
            -1.0 + 1.0 / height,
            1.0 - 1.0 / height,
            height,
            device=device,
            dtype=dtype,
        )
        normalized_x = torch.linspace(
            -1.0 + 1.0 / width,
            1.0 - 1.0 / width,
            width,
            device=device,
            dtype=dtype,
        )
        reference_y, reference_x = torch.meshgrid(
            normalized_y, normalized_x, indexing="ij"
        )
        reference_grid = torch.stack((reference_x, reference_y), dim=-1).reshape(
            1, height * width, 2
        ).expand(batch_size, -1, -1)
        homogeneous_reference_grid = torch.cat(
            (
                reference_grid,
                torch.ones(
                    (batch_size, height * width, 1),
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=-1,
        )
        return torch.bmm(
            homogeneous_reference_grid, affine_prior.transpose(1, 2)
        ).reshape(batch_size, height, width, 2).permute(0, 3, 1, 2).contiguous()

    def forward(self, batch: dict) -> dict:
        reference_image = batch["reference_image"]
        sensed_image = batch["sensed_image"]
        if reference_image.shape != sensed_image.shape:
            raise ValueError(
                "LoRetta requires reference and sensed tensors with identical shapes"
            )
        batch_size, _, height, width = reference_image.shape

        reference_for_dinov3, sensed_for_dinov3 = self._prepare_dinov3_images(
            reference_image, sensed_image
        )
        coarse_reference_descriptors = torch.cat(
            self.dinov3_feature_extractor(reference_for_dinov3), dim=-1
        ).permute(0, 3, 1, 2).contiguous()
        coarse_sensed_descriptors = torch.cat(
            self.dinov3_feature_extractor(sensed_for_dinov3), dim=-1
        ).permute(0, 3, 1, 2).contiguous()
        affine_prior, coarse_matchability = self.affine_locator(
            coarse_reference_descriptors,
            coarse_sensed_descriptors,
            (height, width),
        )

        affine_warp = self._build_affine_warp(affine_prior, height, width)
        inverse_affine_prior = _invert_affine(affine_prior).to(
            device=reference_image.device, dtype=reference_image.dtype
        )
        inverse_affine_grid = F.affine_grid(
            inverse_affine_prior,
            size=(batch_size, 3, height, width),
            align_corners=False,
        )
        affine_warped_reference = F.grid_sample(
            reference_image,
            inverse_affine_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        coarse_matchability_full = F.interpolate(
            coarse_matchability, size=(height, width), mode="nearest"
        )
        affine_aligned_matchability = F.grid_sample(
            coarse_matchability_full,
            inverse_affine_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        feature_pyramid = self.vgg_feature_pyramid(
            torch.cat((affine_warped_reference, sensed_image), dim=0)
        )
        affine_reference_pyramid = []
        sensed_feature_pyramid = []
        for level_features in feature_pyramid:
            affine_reference_features, sensed_features = level_features.split(batch_size)
            affine_reference_pyramid.append(affine_reference_features)
            sensed_feature_pyramid.append(sensed_features)

        matchability = F.interpolate(
            affine_aligned_matchability,
            size=(height // 16, width // 16),
            mode="nearest",
        )
        residual_warp = torch.zeros(
            (batch_size, 2, height // 16, width // 16),
            device=reference_image.device,
            dtype=reference_image.dtype,
        )
        residual_warp, matchability = self.dense_register(
            affine_reference_pyramid[0],
            sensed_feature_pyramid[0],
            residual_warp,
            matchability,
            0,
        )
        residual_warp = F.interpolate(
            residual_warp,
            size=(height // 8, width // 8),
            mode="bilinear",
            align_corners=False,
        )
        matchability = F.interpolate(
            matchability, size=(height // 8, width // 8), mode="nearest"
        )
        residual_warp, matchability = self.dense_register(
            affine_reference_pyramid[1],
            sensed_feature_pyramid[1],
            residual_warp,
            matchability,
            1,
        )
        residual_warp = F.interpolate(
            residual_warp,
            size=(height // 4, width // 4),
            mode="bilinear",
            align_corners=False,
        )
        matchability = F.interpolate(
            matchability, size=(height // 4, width // 4), mode="nearest"
        )
        residual_warp, matchability = self.dense_register(
            affine_reference_pyramid[2],
            sensed_feature_pyramid[2],
            residual_warp,
            matchability,
            2,
        )

        residual_warp = F.interpolate(
            residual_warp,
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=False,
        )
        matchability = F.interpolate(
            matchability, size=(height // 2, width // 2), mode="nearest"
        )
        residual_warp = F.interpolate(
            residual_warp,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        matchability = F.interpolate(
            matchability, size=(height, width), mode="nearest"
        )
        affine_aligned_grid = affine_warp.permute(0, 2, 3, 1)
        sampled_residual_warp = F.grid_sample(
            residual_warp,
            affine_aligned_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        final_matchability = F.grid_sample(
            matchability,
            affine_aligned_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        batch["warp_1"] = affine_warp + sampled_residual_warp
        batch["matchability_1"] = final_matchability
        return batch
