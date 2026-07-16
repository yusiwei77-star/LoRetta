import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.local_correlation import local_correlation


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, *, bias: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_dim, out_dim, kernel_size, padding=kernel_size // 2, bias=bias
        )
        self.norm1 = nn.BatchNorm2d(out_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 1)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu1(self.norm1(self.conv1(features))))
        return F.relu(residual + self.skip(features))


class ResidualRefiner(nn.Module):
    """Paper refiner R_k for residual warp and matchability prediction."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        kernel_size: int,
        hidden_blocks: int,
        local_correlation_radius: int,
        residual_embedding_dim: int,
    ):
        super().__init__()
        self.residual_embedding = nn.Sequential(
            nn.Conv2d(3, residual_embedding_dim, 1),
            nn.GELU(),
            nn.Conv2d(residual_embedding_dim, residual_embedding_dim, 1),
        )
        self.block1 = ResidualBlock(in_dim, hidden_dim, kernel_size, bias=True)
        self.hidden_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, hidden_dim, kernel_size, bias=False)
                for _ in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, 3, 1)
        self.local_correlation_radius = local_correlation_radius

    def forward(
        self,
        reference_features: torch.Tensor,
        sensed_features: torch.Tensor,
        residual_warp: torch.Tensor,
        matchability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = reference_features.shape
        device, dtype = residual_warp.device, residual_warp.dtype
        row_indices, column_indices = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        scale_x, scale_y = 2.0 / width, 2.0 / height
        reference_grid = torch.stack(
            (
                column_indices * scale_x - (1.0 - scale_x / 2.0),
                row_indices * scale_y - (1.0 - scale_y / 2.0),
            ),
            dim=-1,
        ).unsqueeze(0).expand(batch_size, -1, -1, -1)
        residual_grid = reference_grid + residual_warp.permute(0, 2, 3, 1)
        sampled_sensed_features = F.grid_sample(
            sensed_features, residual_grid, align_corners=False, mode="bilinear"
        )
        local_correlation_volume = local_correlation(
            reference_features,
            sensed_features,
            local_radius=self.local_correlation_radius,
            sampling_grid=residual_grid,
            sample_mode="bilinear",
        )
        residual_embedding = self.residual_embedding(
            torch.cat((residual_warp, matchability), dim=1)
        )
        refiner_input = torch.cat(
            (
                reference_features,
                sampled_sensed_features,
                local_correlation_volume,
                residual_embedding,
            ),
            dim=1,
        )
        prediction = self.out_conv(self.hidden_blocks(self.block1(refiner_input)).float())
        return prediction[:, :2], torch.sigmoid(prediction[:, 2:])


class DenseRegister(nn.Module):
    """Localization-guided three-scale dense register from the LoRetta paper."""

    def __init__(self, config: dict):
        super().__init__()
        feature_dims = config["feature_dims"]
        hidden_dims = config["hidden_dims"]
        correlation_radii = config["local_correlation_radii"]
        residual_embedding_dims = config["residual_embedding_dims"]
        kernel_size = config["kernel_size"]
        hidden_blocks = config["hidden_blocks"]

        def build_refiner(index: int) -> ResidualRefiner:
            feature_dim = feature_dims[4 - index]
            input_dim = (
                2 * feature_dim
                + (2 * correlation_radii[index] + 1) ** 2
                + residual_embedding_dims[index]
            )
            return ResidualRefiner(
                in_dim=input_dim,
                hidden_dim=hidden_dims[index],
                kernel_size=kernel_size,
                hidden_blocks=hidden_blocks,
                local_correlation_radius=correlation_radii[index],
                residual_embedding_dim=residual_embedding_dims[index],
            )

        self.refiner0 = build_refiner(0)
        self.refiner1 = build_refiner(1)
        self.refiner2 = build_refiner(2)

    def forward(
        self,
        reference_features: torch.Tensor,
        sensed_features: torch.Tensor,
        residual_warp: torch.Tensor,
        matchability: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if level not in (0, 1, 2):
            raise ValueError(f"Unsupported LoRetta refiner level: {level}")
        refiner = (self.refiner0, self.refiner1, self.refiner2)[level]
        return refiner(reference_features, sensed_features, residual_warp, matchability)
