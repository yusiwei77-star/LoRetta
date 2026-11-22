import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry import make_normalized_grid
from .affine_fitter import MatchabilityWeightedAffineFitter
from .match_embedding import build_match_embeddings
from .vit import MultiViewTransformer


class MatchEmbeddingDecoder(nn.Module):
    """Decode match embeddings into the paper's coarse warp and matchability maps."""

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.warp_head = nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        self.matchability_head = nn.Conv2d(hidden_dim, 1, 3, 1, 1)
        nn.init.zeros_(self.warp_head.weight)
        nn.init.zeros_(self.warp_head.bias)
        nn.init.zeros_(self.matchability_head.weight)
        nn.init.zeros_(self.matchability_head.bias)

    def forward(self, match_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = self.net(match_embeddings.permute(0, 3, 1, 2).contiguous())
        coarse_warp = self.warp_head(decoded)
        coarse_matchability = torch.sigmoid(self.matchability_head(decoded))
        return coarse_warp, coarse_matchability


class AffineLocator(nn.Module):
    """Matchability-weighted affine locator described in the LoRetta paper."""

    descriptor_dim = 1024
    localization_dim = 1024

    def __init__(self, config: dict):
        super().__init__()
        self.enable_amp = bool(config["enable_amp"])
        self.position_frequencies = nn.Buffer(
            2 * torch.pi * torch.randn(self.localization_dim // 2, 2)
        )
        self.position_scale = nn.Buffer(torch.tensor(1.0))
        self.temperature = nn.Buffer(torch.tensor(0.1))
        self.multi_view_transformer = MultiViewTransformer(
            in_dim=2 * self.descriptor_dim,
            out_dim=self.localization_dim,
        )
        self.no_match_embedding = nn.Parameter(torch.zeros(self.localization_dim))
        self.no_match_logit = nn.Parameter(torch.tensor(0.0))
        self.match_embedding_decoder = MatchEmbeddingDecoder(
            dim=self.localization_dim, hidden_dim=256
        )
        self.affine_fitter = MatchabilityWeightedAffineFitter()

    @staticmethod
    def _pixel_affine_to_normalized(
        affine_prior: torch.Tensor,
        *,
        reference_hw: tuple[int, int],
        sensed_hw: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = affine_prior.shape[0]
        reference_height, reference_width = reference_hw
        sensed_height, sensed_width = sensed_hw
        device = affine_prior.device
        reference_scale_x, reference_scale_y = 2.0 / reference_width, 2.0 / reference_height
        sensed_scale_x, sensed_scale_y = 2.0 / sensed_width, 2.0 / sensed_height
        sensed_pixel_to_normalized = torch.tensor(
            [
                [sensed_scale_x, 0.0, -1.0 + sensed_scale_x / 2.0],
                [0.0, sensed_scale_y, -1.0 + sensed_scale_y / 2.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 3, 3)
        reference_normalized_to_pixel = torch.tensor(
            [
                [
                    1.0 / reference_scale_x,
                    0.0,
                    (1.0 - reference_scale_x / 2.0) / reference_scale_x,
                ],
                [
                    0.0,
                    1.0 / reference_scale_y,
                    (1.0 - reference_scale_y / 2.0) / reference_scale_y,
                ],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 3, 3)
        normalized_affine_prior = torch.bmm(
            sensed_pixel_to_normalized.expand(batch_size, -1, -1),
            torch.bmm(
                affine_prior.to(dtype),
                reference_normalized_to_pixel.expand(batch_size, -1, -1),
            ),
        )
        return normalized_affine_prior[:, :2, :]

    def forward(
        self,
        reference_descriptors: torch.Tensor,
        sensed_descriptors: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            reference_descriptors.shape != sensed_descriptors.shape
            or reference_descriptors.shape[1] != 2 * self.descriptor_dim
        ):
            raise ValueError("AffineLocator descriptor shapes do not match")
        reference_layers = [
            layer.permute(0, 2, 3, 1)
            for layer in reference_descriptors.split(self.descriptor_dim, dim=1)
        ]
        sensed_layers = [
            layer.permute(0, 2, 3, 1)
            for layer in sensed_descriptors.split(self.descriptor_dim, dim=1)
        ]
        reference_descriptors = torch.cat(reference_layers, dim=-1)
        sensed_descriptors = torch.cat(sensed_layers, dim=-1)
        batch_size, height, width, _ = reference_descriptors.shape

        with torch.autocast(
            reference_descriptors.device.type,
            torch.bfloat16,
            enabled=self.enable_amp,
        ):
            contextualized = self.multi_view_transformer(
                torch.stack((reference_descriptors, sensed_descriptors), dim=1)
            )
        contextualized_reference = contextualized[:, 0].float()
        contextualized_sensed = contextualized[:, 1].float()

        reference_grid = make_normalized_grid(
            batch_size, height, width, device=reference_descriptors.device
        )
        position_projection = F.linear(
            reference_grid.reshape(batch_size, height * width, 2),
            (self.position_scale * self.position_frequencies).to(
                device=reference_descriptors.device,
                dtype=reference_grid.dtype,
            ),
        ).reshape(batch_size, height, width, -1)
        sensed_position_embeddings = torch.cat(
            (position_projection.sin(), position_projection.cos()), dim=-1
        )
        match_embeddings = build_match_embeddings(
            reference_descriptors=contextualized_reference,
            sensed_descriptors=contextualized_sensed,
            sensed_position_embeddings=sensed_position_embeddings,
            temperature=self.temperature,
            no_match_embedding=self.no_match_embedding,
            no_match_logit=self.no_match_logit,
        )
        coarse_warp, coarse_matchability = self.match_embedding_decoder(match_embeddings)
        affine_prior_pixels, fit_succeeded = self.affine_fitter.fit(
            coarse_warp, coarse_matchability, image_hw
        )
        if not fit_succeeded:
            affine_prior_pixels = torch.eye(
                3, device=coarse_warp.device, dtype=coarse_warp.dtype
            ).view(1, 3, 3).expand(batch_size, -1, -1)
        affine_prior = self._pixel_affine_to_normalized(
            affine_prior_pixels,
            reference_hw=image_hw,
            sensed_hw=image_hw,
            dtype=coarse_warp.dtype,
        )
        return affine_prior, coarse_matchability
