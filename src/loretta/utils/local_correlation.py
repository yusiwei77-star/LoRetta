import torch
import torch.nn.functional as F


def local_correlation(
    reference_features: torch.Tensor,
    sensed_features: torch.Tensor,
    *,
    local_radius: int,
    sampling_grid: torch.Tensor | None = None,
    padding_mode: str = "zeros",
    sample_mode: str = "bilinear",
) -> torch.Tensor:
    """Build the paper's local correlation volume around a residual sampling grid."""
    window_width = 2 * local_radius + 1
    window_size = window_width**2
    batch_size, channels, height, width = reference_features.shape
    correlation_volume = torch.empty(
        (batch_size, window_size, height, width),
        device=reference_features.device,
        dtype=reference_features.dtype,
    )
    if sampling_grid is None:
        rows, columns = torch.meshgrid(
            torch.linspace(
                -1 + 1 / height,
                1 - 1 / height,
                height,
                device=reference_features.device,
            ),
            torch.linspace(
                -1 + 1 / width,
                1 - 1 / width,
                width,
                device=reference_features.device,
            ),
            indexing="ij",
        )
        sampling_grid = torch.stack((columns, rows), dim=-1)[None].expand(
            batch_size, height, width, 2
        )

    row_offsets, column_offsets = torch.meshgrid(
        torch.linspace(
            -2 * local_radius / height,
            2 * local_radius / height,
            window_width,
            device=reference_features.device,
        ),
        torch.linspace(
            -2 * local_radius / width,
            2 * local_radius / width,
            window_width,
            device=reference_features.device,
        ),
        indexing="ij",
    )
    local_offsets = torch.stack((column_offsets, row_offsets), dim=-1)[None].expand(
        1, window_width, window_width, 2
    ).reshape(1, window_size, 2)
    for batch_index in range(batch_size):
        with torch.no_grad():
            local_sampling_grid = (
                sampling_grid[batch_index, :, :, None] + local_offsets[:, None, None]
            ).reshape(1, height, width * window_size, 2)
            window_features = F.grid_sample(
                sensed_features[batch_index : batch_index + 1],
                local_sampling_grid,
                padding_mode=padding_mode,
                align_corners=False,
                mode=sample_mode,
            ).reshape(channels, height, width, window_size)
        correlation_volume[batch_index] = (
            reference_features[batch_index, ..., None]
            / (channels**0.5)
            * window_features
        ).sum(dim=0).permute(2, 0, 1)
    return correlation_volume
