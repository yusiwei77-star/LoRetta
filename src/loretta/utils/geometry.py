import torch


def make_normalized_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    rows, columns = torch.meshgrid(
        torch.linspace(-1 + 1 / height, 1 - 1 / height, height, device=device),
        torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=device),
        indexing="ij",
    )
    return torch.stack((columns, rows), dim=-1).unsqueeze(0).expand(
        batch_size, height, width, 2
    )
