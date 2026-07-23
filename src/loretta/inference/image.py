from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

MODEL_SIZE = (512, 512)
MEAN = np.asarray((0.430, 0.411, 0.296), dtype=np.float32)
STD = np.asarray((0.213, 0.156, 0.143), dtype=np.float32)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    return torch.device(requested)


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not decode image: {image_path}") from error


def image_to_tensor(image: Image.Image, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    resized = image.resize(MODEL_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.uint8)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, rgb


def normalized_warp_to_pixels(
    warp: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    warp_pixels = np.empty_like(warp, dtype=np.float32)
    warp_pixels[..., 0] = (warp[..., 0] + 1.0) * width / 2.0 - 0.5
    warp_pixels[..., 1] = (warp[..., 1] + 1.0) * height / 2.0 - 0.5
    return warp_pixels
