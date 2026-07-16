from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from ..config import get_inference_config
from ..loretta import LoRetta
from .checkpoint import load_checkpoint
from .geometry import estimate_homography
from .image import (
    MODEL_SIZE,
    image_to_tensor,
    load_rgb_image,
    normalized_warp_to_pixels,
    resolve_device,
)
from .visualization import (
    clear_geometry_outputs,
    save_dense_outputs,
    save_geometry_outputs,
)

DEFAULT_MATCHABILITY_THRESHOLD = 0.5
DEFAULT_MAX_KEYPOINTS = 1000
DEFAULT_RANSAC_THRESHOLD = 16.0


class GeometrySummary(TypedDict, total=False):
    requested: bool
    success: bool
    reason: str
    matrix_sensed_to_reference: list[list[float]]
    num_candidates: int
    num_inliers: int
    inlier_ratio: float


class DemoResult(TypedDict):
    device: str
    inference_ms: float
    outputs: dict[str, str]
    geometry: GeometrySummary


def run_demo(
    *,
    reference_path: str | Path,
    sensed_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    requested_device: str = "auto",
    estimate_geometry: bool = False,
    matchability_threshold: float = DEFAULT_MATCHABILITY_THRESHOLD,
    max_keypoints: int = DEFAULT_MAX_KEYPOINTS,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
) -> DemoResult:
    if not 0.0 <= matchability_threshold <= 1.0:
        raise ValueError("matchability_threshold must be in [0, 1]")
    if max_keypoints < 4:
        raise ValueError("max_keypoints must be at least 4")
    if ransac_threshold <= 0:
        raise ValueError("ransac_threshold must be positive")
    device = resolve_device(requested_device)
    if device.type == "cpu":
        print("Warning: CPU inference is supported but can be very slow.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reference_original = load_rgb_image(reference_path)
    sensed_original = load_rgb_image(sensed_path)
    reference_tensor, _ = image_to_tensor(reference_original, device)
    sensed_tensor, sensed_model = image_to_tensor(sensed_original, device)

    model = LoRetta(get_inference_config(enable_amp=device.type == "cuda"))
    load_checkpoint(model, checkpoint_path)
    model = model.eval().to(device)
    batch = {
        "reference_image": reference_tensor,
        "sensed_image": sensed_tensor,
    }

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            model(batch)
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA ran out of memory; close other GPU tasks or retry with --device cpu"
        ) from error
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    warp = batch["warp_1"][0].float().cpu().permute(1, 2, 0).numpy()
    matchability = batch["matchability_1"][0, 0].float().cpu().numpy()
    if not np.isfinite(warp).all() or not np.isfinite(matchability).all():
        raise RuntimeError("Model produced non-finite warp or matchability values")
    warp_pixels = normalized_warp_to_pixels(warp, MODEL_SIZE)
    outputs = save_dense_outputs(
        sensed_model=sensed_model,
        warp_pixels=warp_pixels,
        matchability=matchability,
        reference_size=reference_original.size,
        output_dir=output,
    )
    clear_geometry_outputs(output)

    result: DemoResult = {
        "device": str(device),
        "inference_ms": elapsed_ms,
        "outputs": outputs,
        "geometry": {"requested": estimate_geometry, "success": False},
    }

    if estimate_geometry:
        geometry = estimate_homography(
            warp_pixels,
            matchability,
            reference_size=reference_original.size,
            sensed_size=sensed_original.size,
            matchability_threshold=matchability_threshold,
            max_keypoints=max_keypoints,
            ransac_threshold=ransac_threshold,
        )
        if geometry["success"]:
            result["outputs"].update(
                save_geometry_outputs(
                    reference_image=reference_original,
                    sensed_image=sensed_original,
                    geometry=geometry,
                    output_dir=output,
                )
            )
            result["geometry"] = {
                "requested": True,
                "success": True,
                "matrix_sensed_to_reference": geometry[
                    "matrix_sensed_to_reference"
                ].tolist(),
                "num_candidates": geometry["num_candidates"],
                "num_inliers": geometry["num_inliers"],
                "inlier_ratio": geometry["inlier_ratio"],
            }
        else:
            result["geometry"] = {"requested": True, **geometry}
            print(f"Warning: optional geometry estimation failed: {geometry['reason']}")
    return result
