from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GEOMETRY_OUTPUT_NAMES = ("ransac_inliers.png", "homography_registered.png")


def clear_geometry_outputs(output_dir: Path) -> None:
    """Prevent optional geometry images from leaking across fixed-name demo runs."""
    for name in GEOMETRY_OUTPUT_NAMES:
        (output_dir / name).unlink(missing_ok=True)


def save_dense_outputs(
    *,
    sensed_model: np.ndarray,
    warp_pixels: np.ndarray,
    matchability: np.ndarray,
    reference_size: tuple[int, int],
    output_dir: Path,
) -> dict[str, str]:
    dense_registered = cv2.remap(
        sensed_model,
        warp_pixels[..., 0],
        warp_pixels[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    dense_registered = cv2.resize(
        dense_registered, reference_size, interpolation=cv2.INTER_LINEAR
    )
    dense_path = output_dir / "dense_registered.png"
    matchability_path = output_dir / "matchability.png"
    Image.fromarray(dense_registered).save(dense_path)
    matchability_u8 = np.clip(matchability * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(matchability_u8, cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB,
    )
    heatmap = cv2.resize(heatmap, reference_size, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(heatmap).save(matchability_path)
    return {
        "dense_registered": dense_path.name,
        "matchability": matchability_path.name,
    }


def draw_ransac_matches(
    reference_image: Image.Image,
    sensed_image: Image.Image,
    geometry: dict,
) -> np.ndarray:
    reference = cv2.cvtColor(np.asarray(reference_image), cv2.COLOR_RGB2BGR)
    sensed = cv2.cvtColor(np.asarray(sensed_image), cv2.COLOR_RGB2BGR)
    canvas_height = max(reference.shape[0], sensed.shape[0])
    canvas = np.zeros(
        (canvas_height, reference.shape[1] + sensed.shape[1], 3), dtype=np.uint8
    )
    canvas[: reference.shape[0], : reference.shape[1]] = reference
    canvas[: sensed.shape[0], reference.shape[1] :] = sensed
    offset = np.asarray((reference.shape[1], 0), dtype=np.float32)
    for reference_point, sensed_point, is_inlier in zip(
        geometry["reference_points"],
        geometry["sensed_points"],
        geometry["inliers"],
    ):
        color = (0, 220, 0) if is_inlier else (0, 0, 220)
        first = tuple(np.rint(reference_point).astype(int))
        second = tuple(np.rint(sensed_point + offset).astype(int))
        cv2.circle(canvas, first, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, second, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, first, second, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_geometry_outputs(
    *,
    reference_image: Image.Image,
    sensed_image: Image.Image,
    geometry: dict,
    output_dir: Path,
) -> dict[str, str]:
    inliers_path = output_dir / "ransac_inliers.png"
    homography_path = output_dir / "homography_registered.png"
    Image.fromarray(draw_ransac_matches(reference_image, sensed_image, geometry)).save(
        inliers_path
    )
    registered = cv2.warpPerspective(
        cv2.cvtColor(np.asarray(sensed_image), cv2.COLOR_RGB2BGR),
        geometry["matrix_sensed_to_reference"],
        reference_image.size,
    )
    Image.fromarray(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB)).save(homography_path)
    return {
        "ransac_inliers": inliers_path.name,
        "homography_registered": homography_path.name,
    }
