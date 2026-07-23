import cv2
import numpy as np


def scale_pixel_centers(
    points: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = np.asarray(
        (target_width / source_width, target_height / source_height), dtype=np.float32
    )
    return (points + 0.5) * scale - 0.5


def estimate_homography(
    warp_pixels: np.ndarray,
    matchability: np.ndarray,
    *,
    reference_size: tuple[int, int],
    sensed_size: tuple[int, int],
    matchability_threshold: float,
    max_keypoints: int,
    ransac_threshold: float,
) -> dict:
    height, width = matchability.shape
    candidates = np.flatnonzero(matchability.reshape(-1) >= matchability_threshold)
    if candidates.size > max_keypoints:
        candidates = np.random.choice(candidates, max_keypoints, replace=False)
    if candidates.size < 4:
        return {
            "success": False,
            "reason": "fewer_than_four_correspondences",
            "num_candidates": int(candidates.size),
        }

    rows, columns = np.divmod(candidates, width)
    reference_model_points = np.stack((columns, rows), axis=1).astype(np.float32)
    sensed_model_points = warp_pixels.reshape(-1, 2)[candidates].astype(np.float32)
    reference_points = scale_pixel_centers(
        reference_model_points, (width, height), reference_size
    )
    sensed_points = scale_pixel_centers(
        sensed_model_points, (width, height), sensed_size
    )

    cv2.setRNGSeed(0)
    matrix_sensed_to_reference, mask = cv2.findHomography(
        sensed_points,
        reference_points,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=10000,
        confidence=0.999,
    )
    if (
        matrix_sensed_to_reference is None
        or mask is None
        or not np.isfinite(matrix_sensed_to_reference).all()
    ):
        return {
            "success": False,
            "reason": "homography_estimation_failed",
            "num_candidates": int(candidates.size),
        }
    inliers = mask.reshape(-1).astype(bool)
    return {
        "success": True,
        "matrix_sensed_to_reference": matrix_sensed_to_reference,
        "reference_points": reference_points,
        "sensed_points": sensed_points,
        "inliers": inliers,
        "num_candidates": int(candidates.size),
        "num_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
    }
