import argparse
from pathlib import Path

from src.inference import (
    DEFAULT_MATCHABILITY_THRESHOLD,
    DEFAULT_MAX_KEYPOINTS,
    DEFAULT_RANSAC_THRESHOLD,
    run_demo,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_IMAGE = PROJECT_ROOT / "assets" / "reference.jpg"
DEFAULT_SENSED_IMAGE = PROJECT_ROOT / "assets" / "sensed.jpg"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "loretta.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRetta localization-and-registration on one image pair"
    )
    parser.add_argument(
        "--reference-image",
        default=str(DEFAULT_REFERENCE_IMAGE),
        help=f"Reference image path (default: {DEFAULT_REFERENCE_IMAGE})",
    )
    parser.add_argument(
        "--sensed-image",
        default=str(DEFAULT_SENSED_IMAGE),
        help=f"Sensed image path (default: {DEFAULT_SENSED_IMAGE})",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Trusted LoRetta checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--estimate-geometry",
        action="store_true",
        help="Estimate sensed-to-reference homography",
    )
    parser.add_argument(
        "--matchability-threshold",
        type=float,
        default=DEFAULT_MATCHABILITY_THRESHOLD,
    )
    parser.add_argument("--max-keypoints", type=int, default=DEFAULT_MAX_KEYPOINTS)
    parser.add_argument("--ransac-threshold", type=float, default=DEFAULT_RANSAC_THRESHOLD)
    args = parser.parse_args()
    if not 0.0 <= args.matchability_threshold <= 1.0:
        parser.error("--matchability-threshold must be in [0, 1]")
    if args.max_keypoints < 4:
        parser.error("--max-keypoints must be at least 4")
    if args.ransac_threshold <= 0:
        parser.error("--ransac-threshold must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_demo(
        reference_path=args.reference_image,
        sensed_path=args.sensed_image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        requested_device=args.device,
        estimate_geometry=args.estimate_geometry,
        matchability_threshold=args.matchability_threshold,
        max_keypoints=args.max_keypoints,
        ransac_threshold=args.ransac_threshold,
    )
    print(f"Done: device={result['device']}, inference={result['inference_ms']:.1f} ms")
    print("Saved: " + ", ".join(result["outputs"].values()))
    geometry = result["geometry"]
    if geometry["requested"] and geometry["success"]:
        print(
            f"RANSAC: {geometry['num_inliers']}/{geometry['num_candidates']} inliers "
            f"({geometry['inlier_ratio']:.1%})"
        )


if __name__ == "__main__":
    main()
