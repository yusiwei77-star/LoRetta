def get_inference_config(*, enable_amp: bool) -> dict:
    """Return the fixed configuration used by the released LoRetta checkpoint."""
    return {
        "dinov3_feature_extractor": {"enable_amp": enable_amp, "layer_indices": (11, 17)},
        "affine_locator": {
            "enable_amp": enable_amp,
        },
        "dense_register": {
            "feature_dims": [64, 128, 256, 512, 512],
            "hidden_dims": [512, 512, 256],
            "kernel_size": 3,
            "hidden_blocks": 8,
            "residual_embedding_dims": [128, 64, 32],
            "local_correlation_radii": [5, 3, 2],
        },
    }
