import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn

from loretta.backbone.dinov3 import build_dinov3_feature_extractor
from loretta.module.affine_locator import AffineLocator
from loretta.third_party.dinov3.builder import (
    DINOV3_UPSTREAM_COMMIT,
    dinov3_vitl16,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DummyDinoV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=16, stride=16)
        self.blocks = [None] * 24

    def get_intermediate_layers(self, x: torch.Tensor, *, n: list[int]) -> list[torch.Tensor]:
        del n
        return [self.proj(x).flatten(2).transpose(1, 2)]


class LocalDinoV3Tests(unittest.TestCase):
    def test_builder_uses_the_pinned_vitl16_configuration(self):
        fake_model = Mock()
        with (
            patch(
                "loretta.third_party.dinov3.builder.DinoVisionTransformer",
                return_value=fake_model,
            ) as constructor,
            patch("torch.hub.load", side_effect=AssertionError("network loader used")),
        ):
            returned = dinov3_vitl16()

        self.assertIs(returned, fake_model)
        fake_model.init_weights.assert_called_once_with()
        constructor.assert_called_once_with(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            untie_global_and_local_cls_norm=False,
        )

    def test_runtime_source_does_not_use_torch_hub_load(self):
        offenders = []
        for source_path in (PROJECT_ROOT / "src").rglob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr == "load"
                    and isinstance(function.value, ast.Attribute)
                    and function.value.attr == "hub"
                    and isinstance(function.value.value, ast.Name)
                    and function.value.value.id == "torch"
                ):
                    offenders.append(f"{source_path}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_vendored_source_records_the_exact_upstream_commit(self):
        self.assertEqual(
            DINOV3_UPSTREAM_COMMIT,
            "adc254450203739c8149213a7a69d8d905b4fcfa",  # pragma: allowlist secret
        )
        provenance = (
            PROJECT_ROOT / "src" / "loretta" / "third_party" / "dinov3" / "UPSTREAM.md"
        ).read_text()
        self.assertIn(DINOV3_UPSTREAM_COMMIT, provenance)

    def test_vendored_license_is_packaged_next_to_the_source(self):
        license_path = (
            PROJECT_ROOT / "src" / "loretta" / "third_party" / "dinov3" / "LICENSE.md"
        )
        self.assertTrue(license_path.is_file())
        self.assertIn("DINOv3 License", license_path.read_text())

    def test_feature_extractor_uses_bfloat16_only_when_amp_and_cuda_bf16_are_supported(self):
        cases = (
            (True, True, True, torch.bfloat16),
            (True, True, False, torch.float32),
            (False, True, True, torch.float32),
            (True, False, True, torch.float32),
        )
        for enable_amp, cuda_available, bf16_supported, expected_dtype in cases:
            with (
                patch(
                    "loretta.backbone.dinov3.dinov3_vitl16",
                    side_effect=lambda: _DummyDinoV3(),
                ),
                patch(
                    "loretta.backbone.dinov3.torch.cuda.is_available",
                    return_value=cuda_available,
                ),
                patch(
                    "loretta.backbone.dinov3.torch.cuda.is_bf16_supported",
                    return_value=bf16_supported,
                ) as bf16_check,
            ):
                extractor = build_dinov3_feature_extractor(enable_amp=enable_amp)
            self.assertEqual(extractor.proj.weight.dtype, expected_dtype)
            if enable_amp and cuda_available:
                bf16_check.assert_called_once_with(including_emulation=False)
            else:
                bf16_check.assert_not_called()

    def test_bfloat16_feature_extractor_handles_cpu_fp32_inputs_without_dtype_error(self):
        with (
            patch(
                "loretta.backbone.dinov3.dinov3_vitl16",
                side_effect=lambda: _DummyDinoV3(),
            ),
            patch("loretta.backbone.dinov3.torch.cuda.is_available", return_value=True),
            patch(
                "loretta.backbone.dinov3.torch.cuda.is_bf16_supported",
                return_value=True,
            ),
        ):
            extractor = build_dinov3_feature_extractor(enable_amp=True)

        self.assertEqual(extractor.proj.weight.dtype, torch.bfloat16)
        outputs = extractor(torch.randn(1, 3, 32, 32, dtype=torch.float32))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(tuple(outputs[0].shape), (1, 2, 2, 4))
        self.assertEqual(outputs[0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(outputs[0]).all())

    def test_affine_locator_accepts_bfloat16_descriptors_without_dtype_error(self):
        locator = AffineLocator({"enable_amp": True}).eval()
        descriptors = torch.randn(1, 2048, 2, 2, dtype=torch.bfloat16)
        with torch.inference_mode():
            affine_prior, coarse_matchability = locator(
                descriptors, descriptors.clone(), (32, 32)
            )
        self.assertEqual(tuple(affine_prior.shape), (1, 2, 3))
        self.assertEqual(tuple(coarse_matchability.shape), (1, 1, 2, 2))
        self.assertTrue(torch.isfinite(affine_prior).all())
        self.assertTrue(torch.isfinite(coarse_matchability).all())


if __name__ == "__main__":
    unittest.main()
