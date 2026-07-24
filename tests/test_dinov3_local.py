import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from loretta.third_party.dinov3.builder import (
    DINOV3_UPSTREAM_COMMIT,
    dinov3_vitl16,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
