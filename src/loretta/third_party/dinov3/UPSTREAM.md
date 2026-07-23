# Vendored DINOv3 inference source

This directory contains the minimal DINOv3 ViT inference implementation required by LoRetta.

- Upstream: <https://github.com/facebookresearch/dinov3>
- Commit: `adc254450203739c8149213a7a69d8d905b4fcfa`
- Architecture: `dinov3_vitl16`
- Local changes: package-relative imports, a minimal public export surface, and a local builder fixed to the exact architecture configuration used by LoRetta.

The source files retain their upstream copyright headers and are distributed under the included [DINOv3 License](LICENSE.md), not the repository's Apache 2.0 license.
