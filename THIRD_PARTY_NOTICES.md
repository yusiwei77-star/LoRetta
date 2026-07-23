# Third-party notices

LoRetta-authored source code is licensed under the Apache License 2.0. Third-party components and materials remain subject to their respective licenses; the project-level license does not replace those terms.

## DINOv3

- Upstream: <https://github.com/facebookresearch/dinov3>
- Pinned commit: `adc254450203739c8149213a7a69d8d905b4fcfa`
- License: DINOv3 License
- Local license copy: [`third_party/licenses/DINOV3_LICENSE.md`](third_party/licenses/DINOV3_LICENSE.md)

LoRetta vendors the minimal DINOv3 ViT-L/16 inference implementation required by the released model. The vendored source is derived from the pinned upstream commit and is included under `src/loretta/third_party/dinov3`; its provenance, local adaptations, and license are documented in that directory. LoRetta does not download or execute DINOv3 source code at runtime.

The distributed LoRetta checkpoint incorporates DINOv3 parameters and is distributed separately under the DINOv3 License. The vendored source and checkpoint must be used and redistributed in compliance with that license, including its attribution, redistribution, and use restrictions.

## RoMaV2

- Upstream: <https://github.com/Parskatt/RoMaV2>
- License: MIT License
- Local license copy: [`third_party/licenses/ROMAV2_MIT.txt`](third_party/licenses/ROMAV2_MIT.txt)

Portions of the affine locator, dense register, Transformer, geometry, and local-correlation implementation were derived from or adapted from RoMaV2. Copyright (c) 2025 Johan Edstedt.

## Torchvision

- Upstream: <https://github.com/pytorch/vision>
- Referenced release: `v0.21.0`
- License: BSD 3-Clause License
- Local license copy: [`third_party/licenses/TORCHVISION_BSD_3_CLAUSE.txt`](third_party/licenses/TORCHVISION_BSD_3_CLAUSE.txt)

LoRetta uses the VGG19 architecture provided by Torchvision. No separately pretrained Torchvision VGG19 weights are redistributed.

Other Python dependencies retain the licenses supplied by their respective copyright holders. Names and trademarks of third-party projects are used only for identification and do not imply endorsement.
