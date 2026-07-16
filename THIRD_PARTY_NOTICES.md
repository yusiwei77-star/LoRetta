# Third-party notices

This initial code migration contains or depends on components derived from:

- Meta DINOv3, loaded from the pinned `facebookresearch/dinov3` commit `adc254450203739c8149213a7a69d8d905b4fcfa` through `torch.hub`.
- RoMaV2 architectural utilities used by LoRetta's affine locator and multi-view Transformer.
- Torchvision's VGG19 architecture used by the dense refiner.

Existing source headers were retained in mechanically migrated files. Before public redistribution, review the upstream repositories' license terms and add their required license texts together with the selected project-level license.
