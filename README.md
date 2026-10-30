# LoRetta: A Foundation Model and Extensive Dataset for Global-Scale Remote Sensing Dense Image Matching

<p align="center"><a href="https://arxiv.org/abs/2608.04106">Paper</a> · <a href="https://www.siweiyu.com/work/loretta/">Project Page</a> · <a href="https://huggingface.co/BRoss123/LoRetta">Model</a></p>

**LoRetta** is a foundation model for dense image matching in remote sensing imagery. It introduces a _localization-and-registration_ paradigm: first localize the matchable overlap with an affine prior, then refine dense correspondences within the aligned frame. On the LEVIR-GM benchmark, LoRetta achieves **83.3% AUC** — outperforming RoMa v2 by 1.6 points with **47.8% lower latency**.

<p align="center">
  <img src="assets/teaser.png" alt="LoRetta architecture" width="95%"/>
</p>

Inference code and pretrained weights are available, and the [paper](https://arxiv.org/abs/2608.04106) is on arXiv. The dataset, training code, and evaluation code are coming soon.

## Installation

```bash
git clone https://github.com/yusiwei77-star/LoRetta.git
cd LoRetta
pip install -e .
```

Requires **Python >= 3.10** and **PyTorch >= 2.6**. The pretrained weights are downloaded automatically from Hugging Face on first use. For offline use, pass a local checkpoint with `--checkpoint /path/to/loretta.pth`.

## Quick Start

```bash
python demo.py --reference-image assets/reference.jpg --sensed-image assets/sensed.jpg
```

Input assumptions:

- Both images are independently resized to 512 × 512 without preserving their aspect ratios. Inputs with very large aspect ratios may therefore produce degraded results.
- The sensed image is assumed to have a ground sampling distance (GSD, m/pixel) less than or equal to that of the reference image, and its ground footprint should not exceed that of the reference image. Performance is not guaranteed when this assumption is violated.

This warps `sensed.jpg` into the coordinate frame of `reference.jpg` and saves the results to `outputs/`:

- `dense_registered.png` — original-resolution sensed image warped by the dense correspondence field
- `matchability.png` — per-pixel confidence heatmap

With `--estimate-geometry`, two additional outputs are produced:

- `ransac_inliers.png` — matched keypoints with RANSAC inliers
- `homography_registered.png` — sensed image warped by the estimated homography

## Citation

```bibtex
@article{yu2026loretta,
  title     = {LoRetta: A Foundation Model and Extensive Dataset for
               Global-Scale Remote Sensing Dense Image Matching},
  author    = {Yu, Siwei and Guo, Han and Shi, Zhenwei and Zou, Zhengxia},
  journal   = {arXiv preprint arXiv:2608.04106},
  year      = {2026},
}
```

## License

- **LoRetta-authored source code:** [Apache License 2.0](LICENSE)
- **Vendored DINOv3 source and the released LoRetta checkpoint:** [DINOv3 License](src/loretta/third_party/dinov3/LICENSE.md)

LoRetta also incorporates components from [RoMaV2](https://github.com/Parskatt/RoMaV2) (MIT) and [Torchvision](https://github.com/pytorch/vision) (BSD 3-Clause). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance and complete third-party licensing details.
