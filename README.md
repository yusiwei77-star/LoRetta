# LoRetta

LoRetta 是论文中 Localization-and-Registration 模型的独立纯推理实现。模型先由 `AffineLocator` 估计 matchability-weighted affine prior，再由 `DenseRegister` 在 affine-aligned frame 中通过 `Refiner0–2` 预测 residual warp 和最终 matchability。

## 环境安装

- Python 3.10+
- PyTorch 2.6+
- 推荐使用 CUDA；CPU 可以运行，但速度较慢
- 首次运行需要联网，以便从固定 commit 加载 DINOv3 ViT-L/16 架构；其参数随后由 LoRetta 权重覆盖

```bash
pip install -r requirements.txt
```

## 模型权重

论文发表后，我们会提供纯推理权重 `loretta.pth` 的下载地址。将下载的文件放到：

```text
checkpoints/loretta.pth
```

权重是包含 807 个 tensor entries 的 canonical state dict，不含 Lightning 配置、训练元数据或可执行 Python 对象。读者不需要进行权重迁移；推理代码仅接受官方发布的 canonical 权重。

发布权重的 SHA-256 校验值：

```text
09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e
```

## 运行 Demo

仓库内置 `assets/reference.jpg` 和 `assets/sensed.jpg`。权重放置完成后，直接运行：

```powershell
python demo.py --output-dir outputs
```

使用自己的影像：

```powershell
python demo.py `
  --reference-image C:\path\to\reference.png `
  --sensed-image C:\path\to\sensed.png `
  --checkpoint C:\path\to\loretta.pth `
  --output-dir outputs
```

默认输出：

- `dense_registered.png`：配准至 reference frame 的 sensed image
- `matchability.png`：reference frame 下的 matchability map

增加 `--estimate-geometry` 后，程序从高 matchability 对应中随机、无放回采样，并使用 OpenCV RANSAC 拟合 sensed-to-reference homography。估计成功时额外输出 `ransac_inliers.png` 和 `homography_registered.png`。程序不生成 JSON。

```bash
python demo.py --help
```

## 模型接口

`LoRetta.forward(batch)` 接收：

- `reference_image`：论文中的 \(I_r\)
- `sensed_image`：论文中的 \(I_s\)

并向同一字典增加：

- `warp_1`：论文中的 \(W_{1/1}\)，表示从 reference pixel 指向 sensed image 的 normalized target coordinates
- `matchability_1`：论文中的 \(M_{1/1}\)

主要模块与论文对应如下：

- `dinov3_feature_extractor`：DINOv3 coarse feature extractor
- `affine_locator`：matchability-weighted affine locator
- `vgg_feature_pyramid`：VGG19 feature pyramid
- `dense_register.refiner0`：1/16 residual refiner
- `dense_register.refiner1`：1/8 residual refiner
- `dense_register.refiner2`：1/4 residual refiner

## 验证

无需权重的接口测试：

```bash
python -m unittest discover -s tests -v
```

使用发布权重进行模型集成测试：

```powershell
$env:LORETTA_CHECKPOINT = "checkpoints\loretta.pth"
python -m unittest discover -s tests -p test_model_integration.py -v
```

## 目录结构

```text
LoRetta/
├── assets/              # 默认双图样例
├── src/
│   ├── config/          # 固定推理配置
│   ├── inference/       # 推理流程、图像、几何和可视化
│   └── loretta/         # LoRetta 模型及必要模块
├── tests/               # 公开接口与数值契约测试
└── demo.py              # 双图 Demo 入口
```

输入影像会转为 RGB、固定缩放至 512×512，并使用训练时的 mean/std 归一化。第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。当前尚未选择项目级许可证。
