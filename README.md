# LoRetta

LoRetta 是论文中 Localization-and-Registration 模型的独立纯推理实现。代码命名与论文保持一致：模型先由 `AffineLocator` 估计 matchability-weighted affine prior，再由 `DenseRegister` 在 affine-aligned frame 中通过 `Refiner0–2` 预测 residual warp 和最终 matchability。

## 环境

- Python 3.10+
- PyTorch 2.6+
- 推荐 CUDA；CPU 可以运行，但速度较慢
- 首次运行需要联网。DINOv3 ViT-L/16 架构通过固定 commit 的 `torch.hub` 加载，参数随后由 LoRetta checkpoint 覆盖

```bash
pip install -r requirements.txt
```

也可以将本项目以 editable package 安装：

```bash
pip install -e .
```

## 运行 Demo

仓库默认使用以下文件：

- `assets/reference.jpg`
- `assets/sensed.jpg`
- `checkpoints/loretta.pth`

因此完成环境安装后可以直接运行：

```powershell
python demo.py --output-dir outputs
```

使用自己的影像和权重：

```powershell
python demo.py `
  --reference-image C:\path\to\reference.png `
  --sensed-image C:\path\to\sensed.png `
  --checkpoint C:\path\to\loretta.pth `
  --output-dir outputs
```

默认输出：

- `dense_registered.png`：sensed image 配准到 reference frame
- `matchability.png`：reference frame 下的预测 matchability map

增加 `--estimate-geometry` 后，程序从高 matchability 对应中随机、无放回地采样，并用 OpenCV RANSAC 拟合 sensed-to-reference homography。成功时额外输出 `ransac_inliers.png` 和 `homography_registered.png`。程序不生成 JSON。

完整参数见：

```bash
python demo.py --help
```

## 论文式模型接口

`LoRetta.forward(batch)` 接收：

- `reference_image`：论文中的 (I_r)
- `sensed_image`：论文中的 (I_s)

并向同一个字典增加：

- `warp_1`：论文中的 (W_{1/1})，存储 normalized reference-to-sensed target coordinates
- `matchability_1`：论文中的 (M_{1/1})

模型模块和论文对应如下：

- `dinov3_feature_extractor`：frozen DINOv3 coarse feature extractor
- `affine_locator`：matchability-weighted affine locator
- `vgg_feature_pyramid`：ΦVGG feature pyramid
- `dense_register.refiner0`：1/16 residual refiner
- `dense_register.refiner1`：1/8 residual refiner
- `dense_register.refiner2`：1/4 residual refiner

## 权重

`checkpoints/loretta.pth` 是包含 807 个 canonical state-dict entries 的纯推理权重，不包含 Lightning 训练配置或其他元数据。权重文件被 `.gitignore` 排除。

旧 Lightning checkpoint 不能由推理 loader 直接加载，需要显式转换：

```powershell
python a trusted offline conversion tool `
  /path/to/trusted-legacy.ckpt `
  checkpoints\loretta.pth
```

迁移工具需要额外依赖：

```bash
pip install -r trusted migration dependencies
```

推理 loader 仅允许安全的 tensor-only 反序列化，不会自动执行旧 checkpoint 中的 Python 元数据。旧 Lightning checkpoint 只应在确认可信后交给迁移脚本处理。

## 测试

快速测试：

```bash
python -m unittest discover -s tests -v
```

真实权重集成测试：

```powershell
$env:LORETTA_CHECKPOINT = "/path/to/loretta.pth"
python -m unittest discover -s tests -p test_model_integration.py -v
```

## 目录结构

```text
LoRetta/
├── assets/              # 默认 reference/sensed 样例
├── checkpoints/         # 本地推理权重，不纳入版本控制
├── scripts/             # checkpoint schema 迁移工具
├── src/
│   ├── config/          # 固定推理配置
│   ├── inference/       # CLI pipeline、图像、几何、可视化、checkpoint
│   ├── loretta/         # LoRetta、AffineLocator、DenseRegister、backbones
│   ├── datasets/        # 预留：数据模块
│   ├── evaluation/      # 预留：评估流程
│   ├── lightning/       # 预留：Lightning 模块
│   ├── losses/          # 预留：训练损失
│   ├── metrics/         # 预留：评估指标
│   └── training/        # 预留：训练流程
├── tests/
└── demo.py
```

输入影像会转为 RGB、固定缩放至 512×512，并使用训练时的 mean/std 归一化。第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。当前尚未选择项目级许可证。
