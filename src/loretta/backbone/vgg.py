"""VGG19 feature pyramid used by LoRetta's dense register."""

import torch
from torch import nn
from torchvision import models


class VGG19FeaturePyramid(nn.Module):
    """VGG19 structure initialized without an independent weight download."""

    def __init__(self) -> None:
        super().__init__()
        features = models.vgg19(weights=None).features
        self.block1 = features[:4]
        self.block2 = features[4:9]
        self.block3 = features[9:18]
        self.block4 = features[18:27]
        self.block5 = features[27:36]

    def forward(self, image_pair: torch.Tensor) -> list[torch.Tensor]:
        level1 = self.block1(image_pair)
        level2 = self.block2(level1)
        level3 = self.block3(level2)
        level4 = self.block4(level3)
        level5 = self.block5(level4)
        return [level5, level4, level3, level2, level1]
