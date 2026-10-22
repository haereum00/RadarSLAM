#!/usr/bin/env python3

import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(ResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet18.fc = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.resnet18(x)
        return out


