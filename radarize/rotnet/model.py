#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientChannelAttention(nn.Module):
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = EfficientChannelAttention(
            planes
        )

        self.shortcut = nn.Sequential()
        if (
            stride != 1 or in_planes != self.expansion * planes
        ):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        ECA_out = self.channel(out)
        out = out * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ECAResNet18(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(ECAResNet18, self).__init__()
        self.in_planes = 64
        num_blocks = [2, 2, 2, 2]
        block = BasicBlock

        self.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, 32)
        self.fc = FcBlock(32, n_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=256, dropout=0.05):
        super(FcBlock, self).__init__()
        self.mid_dim = mid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            nn.ReLU(True),
            nn.Linear(self.mid_dim, self.out_dim),
        )

    def forward(self, x):
        x = self.fcs(x)
        return x


