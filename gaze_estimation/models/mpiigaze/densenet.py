from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config

pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_ftrs = pretrained_model.classifier.in_features

        self.conv1 = nn.Conv2d(
            1,
            3,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv1_batch = nn.BatchNorm2d(3)
        self.conv1_relu = nn.ReLU(inplace=True)

        self.pretrained_model = pretrained_model
        self.pretrained_model.classifier = nn.Linear(1024, 2)
        self.pretrained_model.classifier = nn.Linear(num_ftrs, 2)

        print(self)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv1_batch(x)
        x = self.conv1_relu(x)
        return self.pretrained_model(x)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        return x
