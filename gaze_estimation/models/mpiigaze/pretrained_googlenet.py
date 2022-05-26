from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config

pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()

        num_ftrs = pretrained_model.fc.in_features

        self.conv1 = nn.Conv2d(
            1,
            3,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=False,
        )
        # self.conv1_batch = nn.BatchNorm2d(3)
        # self.conv1_relu = nn.ReLU(inplace=True)
        self.pretrained_model = pretrained_model
        if config.train.freeze == True:
            print("Freezing network\n")
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # self.pretrained_model.conv1.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        # self.pretrained_model.conv2.conv =  nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.pretrained_model.fc = nn.Linear(num_ftrs, 2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # x = self.conv1_batch(x)
        # x = self.conv1_relu(x)
        return self.pretrained_model(x)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        # x = x.view(x.size(0), -1)
        # x = torch.cat([x, y], dim=1)
        # x = self.last_layer(x)
        return x
