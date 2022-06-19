# from dataloader import RetinopathyLoader
from torch import Tensor, device, cuda, no_grad, load, save
# from torch import max as tensor_max
from torch.utils.data import TensorDataset, DataLoader
# from torchvision import transforms
# from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Optional, Type, Union, List, Dict
# from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import sys
# import os
import torch.nn as nn
# import torch.optim as op
import torchvision.models as torch_models
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle

class BasicBlock(nn.Module):
    """
    output = (channels, H, W) -> conv2d (3x3) -> (channels, H, W) -> conv2d (3x3) -> (channels, H, W) + (channels, H, W)
    """
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.down_sample = down_sample

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        residual = inputs
        outputs = self.block(inputs)
        if self.down_sample is not None:
            residual = self.down_sample(inputs)

        outputs = self.activation(outputs + residual)

        return outputs

class BottleneckBlock(nn.Module):
    """
    output = (channels * 4, H, W) -> conv2d (1x1) -> (channels, H, W) -> conv2d (3x3) -> (channels, H, W)
             -> conv2d (1x1) -> (channels * 4, H, W) + (channels * 4, H, W)
    """
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None):
        super(BottleneckBlock, self).__init__()

        external_channels = out_channels * self.expansion
        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(in_channels=out_channels,
                      out_channels=external_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(external_channels),
        )
        self.down_sample = down_sample

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        residual = inputs
        outputs = self.block(inputs)
        if self.down_sample is not None:
            residual = self.down_sample(inputs)

        outputs = self.activation(outputs + residual)

        return outputs

class ResNet(nn.Module):
    def __init__(self, architecture: str, block: Type[Union[BasicBlock, BottleneckBlock]], layers: List[int],
                 pretrain: bool):
        super(ResNet, self).__init__()

        if pretrain:
            pretrained_resnet = getattr(torch_models, architecture)(pretrained=True)
            self.conv_1 = nn.Sequential(
                getattr(pretrained_resnet, 'conv1'),
                getattr(pretrained_resnet, 'bn1'),
                getattr(pretrained_resnet, 'relu'),
                getattr(pretrained_resnet, 'maxpool')
            )

            # Layers
            self.conv_2 = getattr(pretrained_resnet, 'layer1')
            self.conv_3 = getattr(pretrained_resnet, 'layer2')
            self.conv_4 = getattr(pretrained_resnet, 'layer3')
            self.conv_5 = getattr(pretrained_resnet, 'layer4')

            self.classify = nn.Sequential(
                getattr(pretrained_resnet, 'avgpool'),
                nn.Flatten(),
                nn.Linear(getattr(pretrained_resnet, 'fc').in_features, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=50, out_features=5)
            )

            del pretrained_resnet
        else:
            self.current_channels = 64

            self.conv_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,
                             stride=2,
                             padding=1)
            )

            # Layers
            self.conv_2 = self.make_layer(block=block,
                                          num_of_blocks=layers[0],
                                          in_channels=64)
            self.conv_3 = self.make_layer(block=block,
                                          num_of_blocks=layers[1],
                                          in_channels=128,
                                          stride=2)
            self.conv_4 = self.make_layer(block=block,
                                          num_of_blocks=layers[2],
                                          in_channels=256,
                                          stride=2)
            self.conv_5 = self.make_layer(block=block,
                                          num_of_blocks=layers[3],
                                          in_channels=512,
                                          stride=2)

            self.classify = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=512 * block.expansion, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=50, out_features=5)
            )

    def make_layer(self, block: Type[Union[BasicBlock, BottleneckBlock]], num_of_blocks: int, in_channels: int,
                   stride: int = 1) -> nn.Sequential:
        """
        Make a layer with given block
        :param block: block to be used to compose the layer
        :param num_of_blocks: number of blocks in this layer
        :param in_channels: channels used in the blocks
        :param stride: stride
        :return: convolution layer composed with given block
        """
        down_sample = None
        if stride != 1 or self.current_channels != in_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.current_channels,
                          out_channels=in_channels * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(in_channels * block.expansion),
            )

        layers = [
            block(in_channels=self.current_channels,
                  out_channels=in_channels,
                  stride=stride,
                  down_sample=down_sample)
        ]
        self.current_channels = in_channels * block.expansion
        layers += [block(in_channels=self.current_channels, out_channels=in_channels) for _ in range(1, num_of_blocks)]

        return nn.Sequential(*layers)

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        partial_results = inputs
        for idx in range(1, 6):
            partial_results = getattr(self, f'conv_{idx}')(partial_results)
        return self.classify(partial_results)