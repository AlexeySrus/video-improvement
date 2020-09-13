import torch.nn as nn
from denoising_pipeline.utils.tensor_utils import center_pad_tensor_like


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=0, bias=False)


class BasicBlock(nn.Module):
    """Basic block without padding from ResNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv5x5(inplanes, planes, stride)

        # self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.InstanceNorm2d(planes)
        # self.bn1 = nn.LocalResponseNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)

        # self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.InstanceNorm2d(planes)
        # self.bn2 = nn.LocalResponseNorm(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += center_pad_tensor_like(residual, out)
        out = self.relu(out)

        return out
