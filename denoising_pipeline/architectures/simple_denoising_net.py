import torch
from torch import nn
from denoising_pipeline.utils.tensor_utils import center_pad_tensor_like


class SimpleDenoisingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 5, 3, bias=False)
        self.conv2 = nn.Conv2d(5, 15, 3, bias=False)
        self.conv3 = nn.Conv2d(15, 5, 3, bias=False)
        self.conv4 = nn.Conv2d(5, 15, 3, bias=False)
        self.conv5 = nn.Conv2d(15, 10, 3, bias=False)
        self.conv6 = nn.Conv2d(10, 5, 3, bias=False)
        self.conv7 = nn.Conv2d(5, 3, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x1 = nn.functional.relu(x, inplace=True)
        x = self.conv2(x1)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv3(x)
        x += center_pad_tensor_like(x1, x)
        x3 = nn.functional.relu(x, inplace=True)
        x = self.conv4(x3)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv5(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv6(x)
        x += center_pad_tensor_like(x3, x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv7(x)
        return x

    def inference(self, x):
        return self(x)
