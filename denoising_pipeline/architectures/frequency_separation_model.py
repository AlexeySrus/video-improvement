import torch
import torch.nn as nn
from denoising_pipeline.utils.tensor_utils import center_pad_tensor_like
from denoising_pipeline.architectures.resnet_block import BasicBlock
import numpy as np


def generate_batt(size=(5, 5), d0=5, n=2):
    kernel = np.fromfunction(
        lambda x, y: \
            1 / (1 + (((x - size[0] // 2) ** 2 + (
                    y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
        (size[0], size[1])
    )
    return kernel


class HightFrequencyImageComponent(nn.Module):
    R = 0.299
    G = 0.587
    B = 0.114

    def __init__(self, shape, four_normalized=True):
        super().__init__()

        self.four_normalized = four_normalized
        self.kernel = 1.0 - generate_batt(shape, 500, 1)
        self.image_shape = shape

        self.kernel[:self.kernel.shape[0] // 2] = \
            np.flip(self.kernel[:self.kernel.shape[0] // 2], 0)
        self.kernel[self.kernel.shape[0] // 2:] = \
            np.flip(self.kernel[self.kernel.shape[0] // 2:], 0)

        self.kernel = torch.FloatTensor(
            [self.kernel, self.kernel]
        ).permute(1, 2, 0)

        self.device = 'cpu'

    def rgb2gray(self, x):
        return x[:, 0] * self.R + x[:, 1] * self.G + x[:, 2] * self.B

    def apply_fft_kernel(self, x):
        return x*self.kernel[:, -x.size(2):]

    def update_device(self, device):
        if self.device != device:
            self.device = device
            self.kernel = self.kernel.to(device)

    def forward(self, x):
        self.update_device(x.device)

        x_gray = self.rgb2gray(x)

        fourier_transform_x = torch.rfft(
            x_gray, 2, normalized=self.four_normalized
        )

        n_fourier_transform_x = self.apply_fft_kernel(
            fourier_transform_x
        )

        x_with_hight_freq = torch.irfft(
            n_fourier_transform_x, 2, normalized=self.four_normalized
        ).unsqueeze(1)

        return x_with_hight_freq[:, :, :self.image_shape[1], :self.image_shape[0]]


class LowFrequencyImageComponent(nn.Module):
    R = 0.299
    G = 0.587
    B = 0.114

    def __init__(self, shape, four_normalized=True):
        super().__init__()

        self.four_normalized = four_normalized
        self.kernel = generate_batt(shape, 500, 1)
        self.image_shape = shape

        self.kernel[:self.kernel.shape[0] // 2] = \
            np.flip(self.kernel[:self.kernel.shape[0] // 2], 0)
        self.kernel[self.kernel.shape[0] // 2:] = \
            np.flip(self.kernel[self.kernel.shape[0] // 2:], 0)

        self.kernel = torch.FloatTensor(
            [self.kernel, self.kernel]
        ).permute(1, 2, 0)

        self.device = 'cpu'

    def rgb2gray(self, x):
        return x[:, 0] * self.R + x[:, 1] * self.G + x[:, 2] * self.B

    def apply_fft_kernel(self, x):
        return x*self.kernel[:, -x.size(2):]

    def update_device(self, device):
        if self.device != device:
            self.device = device
            self.kernel = self.kernel.to(device)

    def forward(self, x):
        self.update_device(x.device)

        x_gray = self.rgb2gray(x)

        fourier_transform_x = torch.rfft(
            x_gray, 2, normalized=self.four_normalized
        )

        n_fourier_transform_x = self.apply_fft_kernel(
            fourier_transform_x
        )

        x_with_low_freq = torch.irfft(
            n_fourier_transform_x, 2, normalized=self.four_normalized
        ).unsqueeze(1)

        return x_with_low_freq[:, :, :self.image_shape[1], :self.image_shape[0]]


class DenoisingNet(nn.Module):
    def __init__(self, shape, n1=1, n2=1):
        super().__init__()

        self.HFILayer = HightFrequencyImageComponent(shape=shape)
        self.LFILayer = LowFrequencyImageComponent(shape=shape)

        self.I_preprocessing = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(15, 20, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(20, 35, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Sequential(*tuple([BasicBlock(35, 35) for _ in range(n1)])),
            nn.Conv2d(35, 15, kernel_size=5, stride=1, padding=0, bias=False)
            # nn.ReLU()
        )

        self.HFI_preprocessing = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(10, 15, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Sequential(*tuple([BasicBlock(15, 15) for _ in range(n1)])),
            nn.Conv2d(15, 5, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.LFI_preprocessing = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(10, 15, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Sequential(*tuple([BasicBlock(15, 15) for _ in range(n1)])),
            nn.Conv2d(15, 5, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.conv1_3d = nn.Conv3d(1, 15, kernel_size=(25, 5, 5), stride=1, padding=0, bias=False)
        self.relu_3d1 = nn.ReLU()

        self.postprocessing = nn.Sequential(
            nn.Conv2d(15, 10, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Sequential(*tuple([BasicBlock(10, 10) for _ in range(n2)])),
            nn.Conv2d(10, 5, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.final_conv2d = nn.Conv2d(5, 3, kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        HFI = self.HFILayer(x)
        LFI = self.LFILayer(x)

        Ipp = self.I_preprocessing(x)
        HFIpp = self.HFI_preprocessing(HFI)
        LFIpp = self.LFI_preprocessing(LFI)

        all_channels = torch.cat(
            (torch.nn.functional.relu(Ipp), HFIpp, LFIpp),
            dim=1
        )

        all_channels = self.relu_3d1(self.conv1_3d(all_channels.unsqueeze(1)))
        all_channels = all_channels.squeeze(2)

        residual_all_channels = torch.nn.functional.relu(
            all_channels + center_pad_tensor_like(Ipp, all_channels)
        )

        final_features = self.postprocessing(residual_all_channels)

        return self.final_conv2d(final_features)

    def inference(self, x):
        return self.forward(x)
