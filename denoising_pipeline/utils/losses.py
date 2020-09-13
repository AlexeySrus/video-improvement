import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module
import numpy as np


def l2(y_pred, y_true):
    return torch.sqrt(((y_pred - y_true) ** 2).sum())


def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)


class FourierImagesLoss(Module):
    """L2 loss between normed fft2 transforms by L1 normalisation"""

    R = 0.299
    G = 0.587
    B = 0.114

    def __init__(self, base_loss=F.mse_loss,
                 loss_sum_coeffs=(1, 1), four_normalized=True,
                 image_shape=(224, 224)):
        super(FourierImagesLoss, self).__init__()

        self.four_normalized = four_normalized
        self.base_loss = base_loss
        self.coeffs = loss_sum_coeffs
        self.kernel = 1.0 - self.generate_batt(image_shape, 500, 1)
        self.image_shape = image_shape

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

    @staticmethod
    def generate_batt(size=(5, 5), d0=5, n=2):
        kernel = np.fromfunction(
            lambda x, y: \
                1 / (1 + (((x - size[0] // 2) ** 2 + (
                            y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
            (size[0], size[1])
        )
        return kernel

    def update_device(self, device):
        if self.device != device:
            self.device = device
            self.kernel = self.kernel.to(device)

    def forward(self, y_pred, y_true):
        """
        Loss forward
        Args:
            y_pred: batch which contains RGB channels images
            y_true: batch which contains RGB channels images
        Returns:
            L2 loss between normed fft2 transforms by L1 normalisation
        """
        self.update_device(y_pred.device)

        y_pred_gray = self.rgb2gray(y_pred)
        y_true_gray = self.rgb2gray(y_true)

        fourier_transform_pred = torch.rfft(
            y_pred_gray, 2, normalized=self.four_normalized
        )

        fourier_transform_true = torch.rfft(
            y_true_gray, 2, normalized=self.four_normalized
        )

        n_fourier_transform_pred = self.apply_fft_kernel(
            fourier_transform_pred
        )

        n_fourier_transform_true = self.apply_fft_kernel(
            fourier_transform_true
        )

        y_pred_with_hight_freq = torch.irfft(
            n_fourier_transform_pred, 2, normalized=self.four_normalized
        )

        y_true_with_hight_freq = torch.irfft(
            n_fourier_transform_true, 2, normalized=self.four_normalized
        )

        return self.base_loss(
            y_pred, y_true
        ) * self.coeffs[0] + self.base_loss(
            y_pred_with_hight_freq, y_true_with_hight_freq
        ) * self.coeffs[1]