import torch
import numpy as np
import cv2
import torch.nn.functional as F
from denoising_pipeline.utils.losses import l2, FourierLoss

img = cv2.imread('../../data/images/video_series/1.png', 1)
img2 = cv2.imread('../../data/images/video_series/2.png', 1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray = cv2.imread('../../data/images/video_series/1.png', 0)
gray2 = cv2.imread('../../data/images/video_series/2.png', 0)

assert gray is not None

tgray = torch.FloatTensor(np.array([gray, gray2])).to('cpu')

print(tgray.shape)
print(tgray.dtype)


def drop_fft_center(x, k=1/225):
    x[:, :int(x.size(1) * k), :int(x.size(2) * k)] = 0
    x[:, -int(x.size(1) * k):, :int(x.size(2) * k)] = 0
    # x[:, int(x.size(1) * k):-int(x.size(1) * k),
    #    int(x.size(2) * k):-int(x.size(2) * k)] = 0
    return x


tft = torch.rfft(tgray, 1, normalized=True)
print(tft.shape)


def complex_abs(x):
    return torch.sqrt(x[:, :, :, 0] ** 2 + x[:, :, :, 1] ** 2)


ctft = drop_fft_center(complex_abs(tft))


print(ctft.shape)

print(l2(ctft[0], ctft[1]))
print(F.mse_loss(ctft[0], ctft[1]))

tgray1 = torch.FloatTensor(np.array(gray)).to('cpu')
tgray2 = torch.FloatTensor(np.array(gray2)).to('cpu')
tft1 = torch.rfft(tgray1, 2, normalized=True).unsqueeze(0)
tft2 = torch.rfft(tgray2, 2, normalized=True).unsqueeze(0)
ctft1 = drop_fft_center(complex_abs(tft1))
ctft2 = drop_fft_center(complex_abs(tft2))


print(l2(ctft1, ctft2))
print(F.mse_loss(ctft1, ctft2))

# timg = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to('cuda') / 255.0 - 0.5
# timg2 = torch.FloatTensor(img2).permute(2, 0, 1).unsqueeze(0).to('cuda') / 255.0 - 0.5
#
# four_loss = FourierLoss(loss_sum_coeffs=(1, 0), four_normalized=False)
# print(four_loss(timg, timg2))

img1_without_center_fft = drop_fft_center(tft1)

inv_img1 = torch.irfft(img1_without_center_fft, 2, True).squeeze(0)
inv_img1 = inv_img1.to('cpu').numpy().astype('uint8')
print(inv_img1.shape)


cv2.imshow('Original image', gray)
cv2.imshow('Inversed image', inv_img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
