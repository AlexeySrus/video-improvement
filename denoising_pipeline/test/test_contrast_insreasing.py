import numpy as np
import cv2
from denoising_pipeline.utils.image_matcher import ImageMatcher


def increase_sharpen(img):
    img_blured = cv2.GaussianBlur(img, (5, 5), 0)
    img_m = cv2.addWeighted(img, 1.5, img_blured, -0.5, 0)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_s = cv2.filter2D(img_m, -1, kernel, borderType=cv2.CV_8U)
    return img_s


def image_preprocessing(img):
    im = cv2.cvtColor(increase_sharpen(img), cv2.COLOR_RGB2GRAY)
    th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    im = upper_bin(im, th)

    k = ring_by_np(5)
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_DILATE, k), 10)

    # k = ring_by_np(3)
    # im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_CLOSE, k), 10)

    im = 255 - im

    im = cv2.medianBlur(im, 7)

    k = ring_by_np(50)
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_CLOSE, k), 10)

    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)


def mse(a1, a2):
    return np.abs(a1 - a2).mean()


def upper_bin(img, threshold):
    res = img.copy()
    res[img > threshold] = 255
    res[img <= threshold] = 0
    return res


def ring_by_np(size):
    res = np.zeros(shape=(size, size), dtype=np.uint8)
    m = size // 2
    for i in range(size):
        for j in range(size):
            if (i - m) ** 2 + (j - m) ** 2 <= m ** 2:
                res[i][j] = 255
    return res


img1 = cv2.imread('../../data/images/night_series/8.png', 1)
img2 = cv2.imread('../../data/images/night_series/9.png', 1)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
#
matcher = ImageMatcher(preprocessing=image_preprocessing)

img2_to_1 = matcher([img1, img2])
print('MSE match by originals: ', mse(img1, img2_to_1))


# cv2.imshow('Image 1', img1)
cv2.imshow('Image 1', image_preprocessing(img1))
cv2.imwrite('../../data/images/0test_img1_transform.png', image_preprocessing(img1))

cv2.imwrite('../../data/images/test_img1.png', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
cv2.imwrite('../../data/images/test_img2.png', cv2.cvtColor(img2_to_1, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()