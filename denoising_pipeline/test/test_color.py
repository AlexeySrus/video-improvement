import numpy as np
import cv2


def eval_avg_color(img):
    region = img[:100, :100].copy()
    return (
        region[:, :, 0].mean(),
        region[:, :, 1].mean(),
        region[:, :, 2].mean()
    )


def correct_by_base_color(img, dst_color, src_color):
    return (img * dst_color / src_color).clip(0, 255).astype('uint8')


def correct_by_avg(img):
    colors_avg = np.array([img[:, :, i].mean() for i in range(3)])
    avg = colors_avg.mean()
    return (img * avg / colors_avg).clip(0, 255).astype('uint8')


img1 = cv2.imread('../../data/images/out.png', 1)
imgb = cv2.imread('../../data/images/night_series/7.png', 1)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)


cv2.imshow('Image', correct_by_base_color(img1, eval_avg_color(imgb), eval_avg_color(img1)))

cv2.waitKey(0)
cv2.destroyAllWindows()
