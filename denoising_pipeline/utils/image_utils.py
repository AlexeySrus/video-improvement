import cv2
from enum import Enum
import numpy as np


class Rotate(Enum):
    """Rotate enumerates class"""
    NONE = lambda x: x
    ROTATE_90_CLOCKWISE = lambda x: np.flip(x.transpose(1, 0, 2), axis=1)
    ROTATE_180 = lambda x: np.flip(np.flip(x, axis=0), axis=1)
    ROTATE_90_COUNTERCLOCKWISE = lambda x: np.flip(x.transpose(1, 0, 2), axis=0)


def rotate_crop(img: np.ndarray, rot_value: Rotate) -> np.ndarray:
    """Rotate numpy image

    Args:
        img: image in OpenCV format
        rot_value: element of Rotate class, possible values
            Rotate.NONE,
            Rotate.ROTATE_90_CLOCKWISE,
            Rotate.ROTATE_180,
            Rotate.ROTATE_90_COUNTERCLOCKWISE,

    Returns:
        Rotated image in same of input format
    """
    return rot_value(img)


def random_crop(
        image1: np.ndarray,
        image2: np.ndarray,
        window_size: int = 224) -> tuple:
    """

    Args:
        image1:
        image2:
        window_size:

    Returns:

    """
    select_image1, select_image2 = np.random.choice(
        [image1, image2],
        [image2, image1]
    )

    x = np.random.randint(0, select_image1.shape[1] - window_size)
    y = np.random.randint(0, select_image1.shape[0] - window_size)

    return select_image1[
           y:y + window_size,
           x:x + window_size
        ], select_image2[
           y:y + window_size,
           x:x + window_size
        ]


def apply_transforms(crop1: np.ndarray, crop2: np.ndarray) -> tuple:
    """

    Args:
        crop1:
        crop2:

    Returns:

    """
    select_rotation = np.random.choice(
        [
            Rotate.NONE,
            Rotate.ROTATE_90_CLOCKWISE,
            Rotate.ROTATE_180,
            Rotate.ROTATE_90_COUNTERCLOCKWISE
        ]
    )

    crop1 = rotate_crop(crop1, select_rotation)
    crop2 = rotate_crop(crop2, select_rotation)

    return np.random.choice(
        [crop1, crop2],
        [np.flip(crop1, 1), np.flip(crop2, 1)]
    )


def random_crop_with_transforms(
        image1: np.ndarray,
        image2:np.ndarray,
        window_size: int = 224) -> tuple:
    """

    Args:
        image1:
        image2:
        window_size:

    Returns:

    """
    return apply_transforms(
        *random_crop(image1.copy(), image2.copy(), window_size)
    )


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

    im = 255 - im

    k = np.ones((15, 15))
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_DILATE, k), 10)
    k = np.ones((55, 55))
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_ERODE, k), 10)
    k = np.ones((55 - 15, 55 - 15))
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_DILATE, k), 10)
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)


def load_image(path):
    """
    Load image in HWC RGB uint8 format
    Args:
        path: path to image

    Returns:
        Image in HWC RGB uint8 format
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('Can\'t open image: {}'.format(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
