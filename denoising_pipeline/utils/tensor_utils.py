from enum import Enum
import numpy as np
import torch


def center_pad_tensor_like(x, shape_tensor):
    """
    Pad target area from tensor by new shape
    Args:
        x: input tensor
        shape_tensor: tensor with result shape

    Returns:
        Padded tensor
    """
    pad_values = (
                         torch.LongTensor(list(shape_tensor.size())) -
                         torch.LongTensor(list(x.size()))
                 ) // 2
    pads = torch.stack(
        (pad_values, pad_values)
    ).transpose(0, 1).reshape(-1).flip(0).numpy().tolist()
    return torch.nn.functional.pad(x, pads)


def reshape_tensor(x, new_shape=(224, 224)):
    scales = np.array(new_shape) / np.array(x.shape[-2:])
    return torch.nn.functional.interpolate(
        x, scale_factor=scales, mode='nearest'
    )


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def L1_norm(x):
    x_sum = x.sum()
    return x / x_sum


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(img).permute(2, 0, 1) / 255.0 - 0.5


class TensorRotate(Enum):
    """Rotate enumerates class"""
    NONE = lambda x: x
    ROTATE_90_CLOCKWISE = lambda x: x.transpose(1, 2).flip(2)
    ROTATE_180 = lambda x: x.flip(1, 2)
    ROTATE_90_COUNTERCLOCKWISE = lambda x: x.transpose(1, 2).flip(1)


def rotate_tensor(img: torch.Tensor, rot_value: TensorRotate) -> torch.Tensor:
    """Rotate image tensor

    Args:
        img: tensor in CHW format
        rot_value: element of TensorRotate class, possible values
            TensorRotate.NONE,
            TensorRotate.ROTATE_90_CLOCKWISE,
            TensorRotate.ROTATE_180,
            TensorRotate.ROTATE_90_COUNTERCLOCKWISE,

    Returns:
        Rotated image in same of input format
    """
    return rot_value(img)
