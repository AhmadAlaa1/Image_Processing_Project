import cv2
import numpy as np

from . import basic_ops


def sample_image(img, xs, ys, method: str = "bilinear"):
    """Remap with OpenCV interpolation."""
    gray = basic_ops.ensure_grayscale(img)
    interp = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}.get(method.lower(), cv2.INTER_LINEAR)
    return cv2.remap(gray.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32), interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def resize(img, new_width: int, new_height: int, method: str = "nearest"):
    """Resize via cv2.resize with chosen interpolation."""
    gray = basic_ops.ensure_grayscale(img)
    interp = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}.get(method.lower(), cv2.INTER_NEAREST)
    w = max(1, int(new_width))
    h = max(1, int(new_height))
    return cv2.resize(gray.astype(np.float32), (w, h), interpolation=interp)
