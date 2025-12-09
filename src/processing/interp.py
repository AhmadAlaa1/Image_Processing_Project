import cv2
import numpy as np

from . import basic_ops


def sample_image(img, xs, ys, method: str = "bilinear"):
    """Remap with chosen interpolation; defaults to bilinear."""
    m = (method or "bilinear").lower()
    if m == "nearest":
        return sample_nearest(img, xs, ys)
    if m == "bicubic":
        return sample_bicubic(img, xs, ys)
    return sample_bilinear(img, xs, ys)


def resize_nearest(img, new_width: int, new_height: int):
    """Resize using nearest-neighbor."""
    gray = basic_ops.ensure_grayscale(img)
    w = max(1, int(new_width))
    h = max(1, int(new_height))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)


def resize_bilinear(img, new_width: int, new_height: int):
    """Resize using bilinear interpolation."""
    gray = basic_ops.ensure_grayscale(img)
    w = max(1, int(new_width))
    h = max(1, int(new_height))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_bicubic(img, new_width: int, new_height: int):
    """Resize using bicubic interpolation."""
    gray = basic_ops.ensure_grayscale(img)
    w = max(1, int(new_width))
    h = max(1, int(new_height))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC)


def sample_nearest(img, xs, ys):
    """Nearest-neighbor remap."""
    gray = basic_ops.ensure_grayscale(img)
    return cv2.remap(gray, xs, ys, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def sample_bilinear(img, xs, ys):
    """Bilinear remap."""
    gray = basic_ops.ensure_grayscale(img)
    return cv2.remap(gray, xs, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sample_bicubic(img, xs, ys):
    """Bicubic remap."""
    gray = basic_ops.ensure_grayscale(img)
    return cv2.remap(gray, xs, ys, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
