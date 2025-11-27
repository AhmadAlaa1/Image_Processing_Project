import cv2
import numpy as np


MAX_OUTPUT_PIXELS = 50_000_000

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def apply_affine(img, matrix, output_shape=None, method: str = "bilinear"):
    """Affine warp via cv2.warpAffine with replicate padding."""
    h, w = img.shape[:2]
    out_h, out_w = output_shape if output_shape is not None else (h, w)
    flags = _INTERP.get(method.lower(), cv2.INTER_LINEAR)
    return cv2.warpAffine(img.astype(np.float32), matrix.astype(np.float32), (out_w, out_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


def translate(img, tx: float, ty: float):
    mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return apply_affine(img, mat, method="bilinear")


def scale(img, sx: float, sy: float):
    new_w = max(1, int(round(img.shape[1] * sx)))
    new_h = max(1, int(round(img.shape[0] * sy)))
    pixels = new_w * new_h
    if pixels > MAX_OUTPUT_PIXELS:
        factor = (MAX_OUTPUT_PIXELS / pixels) ** 0.5
        new_w = max(1, int(round(img.shape[1] * sx * factor)))
        new_h = max(1, int(round(img.shape[0] * sy * factor)))
    return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def rotate(img, angle_deg: float):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img.astype(np.float32), mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def shear_x(img, shx: float):
    mat = np.array([[1, shx, 0], [0, 1, 0]], dtype=np.float32)
    out_w = int(img.shape[1] + abs(shx) * img.shape[0])
    return apply_affine(img, mat, output_shape=(img.shape[0], out_w), method="bilinear")


def shear_y(img, shy: float):
    mat = np.array([[1, 0, 0], [shy, 1, 0]], dtype=np.float32)
    out_h = int(img.shape[0] + abs(shy) * img.shape[1])
    return apply_affine(img, mat, output_shape=(out_h, img.shape[1]), method="bilinear")
