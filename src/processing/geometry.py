import cv2
import numpy as np


MAX_OUTPUT_PIXELS = 50_000_000

def apply_affine(img, matrix, output_shape=None):
    """Apply a 2x3 affine matrix with cv2.warpAffine and replicate padding."""
    h, w = img.shape[:2]
    out_h, out_w = output_shape if output_shape is not None else (h, w)
    return cv2.warpAffine(img.astype(np.float32), matrix.astype(np.float32), (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def translate(img, tx: float, ty: float):
    """Shift the image by tx, ty pixels using an affine matrix."""
    mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return apply_affine(img, mat)


def scale(img, sx: float, sy: float):
    """Scale width/height by sx, sy with cv2.resize. Caps output pixels at MAX_OUTPUT_PIXELS to avoid huge arrays. Falls back to bilinear interpolation. Uses float32 for consistent math."""
    new_w = max(1, int(round(img.shape[1] * sx)))
    new_h = max(1, int(round(img.shape[0] * sy)))
    pixels = new_w * new_h
    if pixels > MAX_OUTPUT_PIXELS:
        factor = (MAX_OUTPUT_PIXELS / pixels) ** 0.5
        new_w = max(1, int(round(img.shape[1] * sx * factor)))
        new_h = max(1, int(round(img.shape[0] * sy * factor)))
    return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def rotate(img, angle_deg: float):
    """Rotate around the center by angle_deg degrees. Builds a cv2 rotation matrix and warps with bilinear sampling. Keeps the original canvas size. Uses replicate padding for edges."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img.astype(np.float32), mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def shear_x(img, shx: float):
    """Shear horizontally by shx and expand canvas so content stays visible."""
    h, w = img.shape[:2]
    min_x = min(0, shx * (h - 1))
    max_x = (w - 1) + max(0, shx * (h - 1))
    out_w = int(round(max_x - min_x + 1))
    tx = -min_x
    mat = np.array([[1, shx, tx], [0, 1, 0]], dtype=np.float32)
    return apply_affine(img, mat, output_shape=(h, out_w))


def shear_y(img, shy: float):
    """Shear vertically by shy and expand canvas so content stays visible."""
    h, w = img.shape[:2]
    min_y = min(0, shy * (w - 1))
    max_y = (h - 1) + max(0, shy * (w - 1))
    out_h = int(round(max_y - min_y + 1))
    ty = -min_y
    mat = np.array([[1, 0, 0], [shy, 1, ty]], dtype=np.float32)
    return apply_affine(img, mat, output_shape=(out_h, w))
