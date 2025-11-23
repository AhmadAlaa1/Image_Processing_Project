import math
import numpy as np
from processing import interp as interp_ops

MAX_OUTPUT_PIXELS = 50_000_000  # hard cap to prevent runaway allocations


def apply_affine(img: np.ndarray, matrix: np.ndarray, output_shape=None, method: str = "bilinear") -> np.ndarray:
    """Apply custom affine transform using inverse mapping (vectorized sampling)."""
    img = img.astype(np.float32, copy=False)
    h, w = img.shape[:2]
    if output_shape is None:
        out_h, out_w = h, w
    else:
        out_h, out_w = output_shape
    inv = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2, :]
    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))
    src_xs = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    src_ys = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    return interp_ops.sample_image(img, src_xs, src_ys, method=method)


def translate(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return apply_affine(img, mat, method="bilinear")


def scale(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    mat = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)
    out_shape = (int(round(img.shape[0] * sy)), int(round(img.shape[1] * sx)))
    if out_shape[0] <= 0 or out_shape[1] <= 0:
        raise ValueError("Scale produced non-positive dimensions.")
    if out_shape[0] * out_shape[1] > MAX_OUTPUT_PIXELS:
        raise ValueError(f"Scaled image too large ({out_shape[1]}x{out_shape[0]} > {MAX_OUTPUT_PIXELS} pixels). Use smaller scale.")
    return apply_affine(img, mat, output_shape=out_shape, method="bilinear")


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # Build homogeneous 3x3 matrices, multiply, then drop back to 2x3
    translate_to_center = np.array([[1, 0, -cx],
                                    [0, 1, -cy],
                                    [0, 0, 1]], dtype=np.float32)
    rotate_mat = np.array([[cos_a, -sin_a, 0],
                           [sin_a, cos_a, 0],
                           [0, 0, 1]], dtype=np.float32)
    translate_back = np.array([[1, 0, cx],
                                [0, 1, cy],
                                [0, 0, 1]], dtype=np.float32)
    mat3 = translate_back @ rotate_mat @ translate_to_center
    mat2 = mat3[:2, :]
    return apply_affine(img, mat2, method="bilinear")


def shear_x(img: np.ndarray, shx: float) -> np.ndarray:
    mat = np.array([[1, shx, 0], [0, 1, 0]], dtype=np.float32)
    out_w = int(img.shape[1] + abs(shx) * img.shape[0])
    return apply_affine(img, mat, output_shape=(img.shape[0], out_w), method="bilinear")


def shear_y(img: np.ndarray, shy: float) -> np.ndarray:
    mat = np.array([[1, 0, 0], [shy, 1, 0]], dtype=np.float32)
    out_h = int(img.shape[0] + abs(shy) * img.shape[1])
    return apply_affine(img, mat, output_shape=(out_h, img.shape[1]), method="bilinear")
