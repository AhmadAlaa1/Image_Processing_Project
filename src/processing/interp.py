import numpy as np
import math


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sampling."""
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1.0001)
    ys = np.clip(ys, 0, h - 1.0001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = xs - x0
    dy = ys - y0

    if img.ndim == 2:
        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]
        top = (1 - dx) * Ia + dx * Ib
        bottom = (1 - dx) * Ic + dx * Id
        return (1 - dy) * top + dy * bottom

    Ia = img[y0, x0, :]
    Ib = img[y0, x1, :]
    Ic = img[y1, x0, :]
    Id = img[y1, x1, :]
    top = (1 - dx)[..., None] * Ia + dx[..., None] * Ib
    bottom = (1 - dx)[..., None] * Ic + dx[..., None] * Id
    return (1 - dy)[..., None] * top + dy[..., None] * bottom


def _nearest_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x_int = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    y_int = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return img[y_int, x_int] if img.ndim == 2 else img[y_int, x_int, :]


def _bicubic_single(img: np.ndarray, x: float, y: float) -> np.ndarray:
    h, w = img.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros(img.shape[2:] if img.ndim == 3 else (), dtype=np.float32)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))

    def cubic(t: float) -> float:
        t = abs(t)
        if t <= 1:
            return 1 - 2 * t * t + t * t * t
        if t < 2:
            return 4 - 8 * t + 5 * t * t - t * t * t
        return 0

    result = np.zeros(img.shape[2:] if img.ndim == 3 else (), dtype=np.float32)
    weight_sum = 0.0
    for m in range(-1, 3):
        for n in range(-1, 3):
            xm, yn = x0 + n, y0 + m
            if xm < 0 or xm >= w or yn < 0 or yn >= h:
                continue
            w_total = cubic(x - xm) * cubic(y - yn)
            weight_sum += w_total
            if img.ndim == 2:
                result[...] += w_total * img[yn, xm]
            else:
                result += w_total * img[yn, xm]
    if weight_sum == 0:
        return result
    return result / weight_sum


def sample_image(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, method: str = "bilinear") -> np.ndarray:
    """Vectorized sampling for nearest/bilinear; bicubic falls back to per-pixel."""
    method = method.lower()
    if method == "nearest":
        return _nearest_sample(img, xs, ys)
    if method == "bilinear":
        return _bilinear_sample(img, xs, ys)
    # Bicubic fallback (slower)
    out = np.zeros(xs.shape + (img.shape[2],), dtype=np.float32) if img.ndim == 3 else np.zeros_like(xs, dtype=np.float32)
    flat_x = xs.ravel()
    flat_y = ys.ravel()
    for idx, (fx, fy) in enumerate(zip(flat_x, flat_y)):
        val = _bicubic_single(img, float(fx), float(fy))
        if img.ndim == 3:
            out.reshape(-1, img.shape[2])[idx] = val
        else:
            out.ravel()[idx] = val
    return out


def resize(img: np.ndarray, new_width: int, new_height: int, method: str = "nearest") -> np.ndarray:
    """Resize image using chosen interpolation."""
    h, w = img.shape[:2]
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Width and height must be positive.")
    scale_x = w / new_width
    scale_y = h / new_height
    xs, ys = np.meshgrid(np.arange(new_width), np.arange(new_height))
    src_xs = (xs + 0.5) * scale_x - 0.5
    src_ys = (ys + 0.5) * scale_y - 0.5
    return sample_image(img, src_xs, src_ys, method=method)
