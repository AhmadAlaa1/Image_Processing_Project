import numpy as np


def rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminance weights."""
    if img.ndim == 2:
        return img.copy()
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def grayscale_to_binary(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert grayscale to binary using average intensity threshold."""
    threshold = float(np.mean(gray))
    binary = (gray >= threshold).astype(np.uint8) * 255
    return binary, threshold


def crop(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop image safely within bounds."""
    h, w = img.shape[:2]
    x_end = min(w, x + width)
    y_end = min(h, y + height)
    x = max(0, x)
    y = max(0, y)
    if x >= x_end or y >= y_end:
        return img[0:1, 0:1].copy()
    return img[y:y_end, x:x_end].copy()


def histogram(gray: np.ndarray) -> np.ndarray:
    """Compute histogram for grayscale image."""
    hist = np.zeros(256, dtype=np.int32)
    flat = gray.astype(np.uint8).ravel()
    for val in flat:
        hist[val] += 1
    return hist


def histogram_goodness(hist: np.ndarray) -> str:
    """Assess histogram spread."""
    total = np.sum(hist)
    if total == 0:
        return "Empty histogram."
    nonzero_bins = np.count_nonzero(hist)
    spread_ratio = nonzero_bins / 256.0
    if spread_ratio > 0.7:
        return "Histogram is well-distributed; good contrast."
    if spread_ratio > 0.4:
        return "Histogram is moderately spread; contrast is acceptable."
    return "Histogram is concentrated; consider equalization to improve contrast."


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Apply manual histogram equalization."""
    hist = histogram(gray)
    cdf = np.cumsum(hist)
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    total_pixels = gray.size
    equalized = (cdf_masked - cdf_min) * 255.0 / (total_pixels - cdf_min)
    equalized = np.ma.filled(equalized, 0).astype(np.uint8)
    flat = gray.astype(np.uint8).ravel()
    result_flat = equalized[flat]
    return result_flat.reshape(gray.shape).astype(np.float32)
