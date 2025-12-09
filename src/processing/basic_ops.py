import numpy as np
import cv2


def rgb_to_grayscale(img):
    """Return float32 grayscale; pass through if already 2D."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def ensure_grayscale(img):
    """Wrapper to always produce grayscale float32."""
    return rgb_to_grayscale(img)


def grayscale_to_binary(img):
    """Threshold at mean intensity and return mask plus threshold."""
    gray = ensure_grayscale(img)
    t = float(gray.mean())
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return binary, t


def crop(img, x, y, w, h):
    """Slice a rectangle from (x, y) with width w and height h."""
    return img[y:y + h, x:x + w]


def histogram(img):
    """256-bin grayscale histogram as int32 array."""
    gray = ensure_grayscale(img)
    return cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()


def histogram_goodness(hist):
    """Tiny heuristic describing spread of nonzero bins."""
    total = np.sum(hist)
    if total == 0:
        return "Empty histogram."
    spread = np.count_nonzero(hist) / 256.0
    if spread > 0.7:
        return "Histogram is well-distributed; good contrast."
    if spread > 0.4:
        return "Histogram is moderately spread; contrast is acceptable."
    return "Histogram is concentrated; consider equalization to improve contrast."


def histogram_equalization(img):
    """Equalize grayscale with cv2 and return float32 result."""
    gray = ensure_grayscale(img)
    gray_u8 = cv2.convertScaleAbs(np.clip(gray, 0, 255))
    return cv2.equalizeHist(gray_u8)
