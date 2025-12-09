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
    """Binary mask using mean threshold plus a quick optimality check."""
    gray = ensure_grayscale(img)
    t = float(gray.mean())
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    gray_u8 = cv2.convertScaleAbs(gray)
    otsu_t, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = abs(t - float(otsu_t))
    if diff < 5:
        eval_note = f"Threshold looks optimal (mean≈Otsu, diff={diff:.1f})."
    else:
        eval_note = f"Threshold likely suboptimal (mean vs Otsu diff={diff:.1f}); consider Otsu."
    return binary, t, eval_note


def crop(img, x, y, w, h):
    """Slice a rectangle from (x, y) with width w and height h."""
    return img[y:y + h, x:x + w]


def histogram(img):
    """256-bin grayscale histogram as int32 array."""
    gray = ensure_grayscale(img)
    return cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()


def histogram_goodness(hist):
    """Heuristic with a short justification about contrast/brightness spread."""
    total = np.sum(hist)
    if total == 0:
        return "Empty histogram."
    spread = np.count_nonzero(hist) / 256.0
    low_mass = float(np.sum(hist[:64])) / total
    high_mass = float(np.sum(hist[192:])) / total
    entropy = -np.sum((hist / total) * np.log2((hist / total) + 1e-9))
    if spread > 0.7 and 0.1 < low_mass < 0.5 and 0.1 < high_mass < 0.5:
        return f"Histogram is well-distributed (spread={spread:.2f}, entropy={entropy:.2f}); contrast looks good."
    if spread > 0.4:
        return f"Histogram is moderately spread (spread={spread:.2f}); contrast is acceptable."
    if low_mass > 0.6:
        return f"Histogram is concentrated in dark tones (low_mass={low_mass:.2f}); image may be underexposed."
    if high_mass > 0.6:
        return f"Histogram is concentrated in bright tones (high_mass={high_mass:.2f}); image may be overexposed."
    return f"Histogram is narrow (spread={spread:.2f}, entropy={entropy:.2f}); consider equalization to improve contrast."


def histogram_equalization(img):
    """Equalize grayscale with cv2 and return float32 result."""
    gray = ensure_grayscale(img)
    gray_u8 = cv2.convertScaleAbs(np.clip(gray, 0, 255))
    return cv2.equalizeHist(gray_u8)
