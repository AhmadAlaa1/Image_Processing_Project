import numpy as np
import cv2

def rgb_to_grayscale(img): #Fixed
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

def grayscale_to_binary(grayscale_image):
    """Convert grayscale to binary using average intensity threshold."""
    threshold = float(cv2.mean(grayscale_image)[0])
    _, binary = cv2.threshold(grayscale_image.astype(np.float32), threshold, 255, cv2.THRESH_BINARY)
    return binary, threshold


def crop(img, x, y, w, h): #Fixed
    return img[y:y+h, x:x+w]


def histogram(gray):
    """Compute histogram for grayscale image."""
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
    return hist.astype(np.int32)


def histogram_goodness(hist):
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


def histogram_equalization(gray):
    """Apply histogram equalization via OpenCV."""
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.equalizeHist(gray_uint8).astype(np.float32)
