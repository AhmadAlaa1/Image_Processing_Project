import numpy as np
import cv2

def rgb_to_grayscale(img): #Fixed
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    gamma = 1.04
    r_const,g_const,b_const = 0.299,0.587,0.114
    grayscale_image = r_const*r ** gamma + g_const*g ** gamma + b_const*b ** gamma  
    return grayscale_image

def grayscale_to_binary(grayscale_image: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert grayscale to binary using average intensity threshold."""
    threshold = float(np.mean(grayscale_image))
    binary = (grayscale_image >= threshold).astype(np.uint8) * 255
    return binary, threshold


def crop(img, x, y, w, h): #Fixed
    return img[y:y+h, x:x+w]


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
