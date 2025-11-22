import math
import numpy as np


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Vectorized convolution for grayscale or color images (edge-padded)."""
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    if img.ndim == 2:
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        # Build sliding windows view: (H, W, kh, kw)
        shape = (img.shape[0], img.shape[1], kh, kw)
        strides = padded.strides * 2
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)
        out = np.tensordot(windows, kernel, axes=([2, 3], [0, 1])).astype(np.float32)
        return out
    # Color
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    shape = (img.shape[0], img.shape[1], kh, kw, img.shape[2])
    strides = padded.strides[:2] + padded.strides[:2] + (padded.strides[2],)
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)
    out = np.tensordot(windows, kernel, axes=([2, 3], [0, 1])).astype(np.float32)
    return out


def gaussian_kernel(size: int = 19, sigma: float = 3.0) -> np.ndarray:
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_blur(img: np.ndarray, size: int = 19, sigma: float = 3.0) -> np.ndarray:
    kernel = gaussian_kernel(size, sigma)
    return convolve(img, kernel)


def median_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    pad = size // 2
    if img.ndim == 2:
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
        out = np.zeros_like(img, dtype=np.float32)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                window = padded[y:y+size, x:x+size]
                out[y, x] = np.median(window)
        return out
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            window = padded[y:y+size, x:x+size]
            out[y, x] = np.median(window.reshape(-1, img.shape[2]), axis=0)
    return out


def laplacian_filter(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    return convolve(img, kernel)


def sobel_filter(img: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return magnitude


def gradient_first_derivative(img: np.ndarray) -> np.ndarray:
    kx = np.array([[1, -1]], dtype=np.float32)
    ky = np.array([[1], [-1]], dtype=np.float32)
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    return np.sqrt(gx ** 2 + gy ** 2)
