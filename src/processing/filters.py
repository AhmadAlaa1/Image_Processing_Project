import cv2
import numpy as np

from . import basic_ops


def convolve(img, kernel):
    """Convolution via OpenCV with replicate padding."""
    gray = basic_ops.ensure_grayscale(img)
    return cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


def gaussian_blur(img, size: int = 19, sigma: float = 3.0):
    gray = basic_ops.ensure_grayscale(img)
    k = size if size % 2 == 1 else size + 1
    k = max(3, k)
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


def median_filter(img, size: int = 7):
    gray = basic_ops.ensure_grayscale(img)
    k = size if size % 2 == 1 else size + 1
    k = max(3, k)
    img_u8 = cv2.convertScaleAbs(gray)
    return cv2.medianBlur(img_u8, k).astype(np.float32)


def laplacian_filter(img):
    gray = basic_ops.ensure_grayscale(img)
    return cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)


def sobel_filter(img):
    gray = basic_ops.ensure_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return cv2.magnitude(gx, gy)


def gradient_first_derivative(img):
    gray = basic_ops.ensure_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1, borderType=cv2.BORDER_REPLICATE)
    return cv2.magnitude(gx, gy)
