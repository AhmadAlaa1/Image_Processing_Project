import cv2
import numpy as np

from . import basic_ops

def gaussian_blur(img, size: int = 19, sigma: float = 3.0):
    """Gaussian blur with an odd kernel size."""
    gray = basic_ops.ensure_grayscale(img)
    k = size if size % 2 == 1 else size + 1
    k = max(3, k)
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


def median_filter(img, size: int = 7):
    """Median blur; converts to 8-bit for OpenCV then returns result."""
    gray = basic_ops.ensure_grayscale(img)
    k = size if size % 2 == 1 else size + 1
    k = max(3, k)
    img_u8 = cv2.convertScaleAbs(gray)
    return cv2.medianBlur(img_u8, k)


def laplacian_filter(img):
    """Laplacian edge response."""
    gray = basic_ops.ensure_grayscale(img)
    return cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)


def sobel_filter(img):
    """Sobel magnitude from x and y derivatives."""
    gray = basic_ops.ensure_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return cv2.magnitude(gx, gy)


def gradient_first_derivative(img):
    """First-derivative gradient using 1x3 Sobel kernels."""
    gray = basic_ops.ensure_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1, borderType=cv2.BORDER_REPLICATE)
    return cv2.magnitude(gx, gy)
