import cv2
import numpy as np


def convolve(img, kernel):
    """Convolution via OpenCV with replicate padding."""
    return cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


def gaussian_blur(img, size: int = 19, sigma: float = 3.0):
    k = size if size % 2 == 1 else size + 1
    k = max(3, k)
    img32 = img.astype(np.float32, copy=False)
    return cv2.GaussianBlur(img32, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


def median_filter(img, size: int = 7):
    k = size if size % 2 == 1 else size + 1
    k = max(3, k)
    img_u8 = cv2.convertScaleAbs(img)
    return cv2.medianBlur(img_u8, k).astype(np.float32)


def laplacian_filter(img):
    return cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)


def sobel_filter(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return cv2.magnitude(gx, gy)


def gradient_first_derivative(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1, borderType=cv2.BORDER_REPLICATE)
    return cv2.magnitude(gx, gy)
