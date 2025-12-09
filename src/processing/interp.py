import cv2
from . import basic_ops


def bilinear(img, new_width, new_height):
    gray = basic_ops.ensure_grayscale(img)
    w = max(1, int(new_width)) 
    h = max(1, int(new_height))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)


def nearest(img, new_width, new_height):
    gray = basic_ops.ensure_grayscale(img)
    w = max(1, int(new_width)) 
    h = max(1, int(new_height))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)


def bicubic(img, new_width, new_height):
    gray = basic_ops.ensure_grayscale(img)
    w = max(1, int(new_width)) 
    h = max(1, int(new_height))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC)
