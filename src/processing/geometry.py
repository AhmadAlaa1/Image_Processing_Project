import cv2
import numpy as np


MAX_OUTPUT_PIXELS = 50_000_000

def translate(img, tx: float, ty: float):
    h, w = img.shape[:2]
    mat = np.array([[1, 0, tx],
                    [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(img,mat,(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0 )


def scale(img, sx: float, sy: float):
    new_w = max(1, int(round(img.shape[1] * sx)))
    new_h = max(1, int(round(img.shape[0] * sy)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def rotate(img, angle_deg: float):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shear_x(img, shx: float):
    h, w = img.shape[:2]

    # How far can the top/bottom rows move in X?
    # Top row y=0 → shift = shx * 0 = 0
    # Bottom row y=h-1 → shift = shx * (h-1)
    min_x = min(0, shx * (h - 1))
    max_x = (w - 1) + max(0, shx * (h - 1))

    # New width after shear
    out_w = int(round(max_x - min_x + 1)) 

    # Translate so that the minimum x becomes 0
    tx = -min_x
    mat = np.array([[1, shx, tx], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, mat, (out_w,h) , flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shear_y(img, shy: float):
    h, w = img.shape[:2]
    min_y = min(0, shy * (w - 1))
    max_y = (h - 1) + max(0, shy * (w - 1))
    out_h = int(round(max_y - min_y + 1))
    ty = -min_y
    mat = np.array([[1, 0, 0], [shy, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(img, mat, (w,out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
