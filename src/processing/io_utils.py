import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load image from disk as RGB float array."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Unable to read image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Clip and convert to uint8 for display."""
    clipped = np.clip(img, 0, 255)
    return clipped.astype(np.uint8)


def info(img: np.ndarray) -> dict:
    """Return basic image info."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    dtype = str(img.dtype)
    return {"width": w, "height": h, "channels": channels, "dtype": dtype}
