import base64
import io
import json
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from processing import basic_ops, compress, filters, geometry, interp, io_utils

COMPRESSION_MAX_PIXELS = 200_000

app = Flask(__name__, static_folder="static", static_url_path="/static")


# ---------- Helpers ---------- #
def _decode_image(data_url: str, max_dim: int = 1600) -> tuple[np.ndarray, dict]:
    """Decode base64 data URL to RGB float32 numpy array, with optional downscale for performance."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    buf = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image.")
    h, w = bgr.shape[:2]
    meta = {"downsized": False, "original_size": (w, h)}
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        meta["downsized"] = True
        meta["new_size"] = (new_w, new_h)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32), meta


def _encode_image(img: np.ndarray) -> str:
    """Encode numpy image to base64 PNG data URL."""
    if img.ndim == 2:
        img_to_save = cv2.cvtColor(io_utils.ensure_uint8(img), cv2.COLOR_GRAY2RGB)
    else:
        img_to_save = cv2.cvtColor(io_utils.ensure_uint8(img), cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_to_save)
    if not success:
        raise ValueError("Failed to encode image.")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _process_request(img: np.ndarray, action: str, params: dict, decode_meta: dict) -> Tuple[np.ndarray, dict]:
    """Apply requested operation and return (image, extra_info)."""
    extra = {}
    if decode_meta.get("downsized"):
        extra["downsized_from"] = decode_meta["original_size"]
        extra["processed_size"] = decode_meta["new_size"]
    gray = basic_ops.ensure_grayscale(img)
    act = action.lower()

    if act == "grayscale":
        return gray, extra
    if act == "binary":
        binary, t = basic_ops.grayscale_to_binary(gray)
        extra["threshold"] = t
        return binary, extra
    if act == "translate":
        tx = float(params.get("tx", 0))
        ty = float(params.get("ty", 0))
        return geometry.translate(img, tx, ty), extra
    if act == "scale":
        sx = float(params.get("sx", 1))
        sy = float(params.get("sy", 1))
        # Guard overly large outputs by auto-adjusting scale.
        target_w = max(1, int(round(img.shape[1] * sx)))
        target_h = max(1, int(round(img.shape[0] * sy)))
        target_pixels = target_w * target_h
        limit = geometry.MAX_OUTPUT_PIXELS
        if target_pixels > limit:
            factor = (limit / target_pixels) ** 0.5
            sx *= factor
            sy *= factor
            adj_w = max(1, int(round(img.shape[1] * sx)))
            adj_h = max(1, int(round(img.shape[0] * sy)))
            extra["scale_adjusted"] = {
                "requested": (target_w, target_h),
                "adjusted": (adj_w, adj_h),
                "factor": round(factor, 3),
                "max_pixels": limit,
            }
        return geometry.scale(img, sx, sy), extra
    if act == "rotate":
        angle = float(params.get("angle", 0))
        return geometry.rotate(img, angle), extra
    if act == "shear_x":
        shx = float(params.get("shx", 0.2))
        return geometry.shear_x(img, shx), extra
    if act == "shear_y":
        shy = float(params.get("shy", 0.2))
        return geometry.shear_y(img, shy), extra
    if act.startswith("resize_"):
        method = act.split("_", 1)[1]
        new_w = int(params.get("width", gray.shape[1]))
        new_h = int(params.get("height", gray.shape[0]))
        return interp.resize(gray, new_w, new_h, method=method), extra
    if act == "crop":
        return basic_ops.crop(img, int(params.get("x", 0)), int(params.get("y", 0)),
                              int(params.get("w", img.shape[1])), int(params.get("h", img.shape[0]))), extra
    if act == "histogram":
        hist = basic_ops.histogram(gray)
        extra["histogram"] = hist.tolist()
        extra["assessment"] = basic_ops.histogram_goodness(hist)
        return gray, extra
    if act == "equalize":
        eq = basic_ops.histogram_equalization(gray)
        hist = basic_ops.histogram(eq)
        extra["histogram"] = hist.tolist()
        extra["assessment"] = basic_ops.histogram_goodness(hist)
        return eq, extra
    if act == "gaussian":
        return filters.gaussian_blur(gray, 19, 3.0), extra
    if act == "median":
        return filters.median_filter(gray, 7), extra
    if act == "laplacian":
        return filters.laplacian_filter(gray), extra
    if act == "sobel":
        return filters.sobel_filter(gray), extra
    if act == "gradient":
        return filters.gradient_first_derivative(gray), extra

    # Compression operations
    if act in {"huffman", "golomb", "arithmetic", "lzw", "rle", "symbol", "bitplane", "dct", "predictive", "wavelet"}:
        if gray.size > COMPRESSION_MAX_PIXELS:
            h, w = gray.shape
            scale = (COMPRESSION_MAX_PIXELS / (h * w)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            extra["compression_downscaled_from"] = (w, h)
            extra["compression_size"] = (new_w, new_h)

    if act == "huffman":
        data = compress.huffman_compress(gray)
        recon = compress.huffman_decompress(data["bitstring"], data["tree"], gray.shape)
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "golomb":
        data = compress.golomb_rice_encode(gray, k=2)
        recon = compress.golomb_rice_decode(data["bitstring"], gray.shape, k=2)
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "arithmetic":
        data = compress.arithmetic_encode(gray)
        recon = compress.arithmetic_decode(data.get("code"), data)
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "lzw":
        data = compress.lzw_encode(gray)
        recon = compress.lzw_decode(data["codes"], gray.shape)
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "rle":
        data = compress.rle_encode(gray)
        recon = compress.rle_decode(data["pairs"], gray.shape)
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "symbol":
        data = compress.symbol_based_encode(gray)
        recon = compress.symbol_based_decode(data["bitstring"], data["codes"], gray.shape)
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "bitplane":
        planes, recon = compress.bit_planes(gray)
        extra["planes"] = [p.tolist() for p in planes]
        return recon, extra
    if act == "dct":
        data = compress.dct_compress(gray)
        extra.update({"ratio": data["ratio"], "kept": data["kept_coefficients"], "total": data["total_coefficients"]})
        return data["image"], extra
    if act == "predictive":
        data = compress.predictive_encode(gray)
        recon = compress.predictive_decode(data["residual"])
        extra.update({"ratio": data["ratio"]})
        return recon, extra
    if act == "wavelet":
        approx, horiz, vert, diag, orig_shape = compress.haar_wavelet_transform(gray)
        recon = compress.haar_wavelet_inverse(approx, horiz, vert, diag, original_shape=orig_shape)
        extra.update({"approx_shape": list(approx.shape)})
        return recon, extra

    raise ValueError(f"Unknown action: {action}")


# ---------- Routes ---------- #
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/process", methods=["POST"])
def process_image():
    try:
        payload = request.get_json(force=True)
        img_data = payload.get("image")
        action = payload.get("action")
        params = payload.get("params", {})
        if img_data is None or action is None:
            return jsonify({"error": "Missing image or action."}), 400

        img, decode_meta = _decode_image(img_data)
        result_img, extra = _process_request(img, action, params, decode_meta)
        encoded = _encode_image(result_img)
        info = io_utils.info(result_img)
        return jsonify({"image": encoded, "info": info, "extra": extra})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
