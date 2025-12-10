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


# ---------- Preview helpers ---------- #
def _matrix_preview(arr, max_rows: int = 4, max_cols: int = 8, round_to: int | None = 0) -> str:
    """Compact text preview of a matrix or vector."""
    a = np.asarray(arr)
    if a.size == 0:
        return "[]"
    if round_to is not None and np.issubdtype(a.dtype, np.floating):
        a = np.round(a, round_to)
    if a.ndim == 1:
        clip = min(max_cols, a.size)
        vals = a[:clip].tolist()
        suffix = "..." if a.size > clip else ""
        return f"shape={a.shape}, first {clip}: {vals}{suffix}"
    rows = min(max_rows, a.shape[0])
    cols = min(max_cols, a.shape[1])
    preview = a[:rows, :cols].tolist()
    suffix = "..." if (a.shape[0] > rows or a.shape[1] > cols) else ""
    return f"shape={a.shape}, top-left {rows}x{cols}: {preview}{suffix}"


# ---------- Helpers ---------- #
def _decode_image(data_url: str, max_dim: int | None = 1600) -> tuple[np.ndarray, dict]:
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
    if max_dim is not None and max(h, w) > max_dim:
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
    gray = basic_ops.rgb_to_grayscale(img)
    act = action.lower()

    if act == "grayscale":
        return gray, extra
    if act == "binary":
        raw_t = params.get("threshold")
        parsed_t = None
        if raw_t not in (None, "", "auto"):
            try:
                parsed_t = float(raw_t)
                extra["threshold_requested"] = parsed_t
            except (TypeError, ValueError):
                extra["threshold_error"] = "Invalid threshold; defaulted to mean."
        binary, t, note = basic_ops.grayscale_to_binary(gray, threshold=parsed_t)
        extra["threshold"] = t
        extra["threshold_eval"] = note
        extra["threshold_mode"] = "manual" if parsed_t is not None else "mean"
        return binary, extra
    if act == "translate":
        tx = float(params.get("tx", 0))
        ty = float(params.get("ty", 0))
        return geometry.translate(gray, tx, ty), extra
    if act == "scale":
        sx = float(params.get("sx", 1))
        sy = float(params.get("sy", 1))
        # Guard overly large outputs by auto-adjusting scale.
        target_w = max(1, int(round(gray.shape[1] * sx)))
        target_h = max(1, int(round(gray.shape[0] * sy)))
        target_pixels = target_w * target_h
        limit = geometry.MAX_OUTPUT_PIXELS
        if target_pixels > limit:
            factor = (limit / target_pixels) ** 0.5
            sx *= factor
            sy *= factor
            adj_w = max(1, int(round(gray.shape[1] * sx)))
            adj_h = max(1, int(round(gray.shape[0] * sy)))
            extra["scale_adjusted"] = {
                "requested": (target_w, target_h),
                "adjusted": (adj_w, adj_h),
                "factor": round(factor, 3),
                "max_pixels": limit,
            }
        return geometry.scale(gray, sx, sy), extra
    if act == "rotate":
        angle = float(params.get("angle", 0))
        return geometry.rotate(gray, angle), extra
    if act == "shear_x":
        shx = float(params.get("shx", 0.2))
        return geometry.shear_x(gray, shx), extra
    if act == "shear_y":
        shy = float(params.get("shy", 0.2))
        return geometry.shear_y(gray, shy), extra
    if act.startswith("resize_"):
        method = act.split("_", 1)[1]
        new_w = int(params.get("width", gray.shape[1]))
        new_h = int(params.get("height", gray.shape[0]))
        resized = None
        if method == "nearest":
            resized = interp.nearest(gray, new_width=new_w, new_height=new_h)
        if method == "bilinear":
            resized = interp.bilinear(gray, new_width=new_w, new_height=new_h)
        if method == "bicubic":
            resized = interp.bicubic(gray, new_width=new_w, new_height=new_h)
        if resized is None:
            resized = interp.bilinear(gray, new_width=new_w, new_height=new_h)

        diff_note = ""
        if method in {"bilinear", "bicubic"}:
            bilinear_out = interp.bilinear(gray, new_width=new_w, new_height=new_h)
            bicubic_out = interp.bicubic(gray, new_width=new_w, new_height=new_h)
            diff = bicubic_out - bilinear_out
            mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            extra["resize_diff"] = {
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "preview": _matrix_preview(diff, round_to=3),
            }
            diff_note = f" | bilinear vs bicubic: mean|Δ|={mean_abs:.3f}, max|Δ|={max_abs:.3f}"

        extra["encoded_preview"] = f"Resize ({method}) preview: {_matrix_preview(resized, round_to=2)}{diff_note}"
        return resized, extra
    if act == "crop":
        return basic_ops.crop(gray, int(params.get("x", 0)), int(params.get("y", 0)),
                              int(params.get("w", gray.shape[1])), int(params.get("h", gray.shape[0]))), extra
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
        full_res = bool(params.get("full_res"))
        if gray.size > COMPRESSION_MAX_PIXELS and not full_res:
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
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        bitstring = data["bitstring"]
        extra["encoded_preview"] = f"Huffman bitstring len={len(bitstring)}: {bitstring[:96]}{'...' if len(bitstring) > 96 else ''}"
        return recon, extra
    if act == "golomb":
        data = compress.golomb_rice_encode(gray, k=2)
        recon = compress.golomb_rice_decode(data["bitstring"], gray.shape, k=2)
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        bitstring = data["bitstring"]
        extra["encoded_preview"] = f"Golomb-Rice bitstring len={len(bitstring)}: {bitstring[:96]}{'...' if len(bitstring) > 96 else ''}"
        return recon, extra
    if act == "arithmetic":
        data = compress.arithmetic_encode(gray)
        recon = compress.arithmetic_decode(data.get("code"), data)
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        bitstring = data.get("code", "")
        extra["encoded_preview"] = f"Arithmetic code len={len(bitstring)}: {bitstring[:96]}{'...' if len(bitstring) > 96 else ''}"
        return recon, extra
    if act == "lzw":
        data = compress.lzw_encode(gray)
        recon = compress.lzw_decode(data["codes"], gray.shape)
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        codes = data["codes"]
        clip = min(len(codes), 32)
        extra["encoded_preview"] = f"LZW codes len={len(codes)}, first {clip}: {codes[:clip]}{'...' if len(codes) > clip else ''}"
        return recon, extra
    if act == "rle":
        data = compress.rle_encode(gray)
        recon = compress.rle_decode(data["pairs"], gray.shape)
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        pairs = data["pairs"]
        clip = min(len(pairs), 12)
        extra["encoded_preview"] = f"RLE pairs count={len(pairs)}, first {clip}: {pairs[:clip]}{'...' if len(pairs) > clip else ''}"
        return recon, extra
    if act == "symbol":
        data = compress.symbol_based_encode(gray)
        recon = compress.symbol_based_decode(data["bitstring"], data["codes"], gray.shape)
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        bitstring = data["bitstring"]
        extra["encoded_preview"] = f"Symbol bitstring len={len(bitstring)}: {bitstring[:96]}{'...' if len(bitstring) > 96 else ''}"
        return recon, extra
    if act == "bitplane":
        planes, recon = compress.bit_planes(gray)
        extra["planes"] = [p.tolist() for p in planes]
        extra["encoded_preview"] = f"Bit-plane 0: {_matrix_preview(planes[0], round_to=0)}"
        return recon, extra
    if act == "dct":
        data = compress.dct_compress(gray)
        extra.update({"ratio": data["ratio"], "kept": data["kept_coefficients"], "total": data["total_coefficients"], "original_bits": data.get("original_bits"), "compressed_bits": data.get("compressed_bits")})
        if data.get("threshold") is not None:
            extra["encoded_preview"] = f"DCT kept {data['kept_coefficients']}/{data['total_coefficients']} coeffs; threshold {data['threshold']:.2f}"
        return data["image"], extra
    if act == "predictive":
        data = compress.predictive_encode(gray)
        recon = compress.predictive_decode(data["residual"])
        extra.update({"ratio": data["ratio"], "original_bits": data["original_bits"], "compressed_bits": data["compressed_bits"]})
        extra["encoded_preview"] = f"Residual preview: {_matrix_preview(data['residual'], round_to=None)}"
        return recon, extra
    if act == "wavelet":
        approx, horiz, vert, diag, orig_shape = compress.haar_wavelet_transform(gray)
        recon = compress.haar_wavelet_inverse(approx, horiz, vert, diag, original_shape=orig_shape)
        extra.update({"approx_shape": list(approx.shape)})
        extra["encoded_preview"] = f"Haar approx: {_matrix_preview(approx, round_to=2)}"
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

        compress_actions = {"huffman", "golomb", "arithmetic", "lzw", "rle", "symbol", "bitplane", "dct", "predictive", "wavelet"}
        full_res = bool(params.get("full_res")) if action in compress_actions else False
        decode_max_dim = None if full_res and action in compress_actions else 1600

        img, decode_meta = _decode_image(img_data, max_dim=decode_max_dim)
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
