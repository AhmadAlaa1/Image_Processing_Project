import math
from collections import Counter
import heapq
import cv2
import numpy as np


def _to_gray_uint8(img):
    if img.ndim == 3:
        # simple average for compression routines
        img = np.mean(img, axis=2)
        return np.clip(img, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _compression_ratio(original_bits: int, compressed_bits: int) -> float:
    if compressed_bits == 0:
        return 0.0
    return original_bits / compressed_bits


# ---------------- Huffman Coding ---------------- #
class _HuffNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(data):
    flat = data.ravel().tolist()
    freq = Counter(flat)
    heap = []
    for sym, f in freq.items():
        heapq.heappush(heap, (f, _HuffNode(f, sym)))
    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        merged = _HuffNode(f1 + f2, None, n1, n2)
        heapq.heappush(heap, (merged.freq, merged))
    return heap[0][1]


def _generate_codes(node: _HuffNode, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
        return codes
    _generate_codes(node.left, prefix + "0", codes)
    _generate_codes(node.right, prefix + "1", codes)
    return codes


def huffman_compress(img):
    gray = _to_gray_uint8(img)
    data = gray.ravel().tolist()
    tree = _build_huffman_tree(gray)
    codes = _generate_codes(tree)
    bitstring = "".join(codes[val] for val in data)
    original_bits = len(data) * 8
    compressed_bits = len(bitstring)
    return {
        "bitstring": bitstring,
        "codes": codes,
        "tree": tree,
        "ratio": _compression_ratio(original_bits, compressed_bits),
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
    }


def huffman_decompress(bitstring: str, tree: _HuffNode, shape):
    decoded_vals = []
    node = tree
    for bit in bitstring:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            decoded_vals.append(node.symbol)
            node = tree
    arr = np.array(decoded_vals, dtype=np.uint8)
    if arr.size < shape[0] * shape[1]:
        pad = shape[0] * shape[1] - arr.size
        arr = np.pad(arr, (0, pad), mode="edge")
    return arr.reshape(shape)


# ---------------- Golomb-Rice Coding ---------------- #
def golomb_rice_encode(img, k: int = 2):
    gray = _to_gray_uint8(img)
    m = 1 << k
    bits = []
    for val in gray.ravel():
        q = val // m
        r = val % m
        bits.append("1" * q + "0")
        bits.append(format(r, f"0{k}b"))
    bitstring = "".join(bits)
    original_bits = gray.size * 8
    return {
        "bitstring": bitstring,
        "k": k,
        "ratio": _compression_ratio(original_bits, len(bitstring)),
        "original_bits": original_bits,
        "compressed_bits": len(bitstring),
    }


def golomb_rice_decode(bitstring: str, shape, k: int = 2):
    m = 1 << k
    values = []
    idx = 0
    while idx < len(bitstring) and len(values) < shape[0] * shape[1]:
        q = 0
        while idx < len(bitstring) and bitstring[idx] == "1":
            q += 1
            idx += 1
        idx += 1  # skip zero
        if idx + k > len(bitstring):
            break
        r = int(bitstring[idx:idx + k], 2)
        idx += k
        values.append(q * m + r)
    arr = np.array(values, dtype=np.uint8)
    if arr.size < shape[0] * shape[1]:
        arr = np.pad(arr, (0, shape[0] * shape[1] - arr.size), mode="edge")
    return arr.reshape(shape)


# ---------------- Arithmetic Coding (entropy-estimate demo) ---------------- #
def arithmetic_encode(img):
    """Binary arithmetic coding for 8-bit grayscale."""
    gray = _to_gray_uint8(img)
    data = gray.ravel().tolist()
    if not data:
        return {"code": "", "bitstring": "", "ratio": 0.0, "original_bits": 0, "compressed_bits": 0, "shape": gray.shape, "freq": [0] * 256, "length": 0}

    # Static model from symbol frequencies
    freq = np.bincount(data, minlength=256).astype(int)
    total = int(freq.sum())
    cumulative = [0]
    for f in freq:
        cumulative.append(cumulative[-1] + int(f))

    TOP = 1 << 32
    HALF = TOP >> 1
    FIRST_QTR = HALF >> 1
    THIRD_QTR = FIRST_QTR * 3

    low, high = 0, TOP - 1
    bits = []
    bits_to_follow = 0

    def _output_bit(bit):
        bits.append("1" if bit else "0")
        complement = "0" if bit else "1"
        if bits_to_follow:
            bits.extend(complement * bits_to_follow)

    for sym in data:
        rng = high - low + 1
        high = low + (rng * cumulative[sym + 1] // total) - 1
        low = low + (rng * cumulative[sym] // total)

        while True:
            if high < HALF:
                _output_bit(0)
                bits_to_follow = 0
            elif low >= HALF:
                _output_bit(1)
                bits_to_follow = 0
                low -= HALF
                high -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                bits_to_follow += 1
                low -= FIRST_QTR
                high -= FIRST_QTR
            else:
                break
            low = low * 2
            high = high * 2 + 1

    # Final bit flush
    final_bit = 0 if low < FIRST_QTR else 1
    _output_bit(final_bit)

    bitstring = "".join(bits)
    original_bits = len(data) * 8
    compressed_bits = len(bitstring)
    return {
        "code": bitstring,
        "bitstring": bitstring,
        "ratio": _compression_ratio(original_bits, compressed_bits),
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "shape": gray.shape,
        "freq": freq.tolist(),
        "length": len(data),
    }


def arithmetic_decode(code, meta):
    """Decode arithmetic-coded bitstring using stored frequency model."""
    bitstring = code or ""
    freq = meta.get("freq", [0] * 256)
    total_symbols = int(meta.get("length", 0))
    shape = meta.get("shape", (0, 0))
    if total_symbols == 0:
        return np.zeros(shape, dtype=np.uint8)

    cumulative = [0]
    for f in freq:
        cumulative.append(cumulative[-1] + int(f))
    total = cumulative[-1]

    TOP = 1 << 32
    HALF = TOP >> 1
    FIRST_QTR = HALF >> 1
    THIRD_QTR = FIRST_QTR * 3

    def next_bit(idx):
        return 1 if (idx < len(bitstring) and bitstring[idx] == "1") else 0

    # Prime the decoder with 32 bits
    value = 0
    for i in range(32):
        value = (value << 1) | next_bit(i)
    bit_idx = 32
    low, high = 0, TOP - 1
    output = []

    for _ in range(total_symbols):
        rng = high - low + 1
        scaled = ((value - low + 1) * total - 1) // rng

        # Find symbol: cumulative[s] <= scaled < cumulative[s+1]
        sym = 0
        # Small linear search is fine for 256 symbols
        for s in range(256):
            if cumulative[s] <= scaled < cumulative[s + 1]:
                sym = s
                break

        output.append(sym)
        high = low + (rng * cumulative[sym + 1] // total) - 1
        low = low + (rng * cumulative[sym] // total)

        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                low -= FIRST_QTR
                high -= FIRST_QTR
                value -= FIRST_QTR
            else:
                break
            low = low * 2
            high = high * 2 + 1
            value = value * 2 + next_bit(bit_idx)
            bit_idx += 1

    arr = np.array(output, dtype=np.uint8)
    return arr[: shape[0] * shape[1]].reshape(shape)


# ---------------- LZW Coding ---------------- #
def lzw_encode(img):
    gray = _to_gray_uint8(img)
    data = gray.ravel().tolist()
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    w = bytes([data[0]])
    codes = []
    for k in data[1:]:
        wk = w + bytes([k])
        if wk in dictionary:
            w = wk
        else:
            codes.append(dictionary[w])
            dictionary[wk] = dict_size
            dict_size += 1
            w = bytes([k])
    codes.append(dictionary[w])
    original_bits = len(data) * 8
    bits_len = math.ceil(math.log2(dict_size + 1))
    compressed_bits = len(codes) * bits_len
    return {
        "codes": codes,
        "dict_size": dict_size,
        "bits_len": bits_len,
        "ratio": _compression_ratio(original_bits, compressed_bits),
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "shape": gray.shape,
    }


def lzw_decode(codes, shape):
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    w = bytes([codes[0]])
    result = bytearray(w)
    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + bytes([w[0]])
        else:
            entry = bytes([0])
        result.extend(entry)
        dictionary[dict_size] = w + bytes([entry[0]])
        dict_size += 1
        w = entry
    arr = np.frombuffer(result, dtype=np.uint8)
    return arr[:shape[0] * shape[1]].reshape(shape)


# ---------------- Run Length Encoding ---------------- #
def rle_encode(img):
    gray = _to_gray_uint8(img)
    data = gray.ravel()
    pairs = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            pairs.append((int(data[i - 1]), count))
            count = 1
    pairs.append((int(data[-1]), count))
    bit_estimate = len(pairs) * (8 + 16)
    ratio = _compression_ratio(len(data) * 8, bit_estimate)
    return {"pairs": pairs, "ratio": ratio, "original_bits": len(data) * 8, "compressed_bits": bit_estimate, "shape": gray.shape}


def rle_decode(pairs, shape):
    values = []
    for val, count in pairs:
        values.extend([val] * count)
    arr = np.array(values, dtype=np.uint8)
    return arr[:shape[0] * shape[1]].reshape(shape)


# ---------------- Symbol-based coding ---------------- #
def symbol_based_encode(img):
    gray = _to_gray_uint8(img)
    freq = Counter(gray.ravel().tolist())
    sorted_symbols = [s for s, _ in freq.most_common()]
    bits_needed = max(1, math.ceil(math.log2(len(sorted_symbols) or 1)))
    codes = {sym: format(idx, f"0{bits_needed}b") for idx, sym in enumerate(sorted_symbols)}
    bitstring = "".join(codes[int(v)] for v in gray.ravel())
    compressed_bits = len(bitstring)
    original_bits = gray.size * 8
    return {
        "codes": codes,
        "bitstring": bitstring,
        "bits_len": bits_needed,
        "ratio": _compression_ratio(original_bits, compressed_bits),
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "shape": gray.shape,
    }


def symbol_based_decode(bitstring: str, codes: dict, shape):
    reverse = {v: k for k, v in codes.items()}
    bits_len = max(len(k) for k in reverse.keys())
    values = []
    for i in range(0, len(bitstring), bits_len):
        chunk = bitstring[i:i + bits_len]
        if chunk in reverse:
            values.append(reverse[chunk])
    arr = np.array(values, dtype=np.uint8)
    target = shape[0] * shape[1]
    if arr.size < target:
        arr = np.pad(arr, (0, target - arr.size), mode="edge")
    if arr.size > target:
        arr = arr[:target]
    return arr.reshape(shape)


# ---------------- Bit-plane coding ---------------- #
def bit_planes(img):
    gray = _to_gray_uint8(img)
    planes = []
    for i in range(8):
        plane = ((gray >> i) & 1) * 255
        planes.append(plane.astype(np.float32))
    reconstructed = sum(((planes[i] > 0).astype(np.uint8) << i) for i in range(8)).astype(np.float32)
    return planes, reconstructed


# ---------------- Block DCT coding ---------------- #
def _dct_matrix(n=8):
    C = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            C[k, i] = alpha * math.cos((math.pi * (2 * i + 1) * k) / (2 * n))
    return C


DCT_MATRIX = _dct_matrix()
DCT_MATRIX_T = DCT_MATRIX.T


def dct_compress(img, keep_ratio: float = 0.5):
    gray = _to_gray_uint8(img)
    h, w = gray.shape
    h8, w8 = h - (h % 8), w - (w % 8)
    gray = gray[:h8, :w8]
    reconstructed = np.zeros_like(gray, dtype=np.float32)
    total_coeffs = kept = 0
    threshold = None
    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = gray[y:y+8, x:x+8].astype(np.float32) - 128
            dct = cv2.dct(block)
            flat = np.abs(dct).ravel()
            total_coeffs += flat.size
            k = max(1, int(flat.size * keep_ratio))
            threshold = np.partition(flat, -k)[-k]
            mask = (np.abs(dct) >= threshold).astype(np.float32)
            kept += int(np.count_nonzero(mask))
            dct_filtered = dct * mask
            block_rec = cv2.idct(dct_filtered) + 128
            reconstructed[y:y+8, x:x+8] = block_rec
    compressed_bits = kept * 16
    original_bits = gray.size * 8
    return {
        "image": reconstructed,
        "ratio": _compression_ratio(original_bits, compressed_bits),
        "kept_coefficients": kept,
        "total_coefficients": total_coeffs,
        "threshold": threshold,
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
    }


def dct_basis_grid(tile_size: int = 24):
    """Generate a tiled 8x8 DCT basis visualization."""
    tile = max(8, int(tile_size))
    grid = np.zeros((8 * tile, 8 * tile), dtype=np.float32)
    for v in range(8):
        for u in range(8):
            coeff = np.zeros((8, 8), dtype=np.float32)
            coeff[v, u] = 1.0
            basis = cv2.idct(coeff)
            # Normalize to 0-255 for visualization
            basis_norm = basis - basis.min()
            if basis_norm.max() > 0:
                basis_norm = basis_norm / basis_norm.max()
            basis_norm = (basis_norm * 255.0).astype(np.float32)
            block = cv2.resize(basis_norm, (tile, tile), interpolation=cv2.INTER_NEAREST)
            y0, x0 = v * tile, u * tile
            grid[y0:y0 + tile, x0:x0 + tile] = block
    return grid


def bitstring_preview_image(bitstring: str, width: int = 64, max_rows: int = 64):
    """Convert leading bits to a small black/white image for visualization."""
    bits = (bitstring or "")
    if len(bits) == 0:
        return np.zeros((1, width), dtype=np.float32)
    max_bits = width * max_rows
    bits = bits[:max_bits]
    rows = math.ceil(len(bits) / width)
    rows = min(rows, max_rows)
    padded = bits.ljust(rows * width, "0")
    arr = np.frombuffer(padded.encode("ascii"), dtype=np.uint8)
    arr = (arr - ord("0")).astype(np.uint8) * 255
    img = arr.reshape(rows, width).astype(np.float32)
    return img

# ---------------- Predictive coding (DPCM) ---------------- #
def predictive_encode(img):
    gray = _to_gray_uint8(img)
    residual = np.zeros_like(gray, dtype=np.int16)
    predictor = np.zeros_like(gray, dtype=np.uint8)
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            pred = gray[y, x - 1] if x > 0 else 0
            predictor[y, x] = pred
            residual[y, x] = int(gray[y, x]) - int(pred)
    bits = residual.size * 4
    return {"residual": residual, "predictor": predictor, "ratio": _compression_ratio(gray.size * 8, bits), "original_bits": gray.size * 8, "compressed_bits": bits}


def predictive_decode(residual):
    h, w = residual.shape
    reconstructed = np.zeros((h, w), dtype=np.int16)
    for y in range(h):
        for x in range(w):
            pred = reconstructed[y, x - 1] if x > 0 else 0
            reconstructed[y, x] = pred + residual[y, x]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


# ---------------- Wavelet coding (Haar, 1 level) ---------------- #
def haar_wavelet_transform(img):
    gray = _to_gray_uint8(img).astype(np.float32)
    h, w = gray.shape
    h2, w2 = (h + 1) // 2, (w + 1) // 2  # ceil to handle odd sizes
    approx = np.zeros((h2, w2), dtype=np.float32)
    horiz = np.zeros_like(approx)
    vert = np.zeros_like(approx)
    diag = np.zeros_like(approx)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            a = gray[y, x]
            b = gray[y, x + 1] if x + 1 < w else a
            c = gray[y + 1, x] if y + 1 < h else a
            d = gray[y + 1, x + 1] if (y + 1 < h and x + 1 < w) else a
            idx_y = y // 2
            idx_x = x // 2
            approx[idx_y, idx_x] = (a + b + c + d) / 4
            horiz[idx_y, idx_x] = (a + c - b - d) / 4
            vert[idx_y, idx_x] = (a + b - c - d) / 4
            diag[idx_y, idx_x] = (a - b - c + d) / 4
    return approx, horiz, vert, diag, (h, w)


def haar_wavelet_inverse(approx, horiz, vert, diag, original_shape=None):
    h2, w2 = approx.shape
    h, w = h2 * 2, w2 * 2
    reconstructed = np.zeros((h, w), dtype=np.float32)
    for y in range(h2):
        for x in range(w2):
            a = approx[y, x] + horiz[y, x] + vert[y, x] + diag[y, x]
            b = approx[y, x] - horiz[y, x] + vert[y, x] - diag[y, x]
            c = approx[y, x] + horiz[y, x] - vert[y, x] - diag[y, x]
            d = approx[y, x] - horiz[y, x] - vert[y, x] + diag[y, x]
            reconstructed[2 * y, 2 * x] = a
            reconstructed[2 * y, 2 * x + 1] = b
            reconstructed[2 * y + 1, 2 * x] = c
            reconstructed[2 * y + 1, 2 * x + 1] = d
    reconstructed = np.clip(reconstructed, 0, 255)
    if original_shape:
        oh, ow = original_shape
        return reconstructed[:oh, :ow]
    return reconstructed
