# Image Processing Toolkit (Tkinter + OpenCV)

Standalone Python GUI that demonstrates classic image processing and compression algorithms with custom, from-scratch implementations. Built only with `tkinter`, `cv2`, `numpy`, and `matplotlib`. A localhost browser UI is also available via Flask.

<img src="Screenshot_20251123_021827.png">
## Features
- Load and preview color images, with info panel (resolution, size, dtype).
- Grayscale and binary conversion (threshold from average intensity, optimality note).
- Affine transforms: translation, scaling, rotation, x/y shear.
- Resizing with nearest-neighbor, bilinear, and bicubic interpolation.
- Cropping by coordinates.
- Histogram analysis with visualization and equalization.
- Filters: 19×19 Gaussian (σ=3), 7×7 median, Laplacian (2nd derivative), Sobel, and first-derivative gradient.
- Compression demos: Huffman, Golomb–Rice, Arithmetic, LZW, Run-Length, Symbol-based frequency coding, Bit-plane, Block DCT, Predictive (DPCM), and Haar Wavelet. Each reports compression ratio and previews decoded output where applicable.

## Project Structure
```
src/
  app.py                  # Tkinter entry point
  ui/controls.py          # GUI layout + event wiring (Tk)
  ui/preview.py           # Canvas rendering via matplotlib (Tk)
  server.py               # Flask server for web UI
  static/                 # Web assets for localhost UI
  processing/io_utils.py  # Load/save helpers
  processing/basic_ops.py # Grayscale, binary, histogram, cropping
  processing/geometry.py  # Affine transforms
  processing/interp.py    # Interpolation & resizing
  processing/filters.py   # Convolution and filtering kernels
  processing/compress.py  # Compression algorithms
```

## Running the desktop app (Tkinter)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/app.py
```

## Running the web app (localhost)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src flask --app src/server run  # or: PYTHONPATH=src python src/server.py
# open http://127.0.0.1:5000
```

## Notes
- All processing uses custom numpy-based implementations (no cv2 convenience functions except for image I/O).
- Matplotlib is used only for rendering into the Tkinter window and showing histograms.
- The GUIs group related operations for clarity and keep a consistent color palette for a tidy student project presentation.
