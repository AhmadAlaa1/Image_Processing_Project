import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from processing import basic_ops, compress, filters, geometry, interp, io_utils
from ui.preview import ImagePreview


class ImageApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=10)
        root.configure(bg="#111827")
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#111827")
        self.style.configure("TLabel", background="#111827", foreground="#e5e7eb")
        self.style.configure("TButton", background="#1f2937", foreground="#e5e7eb", padding=6)
        self.style.map("TButton", background=[("active", "#2563eb")])

        self.original_img = None
        self.processed_img = None
        self.binary_threshold = None

        self.grid(column=0, row=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self._build_layout()

    # ---------- Layout ---------- #
    def _build_layout(self):
        header = ttk.Label(self, text="Image Processing Studio", font=("Segoe UI", 18, "bold"))
        header.grid(column=0, row=0, sticky="w", pady=(0, 6))

        container = ttk.Frame(self)
        container.grid(column=0, row=1, sticky="nsew")
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        controls = ttk.Frame(container, padding=8)
        controls.grid(column=0, row=0, sticky="ns")
        preview_frame = ttk.Frame(container)
        preview_frame.grid(column=1, row=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        self.preview = ImagePreview(preview_frame)
        self.info_label = ttk.Label(preview_frame, text="No image loaded", anchor="center")
        self.info_label.pack(fill="x", pady=4)

        # File section
        file_box = self._labeled_box(controls, "Image I/O")
        ttk.Button(file_box, text="Load Image", command=self.load_image).grid(column=0, row=0, sticky="ew", padx=4, pady=2)
        self.image_path_var = tk.StringVar(value="No file chosen")
        ttk.Label(file_box, textvariable=self.image_path_var, wraplength=180).grid(column=0, row=1, sticky="w", padx=4)

        # Conversion section
        conv_box = self._labeled_box(controls, "Convert")
        ttk.Button(conv_box, text="To Grayscale", command=self.to_grayscale).grid(column=0, row=0, sticky="ew", padx=4, pady=2)
        ttk.Button(conv_box, text="To Binary (avg threshold)", command=self.to_binary).grid(column=0, row=1, sticky="ew", padx=4, pady=2)

        # Affine section
        affine_box = self._labeled_box(controls, "Affine Transforms")
        self.tx_var = tk.DoubleVar(value=30)
        self.ty_var = tk.DoubleVar(value=30)
        ttk.Label(affine_box, text="tx, ty").grid(column=0, row=0, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.tx_var, width=6).grid(column=0, row=1, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.ty_var, width=6).grid(column=1, row=1, sticky="w", padx=4)
        ttk.Button(affine_box, text="Translate", command=self.translate).grid(column=0, row=2, columnspan=2, sticky="ew", padx=4, pady=2)
        self.sx_var = tk.DoubleVar(value=1.2)
        self.sy_var = tk.DoubleVar(value=1.2)
        ttk.Label(affine_box, text="sx, sy").grid(column=0, row=3, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.sx_var, width=6).grid(column=0, row=4, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.sy_var, width=6).grid(column=1, row=4, sticky="w", padx=4)
        ttk.Button(affine_box, text="Scale", command=self.scale).grid(column=0, row=5, columnspan=2, sticky="ew", padx=4, pady=2)
        self.angle_var = tk.DoubleVar(value=20)
        ttk.Label(affine_box, text="Angle").grid(column=0, row=6, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.angle_var, width=6).grid(column=1, row=6, sticky="w", padx=4)
        ttk.Button(affine_box, text="Rotate", command=self.rotate).grid(column=0, row=7, columnspan=2, sticky="ew", padx=4, pady=2)
        self.shx_var = tk.DoubleVar(value=0.2)
        self.shy_var = tk.DoubleVar(value=0.2)
        ttk.Label(affine_box, text="shx, shy").grid(column=0, row=8, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.shx_var, width=6).grid(column=0, row=9, sticky="w", padx=4)
        ttk.Entry(affine_box, textvariable=self.shy_var, width=6).grid(column=1, row=9, sticky="w", padx=4)
        ttk.Button(affine_box, text="Shear X", command=self.shear_x).grid(column=0, row=10, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(affine_box, text="Shear Y", command=self.shear_y).grid(column=0, row=11, columnspan=2, sticky="ew", padx=4, pady=2)

        # Interpolation / resizing
        interp_box = self._labeled_box(controls, "Interpolation")
        self.new_w = tk.IntVar(value=320)
        self.new_h = tk.IntVar(value=240)
        ttk.Label(interp_box, text="W, H").grid(column=0, row=0, sticky="w", padx=4)
        ttk.Entry(interp_box, textvariable=self.new_w, width=6).grid(column=0, row=1, sticky="w", padx=4)
        ttk.Entry(interp_box, textvariable=self.new_h, width=6).grid(column=1, row=1, sticky="w", padx=4)
        ttk.Button(interp_box, text="Nearest", command=lambda: self.resize("nearest")).grid(column=0, row=2, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(interp_box, text="Bilinear", command=lambda: self.resize("bilinear")).grid(column=0, row=3, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(interp_box, text="Bicubic", command=lambda: self.resize("bicubic")).grid(column=0, row=4, columnspan=2, sticky="ew", padx=4, pady=2)

        # Crop
        crop_box = self._labeled_box(controls, "Crop")
        self.crop_x = tk.IntVar(value=10)
        self.crop_y = tk.IntVar(value=10)
        self.crop_w = tk.IntVar(value=200)
        self.crop_h = tk.IntVar(value=200)
        for idx, (lbl, var) in enumerate([("x", self.crop_x), ("y", self.crop_y), ("w", self.crop_w), ("h", self.crop_h)]):
            ttk.Label(crop_box, text=lbl).grid(column=idx % 2, row=idx // 2 * 2, sticky="w", padx=4)
            ttk.Entry(crop_box, textvariable=var, width=6).grid(column=idx % 2, row=idx // 2 * 2 + 1, sticky="w", padx=4)
        ttk.Button(crop_box, text="Crop", command=self.crop).grid(column=0, row=4, columnspan=2, sticky="ew", padx=4, pady=2)

        # Histogram
        hist_box = self._labeled_box(controls, "Histogram")
        ttk.Button(hist_box, text="Compute Histogram", command=self.show_histogram).grid(column=0, row=0, sticky="ew", padx=4, pady=2)
        ttk.Button(hist_box, text="Equalize", command=self.equalize_histogram).grid(column=0, row=1, sticky="ew", padx=4, pady=2)
        self.hist_eval_label = ttk.Label(hist_box, text="", wraplength=180)
        self.hist_eval_label.grid(column=0, row=2, sticky="w", padx=4)

        # Filters
        filter_box = self._labeled_box(controls, "Filtering")
        ttk.Button(filter_box, text="Gaussian 19x19 (σ=3)", command=self.gaussian).grid(column=0, row=0, sticky="ew", padx=4, pady=2)
        ttk.Button(filter_box, text="Median 7x7", command=self.median).grid(column=0, row=1, sticky="ew", padx=4, pady=2)
        ttk.Button(filter_box, text="Laplacian (2nd deriv.)", command=self.laplacian).grid(column=0, row=2, sticky="ew", padx=4, pady=2)
        ttk.Button(filter_box, text="Sobel", command=self.sobel).grid(column=0, row=3, sticky="ew", padx=4, pady=2)
        ttk.Button(filter_box, text="Gradient (1st deriv.)", command=self.gradient).grid(column=0, row=4, sticky="ew", padx=4, pady=2)

        # Compression
        comp_box = self._labeled_box(controls, "Compression")
        ttk.Button(comp_box, text="Huffman", command=lambda: self.compress_action("huffman")).grid(column=0, row=0, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Golomb-Rice", command=lambda: self.compress_action("golomb")).grid(column=0, row=1, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Arithmetic", command=lambda: self.compress_action("arithmetic")).grid(column=0, row=2, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="LZW", command=lambda: self.compress_action("lzw")).grid(column=0, row=3, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Run-Length", command=lambda: self.compress_action("rle")).grid(column=0, row=4, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Symbol-based", command=lambda: self.compress_action("symbol")).grid(column=0, row=5, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Bit-plane", command=lambda: self.compress_action("bitplane")).grid(column=0, row=6, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Block DCT", command=lambda: self.compress_action("dct")).grid(column=0, row=7, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Predictive", command=lambda: self.compress_action("predictive")).grid(column=0, row=8, sticky="ew", padx=4, pady=2)
        ttk.Button(comp_box, text="Wavelet (Haar)", command=lambda: self.compress_action("wavelet")).grid(column=0, row=9, sticky="ew", padx=4, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).grid(column=0, row=2, sticky="ew", pady=(6, 0))

    def _labeled_box(self, parent, title: str):
        box = ttk.LabelFrame(parent, text=title, padding=6)
        box.configure(style="TFrame")
        box.grid(column=0, row=parent.grid_size()[1], sticky="ew", pady=6)
        return box

    # ---------- Helpers ---------- #
    def _require_image(self):
        if self.original_img is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return False
        return True

    def _update_info(self):
        if self.original_img is None:
            self.info_label.config(text="No image loaded")
            return
        info = io_utils.info(self.original_img)
        text = f"{info['width']}x{info['height']}  |  channels: {info['channels']}  |  dtype: {info['dtype']}"
        self.info_label.config(text=text)

    def _set_processed(self, img, title="Processed"):
        self.processed_img = img
        self.preview.show_processed(img, title)

    # ---------- Actions ---------- #
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        try:
            self.original_img = io_utils.load_image(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))
            return
        self.processed_img = None
        self.preview.show_original(self.original_img)
        self.preview.show_processed(None)
        self._update_info()
        self.image_path_var.set(path.split("/")[-1])
        self.status_var.set("Loaded image.")

    def to_grayscale(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.original_img)
        self._set_processed(gray, "Grayscale")
        self.status_var.set("Converted to grayscale.")

    def to_binary(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.original_img)
        binary, threshold = basic_ops.grayscale_to_binary(gray)
        self.binary_threshold = threshold
        self._set_processed(binary, f"Binary (t={threshold:.1f})")
        self.status_var.set("Binary conversion complete.")

    def translate(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = geometry.translate(img, self.tx_var.get(), self.ty_var.get())
        self.preview.show_original(self.original_img)
        self._set_processed(result, "Translated")
        self.status_var.set("Applied translation.")

    def scale(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = geometry.scale(img, self.sx_var.get(), self.sy_var.get())
        self._set_processed(result, "Scaled")
        self.status_var.set("Applied scaling.")

    def rotate(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = geometry.rotate(img, self.angle_var.get())
        self._set_processed(result, "Rotated")
        self.status_var.set("Applied rotation.")

    def shear_x(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = geometry.shear_x(img, self.shx_var.get())
        self._set_processed(result, "Shear X")
        self.status_var.set("Applied X shear.")

    def shear_y(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = geometry.shear_y(img, self.shy_var.get())
        self._set_processed(result, "Shear Y")
        self.status_var.set("Applied Y shear.")

    def resize(self, method: str):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = interp.resize(img, self.new_w.get(), self.new_h.get(), method=method)
        self._set_processed(result, f"Resize ({method})")
        self.status_var.set(f"Resized using {method}.")

    def crop(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = basic_ops.crop(img, self.crop_x.get(), self.crop_y.get(), self.crop_w.get(), self.crop_h.get())
        self._set_processed(result, "Cropped")
        self.status_var.set("Cropped region.")

    def show_histogram(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.processed_img if self.processed_img is not None else self.original_img)
        hist = basic_ops.histogram(gray)
        self.preview.show_histogram(hist)
        self.hist_eval_label.config(text=basic_ops.histogram_goodness(hist))
        self.status_var.set("Histogram computed.")

    def equalize_histogram(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.processed_img if self.processed_img is not None else self.original_img)
        eq = basic_ops.histogram_equalization(gray)
        self._set_processed(eq, "Equalized")
        self.status_var.set("Histogram equalization applied.")

    def gaussian(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = filters.gaussian_blur(img, 19, 3.0)
        self._set_processed(result, "Gaussian Blur")
        self.status_var.set("Applied Gaussian filter.")

    def median(self):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        result = filters.median_filter(img, 7)
        self._set_processed(result, "Median 7x7")
        self.status_var.set("Applied median filter.")

    def laplacian(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.processed_img if self.processed_img is not None else self.original_img)
        result = filters.laplacian_filter(gray)
        self._set_processed(result, "Laplacian")
        self.status_var.set("Applied Laplacian filter.")

    def sobel(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.processed_img if self.processed_img is not None else self.original_img)
        result = filters.sobel_filter(gray)
        self._set_processed(result, "Sobel magnitude")
        self.status_var.set("Applied Sobel filter.")

    def gradient(self):
        if not self._require_image():
            return
        gray = basic_ops.rgb_to_grayscale(self.processed_img if self.processed_img is not None else self.original_img)
        result = filters.gradient_first_derivative(gray)
        self._set_processed(result, "Gradient")
        self.status_var.set("Applied first-derivative gradient.")

    def compress_action(self, mode: str):
        if not self._require_image():
            return
        img = self.processed_img if self.processed_img is not None else self.original_img
        gray = basic_ops.rgb_to_grayscale(img)
        if mode == "huffman":
            data = compress.huffman_compress(gray)
            decoded = compress.huffman_decompress(data["bitstring"], data["tree"], gray.shape)
            self._set_processed(decoded, f"Huffman (r={data['ratio']:.2f})")
            self.status_var.set(f"Huffman ratio {data['ratio']:.2f}")
        elif mode == "golomb":
            data = compress.golomb_rice_encode(gray, k=2)
            decoded = compress.golomb_rice_decode(data["bitstring"], gray.shape, k=2)
            self._set_processed(decoded, f"Golomb-Rice (r={data['ratio']:.2f})")
            self.status_var.set(f"Golomb-Rice ratio {data['ratio']:.2f}")
        elif mode == "arithmetic":
            data = compress.arithmetic_encode(gray)
            decoded = compress.arithmetic_decode(data["code"], data)
            self._set_processed(decoded, f"Arithmetic (r={data['ratio']:.2f})")
            self.status_var.set(f"Arithmetic ratio {data['ratio']:.2f}")
        elif mode == "lzw":
            data = compress.lzw_encode(gray)
            decoded = compress.lzw_decode(data["codes"], gray.shape)
            self._set_processed(decoded, f"LZW (r={data['ratio']:.2f})")
            self.status_var.set(f"LZW ratio {data['ratio']:.2f}")
        elif mode == "rle":
            data = compress.rle_encode(gray)
            decoded = compress.rle_decode(data["pairs"], gray.shape)
            self._set_processed(decoded, f"RLE (r={data['ratio']:.2f})")
            self.status_var.set(f"Run-length ratio {data['ratio']:.2f}")
        elif mode == "symbol":
            data = compress.symbol_based_encode(gray)
            decoded = compress.symbol_based_decode(data["bitstring"], data["codes"], gray.shape)
            self._set_processed(decoded, f"Symbol (r={data['ratio']:.2f})")
            self.status_var.set(f"Symbol-based ratio {data['ratio']:.2f}")
        elif mode == "bitplane":
            planes, reconstructed = compress.bit_planes(gray)
            # stack bit-planes into a grid for display
            grid = self._stack_bitplanes(planes)
            self.preview.show_original(self.original_img)
            self._set_processed(grid, "Bit-planes (stacked)")
            self.status_var.set("Bit-plane coding shown.")
        elif mode == "dct":
            data = compress.dct_compress(gray)
            self._set_processed(data["image"], f"DCT recon (r={data['ratio']:.2f})")
            self.status_var.set(f"DCT kept {data['kept_coefficients']}/{data['total_coefficients']}")
        elif mode == "predictive":
            data = compress.predictive_encode(gray)
            decoded = compress.predictive_decode(data["residual"])
            self._set_processed(decoded, f"Predictive (r={data['ratio']:.2f})")
            self.status_var.set(f"Predictive coding ratio {data['ratio']:.2f}")
        elif mode == "wavelet":
            approx, horiz, vert, diag = compress.haar_wavelet_transform(gray)
            recon = compress.haar_wavelet_inverse(approx, horiz, vert, diag)
            self._set_processed(recon, "Wavelet recon")
            self.status_var.set("Haar wavelet transform applied.")

    def _stack_bitplanes(self, planes):
        # Combine bit-planes into a square grid for visualization.
        rows = []
        for i in range(0, 8, 2):
            left = np.hstack((planes[i], planes[i + 1]))
            rows.append(left)
        return np.vstack(rows)
