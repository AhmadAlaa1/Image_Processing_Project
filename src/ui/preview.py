import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def _prep_image(img: np.ndarray):
    if img is None:
        return None
    if img.ndim == 2:
        return np.clip(img, 0, 255)
    return np.clip(img, 0, 255).astype(np.float32) / 255.0


class ImagePreview:
    def __init__(self, parent):
        self.figure = Figure(figsize=(6.5, 4), dpi=100)
        self.ax_original = self.figure.add_subplot(121)
        self.ax_processed = self.figure.add_subplot(122)
        self.ax_original.axis("off")
        self.ax_processed.axis("off")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        # Draw once so the canvas renders immediately on load.
        self.canvas.draw()

    def show_original(self, img: np.ndarray, title="Original"):
        self.ax_original.clear()
        self.ax_original.axis("off")
        img_prep = _prep_image(img)
        if img_prep is not None:
            if img_prep.ndim == 2:
                self.ax_original.imshow(img_prep, cmap="gray")
            else:
                self.ax_original.imshow(img_prep)
        self.ax_original.set_title(title)
        self.canvas.draw_idle()

    def show_processed(self, img: np.ndarray, title="Processed"):
        self.ax_processed.clear()
        self.ax_processed.axis("off")
        img_prep = _prep_image(img)
        if img_prep is not None:
            if img_prep.ndim == 2:
                self.ax_processed.imshow(img_prep, cmap="gray")
            else:
                self.ax_processed.imshow(img_prep)
        self.ax_processed.set_title(title)
        self.canvas.draw_idle()

    def show_histogram(self, hist: np.ndarray):
        self.ax_processed.clear()
        self.ax_processed.bar(range(256), hist, color="#7f8c8d")
        self.ax_processed.set_title("Histogram")
        self.ax_processed.set_xlim(0, 255)
        self.canvas.draw_idle()
