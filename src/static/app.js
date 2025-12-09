const stageImage = document.getElementById("stageImage");
const originalImg = document.getElementById("original") || new Image(); // legacy ids not used in layout
const processedImg = document.getElementById("processed") || new Image();
const fileName = document.getElementById("fileName");
const statusEl = document.getElementById("status");
const histInfo = document.getElementById("histInfo");
const compInfo = document.getElementById("compInfo");
const infoBox = document.getElementById("info");
const fileMeta = document.getElementById("fileMeta");
const histCanvas = document.getElementById("histCanvas");
let histCtx = histCanvas?.getContext("2d");
let imageDataUrl = null;
let lastMode = "processed"; // or "original"
let originalInfo = null; // { width, height, size, type }

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#f87171" : "#10b981";
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function getImageDimensions(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.width, height: img.height });
    img.onerror = () => reject(new Error("Unable to read image dimensions."));
    img.src = dataUrl;
  });
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "Unknown";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** power;
  const rounded = value >= 10 ? value.toFixed(0) : value.toFixed(1);
  return `${rounded} ${units[power]}`;
}

function formatBits(bits) {
  if (!Number.isFinite(bits)) return "—";
  const bytes = bits / 8;
  return `${bits} bits (${formatBytes(bytes)})`;
}

function resolveFileType(file) {
  if (file.type) return file.type;
  const parts = file.name.split(".");
  if (parts.length > 1) return parts.pop().toUpperCase();
  return "Unknown";
}

function setFileMeta(meta, compressedBits) {
  const { width, height, size, type } = meta || {};
  if (!fileMeta) return;
  if (!width || !height) {
    fileMeta.textContent = "Resolution: — | Size: — | Type: —";
    return;
  }
  const base = `Resolution: ${width}x${height} | Size: ${formatBytes(size)} | Type: ${type}`;
  if (compressedBits !== undefined) {
    fileMeta.textContent = `${base} | Compressed: ${formatBytes(compressedBits / 8)}`;
  } else {
    fileMeta.textContent = base;
  }
}

function downscaleIfLarge(dataUrl, maxDim = 1400) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const { width, height } = img;
      if (Math.max(width, height) <= maxDim) {
        resolve(dataUrl);
        return;
      }
      const ratio = maxDim / Math.max(width, height);
      const w = Math.round(width * ratio);
      const h = Math.round(height * ratio);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      resolve(canvas.toDataURL("image/png"));
    };
    img.src = dataUrl;
  });
}

function renderHistogram(histArr) {
  if (!histCtx || !histArr) return;
  histCtx.clearRect(0, 0, histCanvas.width, histCanvas.height);
  const maxVal = Math.max(...histArr);
  if (maxVal === 0) return;
  for (let x = 0; x < 256; x++) {
    const h = (histArr[x] / maxVal) * histCanvas.height;
    histCtx.fillStyle = "#60a5fa";
    histCtx.fillRect(x, histCanvas.height - h, 1, h);
  }
}

function clearHistogram() {
  if (histCtx) histCtx.clearRect(0, 0, histCanvas.width, histCanvas.height);
  histInfo.textContent = "";
}

function resetToOriginal() {
  if (!originalImg.src) {
    setStatus("Load an image first.", true);
    return;
  }
  stageImage.src = originalImg.src;
  processedImg.src = "";
  lastMode = "original";
  showOriginalBtn?.classList.add("active");
  showProcessedBtn?.classList.remove("active");
  clearHistogram();
  compInfo.textContent = "";
  infoBox.textContent = "";
  setStatus("Reset to original image.");
}

async function sendAction(action, params = {}) {
  if (!imageDataUrl) {
    setStatus("Please choose an image first.", true);
    return;
  }
  setStatus(`Processing: ${action}...`);
  clearHistogram();
  compInfo.textContent = "";
  try {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl, action, params }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");
    if (data.image) {
      processedImg.src = data.image;
      stageImage.src = data.image;
      lastMode = "processed";
      showProcessedBtn?.classList.add("active");
      showOriginalBtn?.classList.remove("active");
    }
    if (data.info) {
      const i = data.info;
      infoBox.textContent = `${i.width}x${i.height}, channels: ${i.channels}, dtype: ${i.dtype}`;
    }
    if (data.extra) {
      if (data.extra.histogram) {
        renderHistogram(data.extra.histogram);
        histInfo.textContent = `Assessment: ${data.extra.assessment}\nHistogram[0..5]: ${data.extra.histogram.slice(0,6)} ...`;
      }
      if (data.extra.ratio !== undefined) {
        const ob = data.extra.original_bits;
        const cb = data.extra.compressed_bits;
        const hasSizes = ob !== undefined && cb !== undefined;
        const sizeLine = hasSizes ? ` (${formatBits(ob)} -> ${formatBits(cb)})` : "";
        const downscaled = data.extra.compression_downscaled_from;
        const downscaleNote = downscaled ? ` | Downscaled from ${downscaled[0]}x${downscaled[1]} for compression` : "";
        compInfo.textContent = `Compression ratio: ${data.extra.ratio.toFixed(2)}${sizeLine}${downscaleNote}`;
      }
      if (data.extra.threshold !== undefined) {
        const note = data.extra.threshold_eval ? `\nNote: ${data.extra.threshold_eval}` : "";
        histInfo.textContent = `Binary threshold: ${data.extra.threshold.toFixed(2)}${note}`;
      }
    }
    if (originalInfo) {
      const cbits = data.extra ? data.extra.compressed_bits : undefined;
      setFileMeta(originalInfo, cbits);
    }
    setStatus("Done.");
  } catch (err) {
    console.error(err);
    setStatus(err.message, true);
  }
}

document.getElementById("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    fileName.textContent = file.name;
    const raw = await readFile(file);
    const { width, height } = await getImageDimensions(raw);
    imageDataUrl = await downscaleIfLarge(raw);
    originalInfo = { width, height, size: file.size, type: resolveFileType(file) };
    originalImg.src = imageDataUrl;
    processedImg.src = "";
    stageImage.src = imageDataUrl;
    lastMode = "original";
    showOriginalBtn?.classList.add("active");
    showProcessedBtn?.classList.remove("active");
    setFileMeta(originalInfo);
    infoBox.textContent = "";
    clearHistogram();
    compInfo.textContent = "";
    setStatus("Image loaded.");
  } catch (err) {
    console.error(err);
    setFileMeta({}); // reset meta display
    setStatus("Failed to load image.", true);
  }
});

document.getElementById("downloadBtn").addEventListener("click", () => {
  const src = stageImage.src || processedImg.src || originalImg.src;
  if (!src) {
    setStatus("Nothing to download. Load or process an image first.", true);
    return;
  }
  const link = document.createElement("a");
  link.href = src;
  link.download = "processed.png";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

const showOriginalBtn = document.getElementById("showOriginalBtn");
const showProcessedBtn = document.getElementById("showProcessedBtn");
if (showOriginalBtn && showProcessedBtn) {
  showOriginalBtn.addEventListener("click", () => {
    if (originalImg.src) {
      stageImage.src = originalImg.src;
      lastMode = "original";
      showOriginalBtn.classList.add("active");
      showProcessedBtn.classList.remove("active");
    }
  });
  showProcessedBtn.addEventListener("click", () => {
    if (processedImg.src) {
      stageImage.src = processedImg.src;
      lastMode = "processed";
      showProcessedBtn.classList.add("active");
      showOriginalBtn.classList.remove("active");
    }
  });
}

const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", resetToOriginal);
}

// Wire buttons
document.querySelectorAll("button[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const action = btn.dataset.action;
    const params = {};
    if (action === "translate") {
      params.tx = parseFloat(document.getElementById("tx").value);
      params.ty = parseFloat(document.getElementById("ty").value);
    } else if (action === "scale") {
      params.sx = parseFloat(document.getElementById("sx").value);
      params.sy = parseFloat(document.getElementById("sy").value);
    } else if (action === "rotate") {
      params.angle = parseFloat(document.getElementById("angle").value);
    } else if (action === "shear_x" || action === "shear_y") {
      params.shx = parseFloat(document.getElementById("shx").value);
      params.shy = parseFloat(document.getElementById("shy").value);
    } else if (action.startsWith("resize_")) {
      params.width = parseInt(document.getElementById("newW").value, 10);
      params.height = parseInt(document.getElementById("newH").value, 10);
    } else if (action === "crop") {
      params.x = parseInt(document.getElementById("cropX").value, 10);
      params.y = parseInt(document.getElementById("cropY").value, 10);
      params.w = parseInt(document.getElementById("cropW").value, 10);
      params.h = parseInt(document.getElementById("cropH").value, 10);
    } else if (action === "huffman" || action === "golomb" || action === "arithmetic" || action === "lzw" || action === "rle" || action === "symbol" || action === "bitplane" || action === "dct" || action === "predictive" || action === "wavelet") {
      params.full_res = document.getElementById("fullResToggle")?.checked;
    }
    sendAction(action, params);
  });
});
