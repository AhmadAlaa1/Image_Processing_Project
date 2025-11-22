const originalImg = document.getElementById("original");
const processedImg = document.getElementById("processed");
const fileName = document.getElementById("fileName");
const statusEl = document.getElementById("status");
const histInfo = document.getElementById("histInfo");
const compInfo = document.getElementById("compInfo");
const infoBox = document.getElementById("info");
const histCanvas = document.getElementById("histCanvas");
let histCtx = histCanvas?.getContext("2d");
let imageDataUrl = null;

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

async function sendAction(action, params = {}) {
  if (!imageDataUrl) {
    setStatus("Please choose an image first.", true);
    return;
  }
  setStatus(`Processing: ${action}...`);
  histInfo.textContent = "";
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
        compInfo.textContent = `Compression ratio: ${data.extra.ratio.toFixed(2)}`;
      }
      if (data.extra.threshold !== undefined) {
        histInfo.textContent = `Binary threshold: ${data.extra.threshold.toFixed(2)}`;
      }
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
  fileName.textContent = file.name;
  const raw = await readFile(file);
  imageDataUrl = await downscaleIfLarge(raw);
  originalImg.src = imageDataUrl;
  processedImg.src = "";
  infoBox.textContent = "";
  histInfo.textContent = "";
  compInfo.textContent = "";
  setStatus("Image loaded.");
});

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
    }
    sendAction(action, params);
  });
});
