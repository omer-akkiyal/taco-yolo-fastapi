// frontend/app.js

const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const previewImg = document.getElementById("previewImg");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const confRange = document.getElementById("confRange");
const confLabel = document.getElementById("confLabel");

const countEl = document.getElementById("count");
const latencyEl = document.getElementById("latency");
const listEl = document.getElementById("list");
const jsonEl = document.getElementById("json");
const healthPill = document.getElementById("healthPill");

let selectedFile = null;
let lastDetections = [];


const API_BASE =
  (window.location.origin && window.location.origin !== "null")
    ? window.location.origin
    : "http://127.0.0.1:8000";

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, m => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[m]));
}

async function fetchWithTimeout(url, options = {}, ms = 60000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

function setBusy(isBusy) {
  if (isBusy) {
    predictBtn.disabled = true;
    predictBtn.textContent = "Tahmin ediliyor…";
  } else {
    predictBtn.disabled = !selectedFile;
    predictBtn.textContent = "Tahmin Et";
  }
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function fitCanvasToImage() {
  const wrap = previewImg.parentElement;
  const rect = wrap.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
}

function renderList(detections) {
  if (!detections.length) {
    listEl.innerHTML = `<div class="item">Sonuç yok. <span class="badge">conf’u düşürmeyi deneyin</span></div>`;
    return;
  }

  const rows = detections
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
    .slice(0, 50)
    .map(d => {
      const name = escapeHtml(String(d.class_name ?? d.class_id));
      const conf = (d.confidence ?? 0).toFixed(2);
      return `<div class="item"><b>${name}</b> <span class="badge">${conf}</span></div>`;
    });

  listEl.innerHTML = rows.join("");
}

function drawBoxes(detections) {
  if (!previewImg.naturalWidth || !previewImg.naturalHeight) return;
  clearCanvas();

  const imgW = previewImg.naturalWidth;
  const imgH = previewImg.naturalHeight;

  const cw = canvas.width;
  const ch = canvas.height;

  const scale = Math.min(cw / imgW, ch / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;

  const offsetX = (cw - drawW) / 2;
  const offsetY = (ch - drawH) / 2;

  ctx.drawImage(previewImg, offsetX, offsetY, drawW, drawH);

  ctx.save();
  ctx.lineWidth = 2;
  ctx.font = "12px ui-sans-serif, system-ui";
  ctx.textBaseline = "top";

  const uiConf = Number(confRange.value);

  for (const det of detections) {
    const c = det.confidence ?? 0;
    if (c < uiConf) continue;

    const [x1, y1, x2, y2] = det.box_xyxy;
    const sx1 = offsetX + x1 * scale;
    const sy1 = offsetY + y1 * scale;
    const sx2 = offsetX + x2 * scale;
    const sy2 = offsetY + y2 * scale;

    const w = sx2 - sx1;
    const h = sy2 - sy1;

    ctx.strokeStyle = "rgba(110,231,255,.95)";
    ctx.strokeRect(sx1, sy1, w, h);

    const label = `${det.class_name ?? det.class_id} ${c.toFixed(2)}`;
    const pad = 4;
    const tw = ctx.measureText(label).width;

    ctx.fillStyle = "rgba(0,0,0,.55)";
    ctx.fillRect(sx1, sy1, tw + pad * 2, 16 + pad);

    ctx.fillStyle = "rgba(234,240,255,.95)";
    ctx.fillText(label, sx1 + pad, sy1 + 2);
  }

  ctx.restore();
}

async function checkHealth() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/health`, {}, 8000);
    if (!res.ok) throw new Error("health not ok");
    const data = await res.json();
    healthPill.textContent = `API: hazır • ${data.model ?? ""}`;
    healthPill.style.borderColor = "rgba(110,231,255,.35)";
  } catch {
    healthPill.textContent = "API: erişilemiyor";
    healthPill.style.borderColor = "rgba(255,120,120,.35)";
  }
}

confRange.addEventListener("input", () => {
  confLabel.textContent = Number(confRange.value).toFixed(2);
  if (lastDetections.length) drawBoxes(lastDetections);
});

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  selectedFile = f ?? null;
  predictBtn.disabled = !selectedFile;

  countEl.textContent = "—";
  latencyEl.textContent = "—";
  listEl.innerHTML = "";
  jsonEl.textContent = "";
  lastDetections = [];

  if (!selectedFile) {
    previewImg.style.display = "none";
    clearCanvas();
    return;
  }

  const url = URL.createObjectURL(selectedFile);
  previewImg.onload = () => {
    previewImg.style.display = "block";
    fitCanvasToImage();
    clearCanvas();
    drawBoxes([]); 
  };
  previewImg.src = url;
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setBusy(true);
  listEl.innerHTML = "";
  jsonEl.textContent = "";
  countEl.textContent = "—";
  latencyEl.textContent = "—";
  lastDetections = [];
  clearCanvas();
  drawBoxes([]); 

  const form = new FormData();
  form.append("file", selectedFile);

  const conf = Number(confRange.value).toFixed(2);
  const t0 = performance.now();

  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/predict?conf=${conf}`,
      { method: "POST", body: form },
      60000
    );

    const raw = await res.text();
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${raw}`);
    }

    const data = JSON.parse(raw);
    const t1 = performance.now();

    latencyEl.textContent = `${Math.round(t1 - t0)} ms`;
    countEl.textContent = data.count ?? 0;

    lastDetections = data.detections ?? [];
    renderList(lastDetections);
    jsonEl.textContent = JSON.stringify(data, null, 2);

    fitCanvasToImage();
    drawBoxes(lastDetections);
  } catch (e) {
    const msg = (e && e.name === "AbortError")
      ? "Timeout: Model çok uzun sürdü. conf'u düşürün veya daha küçük görsel deneyin."
      : String(e.message ?? e);

    listEl.innerHTML = `<div class="item">❌ Hata: <b>${escapeHtml(msg)}</b></div>`;
  } finally {
    setBusy(false);
  }
});

window.addEventListener("resize", () => {
  if (previewImg.style.display === "block") {
    fitCanvasToImage();
    drawBoxes(lastDetections);
  }
});


checkHealth();
