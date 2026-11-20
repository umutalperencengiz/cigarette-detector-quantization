/* Live FP32 vs INT8 comparison, entirely in the browser.
 *
 * Both graphs are the same YOLO11n checkpoint, exported from the same script.
 * The page loads ONNX Runtime Web once and keeps one session per precision, so
 * switching models costs nothing after the first download.
 */

const INPUT = 640;
const CLASS_NAME = "cigarette";

const VARIANTS = {
  fp32: {
    key: "fp32",
    label: "FP32",
    file: "models/yolo11n_fp32_640.onnx",
    bytes: 10.56 * 1e6,
  },
  int8: {
    key: "int8",
    label: "INT8 static",
    file: "models/yolo11n_int8_static_640.onnx",
    bytes: 4.44 * 1e6,
  },
};

const state = {
  variant: "int8",
  conf: 0.25,
  iou: 0.7,
  session: null,
  sessions: new Map(),
  loading: new Map(),
  source: null,
  sourceName: "sample frame",
  lastDetections: [],
  stream: null,
  raf: 0,
  liveMs: [],
};

const $ = (selector) => document.querySelector(selector);
const canvasIn = document.createElement("canvas");
canvasIn.width = INPUT;
canvasIn.height = INPUT;
const ctxIn = canvasIn.getContext("2d", { willReadFrequently: true });

/* ------------------------------------------------------------------ status */
function setStatus(text, tone = "") {
  const pill = $("#engineStatus");
  if (!pill) return;
  pill.className = `status-pill ${tone}`.trim();
  const label = $("#engineText");
  if (label) label.textContent = text;
}

function setChip(selector, text) {
  const node = $(selector);
  if (node) node.textContent = text;
}

function log(message, highlight = false) {
  const box = $("#runLog");
  if (!box) return;
  const line = document.createElement("div");
  line.innerHTML = highlight ? `<b>${message}</b>` : message;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

/* ------------------------------------------------------------------ runtime */
async function loadVariant(key) {
  if (state.sessions.has(key)) return state.sessions.get(key);
  if (state.loading.has(key)) return state.loading.get(key);

  const variant = VARIANTS[key];
  setStatus(`loading ${variant.label}`, "");

  const promise = (async () => {
    if (typeof ort === "undefined") {
      throw new Error("ONNX Runtime Web did not load — check the connection and reload.");
    }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

    const started = performance.now();
    const session = await ort.InferenceSession.create(variant.file, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    const seconds = (performance.now() - started) / 1000;
    state.sessions.set(key, session);
    const mb = (variant.bytes / 1e6).toFixed(1);
    setStatus(`${variant.label} ready`, "ok");
    setChip("#chipModel", `${variant.file.split("/").pop()} · ${mb} MB`);
    log(`loaded <b>${variant.file.split("/").pop()}</b> (${mb} MB) in ${seconds.toFixed(1)} s`, true);
    return session;
  })().catch((err) => {
    state.loading.delete(key);
    setStatus("model unavailable", "bad");
    log(`load failed: ${err.message}`);
    console.error(err);
    throw err;
  });

  state.loading.set(key, promise);
  return promise;
}

function activateVariant(key) {
  state.variant = key;
  state.session = state.sessions.get(key) || null;
  document.querySelectorAll("[data-variant]").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.variant === key));
  });
  const variant = VARIANTS[key];
  setChip("#chipModel", `${variant.file.split("/").pop()} · ${(variant.bytes / 1e6).toFixed(1)} MB`);
  if (state.session) setStatus(`${variant.label} ready`, "ok");
}

/* -------------------------------------------------------------- preprocess */
function preprocess(sw, sh) {
  const scale = Math.min(INPUT / sw, INPUT / sh);
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));
  const dx = Math.floor((INPUT - dw) / 2);
  const dy = Math.floor((INPUT - dh) / 2);

  ctxIn.fillStyle = "#727272";
  ctxIn.fillRect(0, 0, INPUT, INPUT);
  ctxIn.drawImage(state.source, 0, 0, sw, sh, dx, dy, dw, dh);

  const { data } = ctxIn.getImageData(0, 0, INPUT, INPUT);
  const plane = INPUT * INPUT;
  const tensor = new Float32Array(plane * 3);
  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    tensor[i] = data[p] / 255;
    tensor[plane + i] = data[p + 1] / 255;
    tensor[plane * 2 + i] = data[p + 2] / 255;
  }
  return { tensor, meta: { scale, dx, dy, sw, sh } };
}

function boxIou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (!inter) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

function decode(raw, meta, confThreshold, iouThreshold) {
  const anchors = raw.dims[2];
  const channels = raw.dims[1];
  const out = raw.data;
  const boxes = [];
  const scores = [];

  for (let a = 0; a < anchors; a += 1) {
    let best = 0;
    for (let c = 4; c < channels; c += 1) {
      const score = out[c * anchors + a];
      if (score > best) best = score;
    }
    if (best < confThreshold) continue;

    const cx = out[a];
    const cy = out[anchors + a];
    const w = out[anchors * 2 + a];
    const h = out[anchors * 3 + a];

    boxes.push([
      (cx - w / 2 - meta.dx) / meta.scale,
      (cy - h / 2 - meta.dy) / meta.scale,
      (cx + w / 2 - meta.dx) / meta.scale,
      (cy + h / 2 - meta.dy) / meta.scale,
    ]);
    scores.push(best);
  }

  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const kept = [];
  while (order.length) {
    const index = order.shift();
    kept.push(index);
    for (let k = order.length - 1; k >= 0; k -= 1) {
      if (boxIou(boxes[index], boxes[order[k]]) > iouThreshold) order.splice(k, 1);
    }
  }

  return kept.map((index) => {
    const [x1, y1, x2, y2] = boxes[index];
    return {
      label: CLASS_NAME,
      confidence: scores[index],
      box: [Math.max(0, x1), Math.max(0, y1), Math.min(meta.sw, x2), Math.min(meta.sh, y2)],
    };
  });
}

async function infer(key, sw, sh) {
  const session = state.sessions.get(key) || (await loadVariant(key));
  const { tensor, meta } = preprocess(sw, sh);
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, INPUT, INPUT]) };

  const started = performance.now();
  const output = await session.run(feeds);
  const ms = performance.now() - started;

  const detections = decode(output[session.outputNames[0]], meta, state.conf, state.iou);
  return { detections, ms };
}

/* ------------------------------------------------------------------ drawing */
function mediaElement() {
  return state.stream ? $("#video") : $("#stageImage");
}

function drawOverlay(detections, sw, sh, el = mediaElement()) {
  const overlay = $("#overlay");
  if (!overlay || !el) return;

  const rect = el.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.style.transform = `translate(${el.offsetLeft}px, ${el.offsetTop}px)`;

  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const sx = rect.width / sw;
  const sy = rect.height / sh;
  const fontPx = Math.max(11, Math.round(rect.width / 52));
  ctx.lineWidth = Math.max(1.5, rect.width / 420);
  ctx.font = `${fontPx}px ui-monospace, Consolas, monospace`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const X = x1 * sx;
    const Y = y1 * sy;
    const W = (x2 - x1) * sx;
    const H = (y2 - y1) * sy;
    const colour = det.confidence >= 0.5 ? "#35d0a5" : "#ffb454";

    ctx.strokeStyle = colour;
    ctx.strokeRect(X, Y, W, H);

    const text = `${det.label} ${det.confidence.toFixed(2)}`;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = colour;
    ctx.fillRect(X, Math.max(0, Y - fontPx - 6), tw + 10, fontPx + 6);
    ctx.fillStyle = "#04120e";
    ctx.fillText(text, X + 5, Math.max(0, Y - fontPx - 3));
  }
}

/* ------------------------------------------------------------------- frames */
function showSource(img, name) {
  state.source = img;
  state.sourceName = name;
  const stage = $("#stageImage");
  stage.src = img.src;
  stage.hidden = false;
  const msg = $("#stageMsg");
  if (msg) msg.hidden = true;
  stage.decode?.().then(() => drawOverlay(state.lastDetections, img.naturalWidth, img.naturalHeight));
}

function loadImage(url, name) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${url}`));
    img.src = url;
    img.dataset.name = name;
  });
}

async function runOnce(key) {
  const img = state.source;
  return infer(key, img.naturalWidth, img.naturalHeight);
}

async function runCurrent() {
  if (!state.source) return;
  if (state.stream) return; // the camera loop owns continuous inference
  const button = $("#runButton");
  button.disabled = true;
  try {
    const key = state.variant;
    const result = await runOnce(key);
    state.lastDetections = result.detections;
    drawOverlay(result.detections, state.source.naturalWidth, state.source.naturalHeight);
    updateStats(result.ms, result.detections, key);
    log(`${VARIANTS[key].label}: ${result.detections.length} det · ${result.ms.toFixed(1)} ms`);
  } catch (err) {
    log(`error: ${err.message}`);
    console.error(err);
  } finally {
    button.disabled = false;
  }
}

function updateStats(ms, detections, key) {
  setChip("#statLatency", `${ms.toFixed(1)} ms`);
  setChip("#statDetections", String(detections.length));
  setChip("#statSize", `${(VARIANTS[key].bytes / 1e6).toFixed(1)} MB`);
  const peak = detections.reduce((max, d) => Math.max(max, d.confidence), 0);
  setChip("#statPeak", peak ? peak.toFixed(2) : "—");
}

/** Run both graphs over the same frame, several times, and report the gap. */
async function runComparison() {
  if (!state.source) return;
  const button = $("#compareButton");
  button.disabled = true;
  const repeats = 5;

  try {
    const sw = state.source.naturalWidth;
    const sh = state.source.naturalHeight;
    const rows = {};

    for (const key of ["fp32", "int8"]) {
      await loadVariant(key);
      await infer(key, sw, sh); // warm-up
      const times = [];
      let detections = [];
      for (let i = 0; i < repeats; i += 1) {
        const run = await infer(key, sw, sh);
        times.push(run.ms);
        detections = run.detections;
      }
      times.sort((a, b) => a - b);
      rows[key] = {
        ms: times[Math.floor(times.length / 2)],
        mean: times.reduce((a, b) => a + b, 0) / times.length,
        detections,
      };
    }

    const fp32 = rows.fp32;
    const int8 = rows.int8;
    const speedup = fp32.mean / int8.mean;
    const matched = int8.detections.filter((det) =>
      fp32.detections.some((other) => boxIou(det.box, other.box) > 0.5)
    ).length;
    const same = fp32.detections.length === int8.detections.length && matched === fp32.detections.length;

    setChip("#compareFp32", `${fp32.ms.toFixed(1)} ms`);
    setChip("#compareInt8", `${int8.ms.toFixed(1)} ms`);
    setChip("#compareSpeedup", `${speedup.toFixed(2)}×`);
    setChip(
      "#compareAgree",
      same ? `identical (${matched}/${fp32.detections.length})` : `${matched}/${fp32.detections.length} matched`
    );

    const box = $("#compareResult");
    if (box) box.hidden = false;
    log(
      `A/B: FP32 ${fp32.ms.toFixed(1)} ms · INT8 ${int8.ms.toFixed(1)} ms · ${speedup.toFixed(2)}× · ${matched}/${fp32.detections.length} boxes agree`,
      true
    );

    state.lastDetections = rows[state.variant].detections;
    drawOverlay(state.lastDetections, sw, sh);
  } catch (err) {
    log(`comparison failed: ${err.message}`);
    console.error(err);
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------- webcam */
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
      audio: false,
    });
    state.stream = stream;
    const video = $("#video");
    video.srcObject = stream;
    video.hidden = false;
    $("#stageImage").hidden = true;
    const msg = $("#stageMsg");
    if (msg) msg.hidden = true;
    await video.play();
    state.liveMs = [];
    $("#cameraToggle").textContent = "Stop camera";
    log("camera started · running INT8 on every frame", true);
    await loadVariant("int8");
    loopCamera();
  } catch (err) {
    log(`camera failed: ${err.message}`);
    const msg = $("#stageMsg");
    if (msg) {
      msg.hidden = false;
      msg.innerHTML = `Camera unavailable: <code>${err.message}</code>`;
    }
  }
}

function stopCamera() {
  cancelAnimationFrame(state.raf);
  state.raf = 0;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  const video = $("#video");
  video.srcObject = null;
  video.hidden = true;
  $("#cameraToggle").textContent = "Start camera";
  if (state.source) {
    $("#stageImage").hidden = false;
    drawOverlay(state.lastDetections, state.source.naturalWidth, state.source.naturalHeight);
  }
  log("camera stopped");
}

async function loopCamera() {
  const video = $("#video");
  if (!state.stream) return;

  if (video.videoWidth && state.sessions.has(state.variant)) {
    state.source = video;
    const result = await infer(state.variant, video.videoWidth, video.videoHeight);
    state.lastDetections = result.detections;
    state.liveMs.push(result.ms);
    if (state.liveMs.length > 30) state.liveMs.shift();
    const mean = state.liveMs.reduce((a, b) => a + b, 0) / state.liveMs.length;
    updateStats(mean, result.detections, state.variant);
    setChip("#statLatency", `${mean.toFixed(0)} ms`);
    drawOverlay(result.detections, video.videoWidth, video.videoHeight);
  }

  state.raf = requestAnimationFrame(loopCamera);
}

/* -------------------------------------------------------------------- wiring */
async function boot() {
  document.querySelectorAll("[data-variant]").forEach((btn) => {
    btn.addEventListener("click", () => activateVariant(btn.dataset.variant));
  });

  document.querySelectorAll("[data-frame]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll("[data-frame]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      const img = await loadImage(btn.dataset.frame, btn.dataset.name || "sample frame");
      showSource(img, img.dataset.name);
      state.lastDetections = [];
      $("#compareResult")?.setAttribute("hidden", "");
      await runCurrent();
    });
  });

  $("#runButton")?.addEventListener("click", runCurrent);
  $("#compareButton")?.addEventListener("click", runComparison);
  $("#cameraToggle")?.addEventListener("click", () => (state.stream ? stopCamera() : startCamera()));

  const fileInput = $("#customInput");
  fileInput?.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = await loadImage(url, file.name);
    showSource(img, file.name);
    state.lastDetections = [];
    await runCurrent();
  });

  $("#confSlider")?.addEventListener("input", (event) => {
    state.conf = Number(event.target.value);
    $("#confValue").textContent = state.conf.toFixed(2);
    if (state.lastDetections.length) runCurrent();
  });

  window.addEventListener("resize", () => {
    if (state.source && !state.stream) {
      drawOverlay(state.lastDetections, state.source.naturalWidth, state.source.naturalHeight);
    }
  });

  const first = document.querySelector("[data-frame][aria-pressed='true']");
  const runBtn = $("#runButton");
  if (runBtn) runBtn.disabled = false;
  const compareBtn = $("#compareButton");
  if (compareBtn) compareBtn.disabled = false;

  try {
    await loadVariant("int8");
    state.session = state.sessions.get("int8");
    activateVariant("int8");
    if (first) {
      const img = await loadImage(first.dataset.frame, first.dataset.name || "sample frame");
      showSource(img, img.dataset.name);
      await runCurrent();
    }
  } catch (err) {
    console.error(err);
  }
}

boot();
