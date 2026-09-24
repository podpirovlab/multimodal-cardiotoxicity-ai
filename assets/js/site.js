/* CardioOncoPredict — page logic: upload/synthesise a recording, run the TWA analysis, draw the charts.
   All numbers are computed by CardioDSP (assets/js/dsp.js) and CardioEDF (assets/js/edf.js). Nothing is hard-coded. */
(function () {
  "use strict";
  const D = window.CardioDSP;
  const EDF = window.CardioEDF;
  const LANG = (document.documentElement.lang || "en").slice(0, 2);
  const T = {
    ru: {
      pos: "TWA обнаружена", neg: "TWA не обнаружена", err: "Не удалось проанализировать",
      posText: (r) => `Альтернация ${fmt(r.vAlt, 1)} мкВ (RMS по окну ST-T) при K = ${fmt(r.k, 1)}: выполнен критерий V_alt ≥ 1,9 мкВ и K ≥ 3. На бумажной ЭКГ это ${fmt(r.vPeak / 100, 2)} мм — глазом не увидеть.`,
      negText: (r) => r.vAlt >= 1.9 ? `Альтернация ${fmt(r.vAlt, 1)} мкВ, но K = ${fmt(r.k, 1)} < 3: пик на 0,5 цикла/удар не выделяется из шума. Уменьшите шум или добавьте ударов.`
                                    : `V_alt = ${fmt(r.vAlt, 1)} мкВ < 1,9 мкВ, K = ${fmt(r.k, 1)}. Чередования зубца T не выявлено.`,
      hr: "ЧСС", valt: "V_alt", vpeak: "Пик", k: "K-score", mma: "MMA", noise: "Шум",
      bpm: "уд/мин", uv: "мкВ", beats: (n) => `по ${n} ударам`,
      dValt: "RMS по окну ST-T", dPeak: "в самой «качающейся» точке", dK: "сигнал / шум", dMma: "метод скольз. среднего", dNoise: "полоса 0,44–0,49",
      paper: "на бумаге", mm: "мм", even: "чётные удары (A)", odd: "нечётные удары (B)", diff: "разность A−B ×10",
      series: "значение в точке ST-T от удара к удару", band: "полоса шума", alt: "0,5 цикла/удар",
      stt: "окно ST-T", rp: "R-пики", cpb: "циклы / удар", beat: "номер удара", fromR: "время от R-пика, с",
      mv: "мВ", paperScale: "25 мм/с · 10 мм/мВ",
      fileErr: "В файле не найдено чисел. Нужен CSV/TXT: один столбец значений в мВ (или мкВ).",
      fileErrEdf: (msg) => `Не удалось прочитать EDF: ${msg}`,
      fileOk: (n, fs) => `Загружено ${n} отсчётов, ${fs} Гц`,
      edfOk: (n, fs, label) => `EDF: канал «${label}», ${n} отсчётов, ${fs} Гц`,
      edfChannel: "Канал (отведение)", edfAnnotations: "(служебный канал, пропущен)",
      shortWarn: (n) => `Найдено всего ${n} ударов. Спектральному методу нужно около 128 (~2 минуты записи) для надёжного результата — при меньшей длине запись всё равно анализируется, но осторожнее с выводами.`,
      pAge: "Возраст, лет", pSex: "Пол", pSexU: "не указан", pSexM: "мужской", pSexF: "женский",
      pDose: "Кумулятивная доза доксорубицина, мг/м²", pDoseHint: "необязательно — только для контекста в отчёте, не входит в расчёт TWA",
      doseCtx: (pct, dose) => `При дозе ${fmt(dose, 0)} мг/м² в популяции сердечная недостаточность развивается примерно у ${fmt(pct, 0)} % пациентов (Swain et al., Cancer, 2003). Это статистика по группе, не прогноз для конкретного человека.`,
      fhirBtn: "Экспортировать FHIR-отчёт", fhirNote: "Скачивает JSON (FHIR DiagnosticReport) с этими числами и указанным контекстом пациента — на устройство, никуда не отправляется.",
    },
    en: {
      pos: "TWA detected", neg: "No TWA", err: "Could not analyse",
      posText: (r) => `Alternans ${fmt(r.vAlt, 1)} µV (RMS over ST-T) with K = ${fmt(r.k, 1)}: meets V_alt ≥ 1.9 µV and K ≥ 3. On paper ECG that is ${fmt(r.vPeak / 100, 2)} mm, invisible to the eye.`,
      negText: (r) => r.vAlt >= 1.9 ? `Alternans ${fmt(r.vAlt, 1)} µV, but K = ${fmt(r.k, 1)} < 3: the 0.5 cycles/beat peak does not rise above noise. Lower the noise or add beats.`
                                    : `V_alt = ${fmt(r.vAlt, 1)} µV < 1.9 µV, K = ${fmt(r.k, 1)}. No beat-to-beat T-wave alternation found.`,
      hr: "Heart rate", valt: "V_alt", vpeak: "Peak", k: "K-score", mma: "MMA", noise: "Noise",
      bpm: "bpm", uv: "µV", beats: (n) => `from ${n} beats`,
      dValt: "RMS over ST-T", dPeak: "at the most alternating point", dK: "signal / noise", dMma: "modified moving average", dNoise: "band 0.44–0.49",
      paper: "on paper", mm: "mm", even: "even beats (A)", odd: "odd beats (B)", diff: "A−B difference ×10",
      series: "ST-T value at one point, beat after beat", band: "noise band", alt: "0.5 cycles/beat",
      stt: "ST-T window", rp: "R peaks", cpb: "cycles / beat", beat: "beat number", fromR: "time from R peak, s",
      mv: "mV", paperScale: "25 mm/s · 10 mm/mV",
      fileErr: "No numbers found. Use CSV/TXT with one column of values in mV (or µV).",
      fileErrEdf: (msg) => `Could not read the EDF file: ${msg}`,
      fileOk: (n, fs) => `Loaded ${n} samples at ${fs} Hz`,
      edfOk: (n, fs, label) => `EDF: channel "${label}", ${n} samples at ${fs} Hz`,
      edfChannel: "Channel (lead)", edfAnnotations: "(service channel, skipped)",
      shortWarn: (n) => `Only ${n} beats found. The Spectral Method wants about 128 (~2 minutes) for a reliable read — it still runs on shorter recordings, but treat the result with more caution.`,
      pAge: "Age, years", pSex: "Sex", pSexU: "unspecified", pSexM: "male", pSexF: "female",
      pDose: "Cumulative doxorubicin dose, mg/m²", pDoseHint: "optional — for report context only, not used in the TWA math above",
      doseCtx: (pct, dose) => `At ${fmt(dose, 0)} mg/m², population heart-failure incidence is about ${fmt(pct, 0)}% (Swain et al., Cancer, 2003) — a group statistic, not a prediction for this person.`,
      fhirBtn: "Export FHIR report", fhirNote: "Downloads a JSON file (FHIR DiagnosticReport) with these numbers and the patient context above — saved to your device, sent nowhere.",
    },
  }[LANG === "ru" ? "ru" : "en"];

  function fmt(v, d) {
    if (!isFinite(v)) return "—";
    const s = v.toFixed(d);
    return LANG === "ru" ? s.replace(".", ",") : s;
  }
  const $ = (s) => document.querySelector(s);

  // ---------- theme-aware colours (read once from CSS custom properties) ----------
  let C = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["paper", "surface", "ink", "muted", "line", "grid-minor", "grid-major", "trace", "accent", "blue", "accent-soft", "blue-soft"])
      C[k] = cs.getPropertyValue("--" + k).trim();
  }
  readColors();

  function setupCanvas(cv, cssH) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth, h = cssH || cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  const MONO = "500 11px 'JetBrains Mono', ui-monospace, monospace";

  // ECG-paper grid: minor every 1 mm, major every 5 mm
  function paperGrid(ctx, w, h, pxmm, x0 = 0) {
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, h);
    for (let pass = 0; pass < 2; pass++) {
      ctx.strokeStyle = pass ? C["grid-major"] : C["grid-minor"];
      ctx.lineWidth = pass ? 1 : 0.6;
      ctx.beginPath();
      const step = pass ? 5 * pxmm : pxmm;
      if (!pass && pxmm < 3) continue;
      for (let x = x0; x <= w; x += step) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); }
      for (let y = h; y >= 0; y -= step) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(w, Math.round(y) + 0.5); }
      ctx.stroke();
    }
  }

  // generic axes for the analysis panels
  function axes(ctx, w, h, pad, xr, yr, opts) {
    const X = (v) => pad.l + (v - xr[0]) / (xr[1] - xr[0]) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - (v - yr[0]) / (yr[1] - yr[0]) * (h - pad.t - pad.b);
    ctx.fillStyle = C.surface; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.font = MONO; ctx.fillStyle = C.muted;
    ctx.beginPath();
    for (const t of opts.xt || []) { const x = Math.round(X(t)) + 0.5; ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); }
    for (const t of opts.yt || []) { const y = Math.round(Y(t)) + 0.5; ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); }
    ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of opts.xt || []) ctx.fillText(opts.xf ? opts.xf(t) : t, X(t), h - pad.b + 5);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of opts.yt || []) ctx.fillText(opts.yf ? opts.yf(t) : t, pad.l - 6, Y(t));
    if (opts.xl) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(opts.xl, w - pad.r, h - 2); }
    if (opts.yl) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(opts.yl, pad.l + 4, 2); }
    return { X, Y };
  }
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  // ======================= PATIENT CONTEXT (age / sex / dose) =======================
  // Reference points: heart-failure incidence by cumulative doxorubicin dose (Swain et al., Cancer 2003).
  const DOSE_POINTS = [[0, 0], [400, 5], [550, 26], [700, 48], [1000, 48]];
  function doseIncidence(dose) {
    for (let i = 1; i < DOSE_POINTS.length; i++) {
      const [d0, p0] = DOSE_POINTS[i - 1], [d1, p1] = DOSE_POINTS[i];
      if (dose <= d1) return p0 + (p1 - p0) * (dose - d0) / (d1 - d0);
    }
    return DOSE_POINTS[DOSE_POINTS.length - 1][1];
  }
  function patient() {
    const age = parseFloat($("#p-age").value);
    const sex = $("#p-sex").value;
    const dose = parseFloat($("#p-dose").value);
    return { age: isFinite(age) ? age : null, sex: sex || null, dose: isFinite(dose) && dose > 0 ? dose : null };
  }
  function updatePatientContext() {
    const p = patient(), out = $("#p-context");
    if (!out) return;
    out.textContent = p.dose ? T.doseCtx(clamp01(doseIncidence(p.dose) / 100) * 100, p.dose) : "";
  }

  // ======================= TWA LAB =======================
  const lab = { res: null, sig: null, seed: 7, source: "synth", fileSig: null, fileFs: 500, fileName: "", edf: null, csvRawText: null };
  const sliders = ["alt", "noise", "hr", "tamp", "mains"];
  function params() {
    const v = (id) => parseFloat($("#s-" + id).value);
    return { altUv: v("alt"), noiseUv: v("noise"), hr: v("hr"), tAmp: v("tamp"), mainsUv: v("mains") };
  }
  function syncOutputs() {
    const p = params();
    $("#o-alt").textContent = `${fmt(p.altUv, 1)} ${T.uv}`;
    $("#o-noise").textContent = `${fmt(p.noiseUv, 0)} ${T.uv}`;
    $("#o-hr").textContent = `${fmt(p.hr, 0)} ${T.bpm}`;
    $("#o-tamp").textContent = `${fmt(p.tAmp, 2)} ${LANG === "ru" ? "мВ" : "mV"}`;
    $("#o-mains").textContent = `${fmt(p.mainsUv, 0)} ${T.uv}`;
  }
  let pending = null;
  function schedule() { syncOutputs(); clearTimeout(pending); pending = setTimeout(runLab, 90); }

  function runLab() {
    let x, fs;
    if (lab.source === "file" && lab.fileSig) { x = lab.fileSig; fs = lab.fileFs; }
    else { const s = D.synth(Object.assign(params(), { seed: lab.seed })); x = s.x; fs = s.fs; }
    lab.sig = { x, fs };
    const box = $("#verdict");
    try {
      const r = D.analyzeTWA(x, fs);
      lab.res = r;
      box.className = "verdict " + (r.positive ? "pos" : "neg");
      box.querySelector(".pill").textContent = r.positive ? T.pos : T.neg;
      let msg = (lab.source === "file" ? lab.fileName + ". " : "") + (r.positive ? T.posText(r) : T.negText(r));
      if (r.nBeats < 100) msg += " " + T.shortWarn(r.nBeats);
      box.querySelector("p").textContent = msg;
      $("#m-hr").innerHTML = `${fmt(r.hr, 0)} <small>${T.bpm}</small>`; $("#d-hr").textContent = T.beats(r.nBeats);
      $("#m-valt").innerHTML = `${fmt(r.vAlt, 2)} <small>${T.uv}</small>`;
      $("#m-peak").innerHTML = `${fmt(r.vPeak, 1)} <small>${T.uv}</small>`;
      $("#m-k").innerHTML = fmt(r.k, 1);
      $("#m-mma").innerHTML = `${fmt(r.mma, 1)} <small>${T.uv}</small>`;
      $("#b-fhir").disabled = false;
    } catch (e) {
      lab.res = null;
      box.className = "verdict err"; box.querySelector(".pill").textContent = T.err;
      box.querySelector("p").textContent = e.message;
      $("#b-fhir").disabled = true;
    }
    drawLab();
  }

  function drawLab() {
    drawStrip(); drawOverlay(); drawSeries(); drawSpectrum();
  }

  function drawStrip() {
    const cv = $("#c-strip"); const { ctx, w, h } = setupCanvas(cv, 200);
    const { x, fs } = lab.sig, r = lab.res;
    const pxmm = Math.max(3, w / 125), pps = 25 * pxmm, pxmv = 10 * pxmm;
    const win = Math.min(w / pps, x.length / fs);
    paperGrid(ctx, w, h, pxmm);
    const xs = r ? r.xf : x;
    // vertical centre from median
    const n = Math.floor(win * fs);
    const sl = Array.from(xs.subarray ? xs.subarray(0, n) : xs.slice(0, n));
    const med = D.percentile(sl, 50), base = h * 0.62;
    if (r) {
      const [a, b] = r.window;
      ctx.fillStyle = C["accent-soft"];
      for (const ri of r.r) { const t = ri / fs; if (t > win) break; ctx.fillRect((t + a) * pps, 0, (b - a) * pps, h); }
    }
    ctx.beginPath(); ctx.strokeStyle = C.trace; ctx.lineWidth = 1.3;
    for (let i = 0; i < n; i++) { const X = i / fs * pps, Y = base - (sl[i] - med) * pxmv; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke();
    if (r) {
      ctx.fillStyle = C.accent;
      for (const ri of r.r) { const t = ri / fs; if (t > win) break; ctx.beginPath(); ctx.arc(t * pps, base - (xs[ri] - med) * pxmv, 3.2, 0, 7); ctx.fill(); }
    }
    ctx.font = MONO; ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText(T.paperScale, w - 8, h - 6);
  }

  function drawOverlay() {
    const cv = $("#c-overlay"); const { ctx, w, h } = setupCanvas(cv, 230);
    const r = lab.res;
    if (!r) { ctx.clearRect(0, 0, w, h); return; }
    const B = r.fullBeats, L = B[0].length, fs = lab.sig.fs;
    const A = new Float64Array(L), Bo = new Float64Array(L); let na = 0, nb = 0;
    B.forEach((row, i) => { const T0 = i % 2 ? Bo : A; row.forEach((v, k) => (T0[k] += v)); i % 2 ? nb++ : na++; });
    for (let k = 0; k < L; k++) { A[k] /= na; Bo[k] /= nb; }
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < L; k++) { lo = Math.min(lo, A[k], Bo[k], (A[k] - Bo[k]) * 10); hi = Math.max(hi, A[k], Bo[k], (A[k] - Bo[k]) * 10); }
    const yt = niceTicks(lo, hi, 4), pad = { l: 44, r: 12, t: 20, b: 28 };
    const { X, Y } = axes(ctx, w, h, pad, [-0.25, 0.55], [Math.min(lo, yt[0]), Math.max(hi, yt[yt.length - 1])],
      { xt: [-0.2, 0, 0.2, 0.4], yt, xf: (v) => fmt(v, 1), yf: (v) => fmt(v, 1), xl: T.fromR, yl: T.mv });
    const [a, b] = r.window;
    ctx.fillStyle = C["accent-soft"]; ctx.fillRect(X(a), pad.t, X(b) - X(a), h - pad.t - pad.b);
    const line = (arr, col, lw, f = 1) => { ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = lw;
      for (let k = 0; k < L; k++) { const px = X(k / fs - 0.25), py = Y(arr[k] * f); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke(); };
    const diff = A.map((v, k) => v - Bo[k]);
    ctx.setLineDash([4, 3]); line(diff, C.muted, 1.2, 10); ctx.setLineDash([]);
    line(A, C.blue, 2); line(Bo, C.accent, 2);
  }

  function drawSeries() {
    const cv = $("#c-series"); const { ctx, w, h } = setupCanvas(cv, 200);
    const r = lab.res;
    if (!r) { ctx.clearRect(0, 0, w, h); return; }
    // pick the ST-T sample with the largest alternating component
    const fs = lab.sig.fs, off = Math.round((r.window[0] + 0.25) * fs), len = Math.round((r.window[1] - r.window[0]) * fs);
    const B = r.fullBeats.slice(0, r.nBeats);
    let bestK = off, best = -1;
    for (let k = off; k < off + len && k < B[0].length; k++) {
      let s = 0; B.forEach((row, n) => (s += row[k] * (n % 2 ? -1 : 1)));
      if (Math.abs(s) > best) { best = Math.abs(s); bestK = k; }
    }
    const vals = B.map((row) => row[bestK]);
    const m = vals.reduce((s, v) => s + v, 0) / vals.length;
    const uv = vals.map((v) => (v - m) * 1000);
    const lim = Math.max(2, ...uv.map(Math.abs)) * 1.1;
    const yt = niceTicks(-lim, lim, 4), pad = { l: 44, r: 12, t: 20, b: 28 };
    const N = uv.length;
    const { X, Y } = axes(ctx, w, h, pad, [0, N], [-lim, lim], { xt: niceTicks(0, N, 6), yt, yf: (v) => fmt(v, 0), xl: T.beat, yl: T.uv });
    const bw = Math.max(1, (X(1) - X(0)) * 0.7);
    uv.forEach((v, n) => { ctx.fillStyle = n % 2 ? C.accent : C.blue; const y0 = Y(0), y1 = Y(v);
      ctx.fillRect(X(n + 0.5) - bw / 2, Math.min(y0, y1), bw, Math.max(1, Math.abs(y1 - y0))); });
  }

  function drawSpectrum() {
    const cv = $("#c-spec"); const { ctx, w, h } = setupCanvas(cv, 200);
    const r = lab.res;
    if (!r) { ctx.clearRect(0, 0, w, h); return; }
    const f = r.freqs, P = r.P.map((v) => Math.max(v, 1e-3));
    const lp = P.slice(1).map(Math.log10), lo = Math.floor(Math.min(...lp)), hi = Math.ceil(Math.max(...lp));
    const yt = []; for (let e = lo; e <= hi; e++) yt.push(e);
    const pad = { l: 52, r: 12, t: 20, b: 28 };
    const { X, Y } = axes(ctx, w, h, pad, [0, 0.5], [lo, hi], { xt: [0, 0.1, 0.2, 0.3, 0.4, 0.5], yt,
      xf: (v) => fmt(v, 1), yf: (e) => (e >= 0 ? "1e" + e : "1e" + e), xl: T.cpb, yl: T.uv + "²" });
    ctx.fillStyle = C["blue-soft"]; ctx.fillRect(X(0.44), pad.t, X(0.49) - X(0.44), h - pad.t - pad.b);
    ctx.beginPath(); ctx.strokeStyle = C.trace; ctx.lineWidth = 1.4;
    for (let i = 1; i < f.length; i++) { const px = X(f[i]), py = Y(Math.log10(P[i])); i > 1 ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    const pa = P[P.length - 1];
    ctx.fillStyle = r.positive ? C.accent : C.muted; ctx.beginPath(); ctx.arc(X(0.5) - 2, Y(Math.log10(pa)), 4.5, 0, 7); ctx.fill();
  }

  // ---------- file upload: plain CSV/TXT, or binary EDF/EDF+ with a channel picker ----------
  function isEdf(file) {
    return /\.edf$/i.test(file.name);
  }
  function loadFromChannel(sig, fs, name) {
    lab.fileSig = sig; lab.fileFs = fs; lab.fileName = name; lab.source = "file";
    runLab();
  }
  function populateEdfChannels(edf) {
    const sel = $("#f-edf-channel");
    sel.innerHTML = "";
    edf.signals.forEach((s, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = s.annotations ? `${s.label} ${T.edfAnnotations}` : `${s.label} (${fmt(s.fs, 0)} Hz)`;
      opt.disabled = s.annotations;
      sel.appendChild(opt);
    });
    const guess = edf.signals.findIndex((s) => !s.annotations && /ecg|ekg|\bii\b|lead/i.test(s.label));
    const first = edf.signals.findIndex((s) => !s.annotations);
    sel.value = String(guess >= 0 ? guess : Math.max(0, first));
    selectEdfChannel(parseInt(sel.value, 10));
  }
  function selectEdfChannel(idx) {
    const edf = lab.edf; if (!edf) return;
    const s = edf.signals[idx];
    const sig = edf.getChannel(idx);
    $("#f-status").textContent = T.edfOk(sig.length, Math.round(s.fs), s.label);
    loadFromChannel(sig, s.fs, lab.fileName);
  }
  function parseCsvText(text) {
    const col = Math.max(0, parseInt($("#f-col").value || "0", 10));
    const unit = $("#f-unit").value, fs = Math.max(50, parseFloat($("#f-fs").value) || 500);
    const vals = [];
    for (const line of text.split(/\r?\n/)) {
      const parts = line.split(/[,;\t ]+/).filter(Boolean);
      const v = parseFloat(parts[col]);
      if (isFinite(v)) vals.push(unit === "uv" ? v / 1000 : v);
    }
    const status = $("#f-status");
    if (vals.length < fs * 10) { status.textContent = T.fileErr; return; }
    status.textContent = T.fileOk(vals.length, fs);
    loadFromChannel(Float64Array.from(vals), fs, lab.fileName);
  }
  function onFile(ev) {
    const file = ev.target.files && ev.target.files[0]; if (!file) return;
    lab.fileName = file.name;
    const csvWrap = $("#f-csv-wrap"), edfWrap = $("#f-edf-wrap");
    if (isEdf(file)) {
      lab.csvRawText = null;
      const rd = new FileReader();
      rd.onload = () => {
        try {
          const edf = EDF.parseEDF(rd.result);
          lab.edf = edf;
          csvWrap.hidden = true; edfWrap.hidden = false;
          populateEdfChannels(edf);
        } catch (e) {
          $("#f-status").textContent = T.fileErrEdf(e.message);
        }
      };
      rd.readAsArrayBuffer(file);
      return;
    }
    csvWrap.hidden = false; edfWrap.hidden = true; lab.edf = null;
    const rd = new FileReader();
    rd.onload = () => { lab.csvRawText = String(rd.result); parseCsvText(lab.csvRawText); };
    rd.readAsText(file);
  }

  // ---------- FHIR export (mirrors cardioonco/fhir.py: same codes, same "research only" tag) ----------
  const FHIR_CS = "https://github.com/podpirovlab/multimodal-cardiotoxicity-ai/fhir/CodeSystem/cardioonco";
  const UCUM = "http://unitsofmeasure.org";
  function fhirObs(id, code, display, value, unit) {
    return { resourceType: "Observation", id, status: "final",
      code: { coding: [{ system: FHIR_CS, code, display }], text: display },
      valueQuantity: { value: +value.toFixed(4), unit, system: UCUM, code: unit } };
  }
  function exportFHIR() {
    const r = lab.res; if (!r) return;
    const p = patient();
    const now = new Date().toISOString();
    const contained = [
      fhirObs("hr", "heart-rate", "Heart rate (from R-R intervals)", r.hr, "/min"),
      fhirObs("valt", "twa-valt", "T-wave alternans voltage (Spectral Method)", r.vAlt, "uV"),
      fhirObs("kscore", "twa-k", "T-wave alternans K-score", r.k, "1"),
      fhirObs("mma", "twa-mma", "T-wave alternans (Modified Moving Average)", r.mma, "uV"),
    ];
    const patientResource = {
      resourceType: "Patient", id: "patient",
      gender: p.sex === "male" ? "male" : p.sex === "female" ? "female" : "unknown",
    };
    if (p.age) patientResource.extension = [{ url: "age-years", valueInteger: Math.round(p.age) }];
    if (p.dose) patientResource.extension = (patientResource.extension || []).concat(
      [{ url: `${FHIR_CS}/cumulative-doxorubicin-dose-mg-m2`, valueDecimal: p.dose }]);
    const report = {
      resourceType: "DiagnosticReport", id: crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2),
      meta: { tag: [{ system: FHIR_CS, code: "research-only", display: "Research prototype output - not for clinical use" }] },
      contained: [patientResource, ...contained],
      status: "preliminary",
      category: [{ coding: [{ system: "http://terminology.hl7.org/CodeSystem/v2-0074", code: "EC", display: "Electrocardiac (e.g., EKG, EEC, Holter)" }] }],
      code: { coding: [{ system: "http://loinc.org", code: "11524-6", display: "EKG study" }], text: "CardioOncoPredict TWA analysis (research prototype)" },
      subject: { reference: "#patient" },
      effectiveDateTime: now, issued: now,
      result: contained.map((o) => ({ reference: `#${o.id}` })),
      conclusion: r.positive ? T.posText(r) : T.negText(r),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "cardioonco_twa_report.json";
    document.body.appendChild(a); a.click(); a.remove();
  }

  function initLab() {
    sliders.forEach((id) => $("#s-" + id).addEventListener("input", () => { lab.source = "synth"; clearPresets(); schedule(); }));
    document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const p = JSON.parse(b.dataset.preset);
      for (const k in p) $("#s-" + k).value = p[k];
      clearPresets(); b.setAttribute("aria-pressed", "true"); lab.source = "synth"; schedule();
    }));
    $("#b-reseed").addEventListener("click", () => { lab.seed = (lab.seed * 48271) % 2147483647; lab.source = "synth"; schedule(); });
    $("#f-file").addEventListener("change", onFile);
    ["#f-fs", "#f-col", "#f-unit"].forEach((sel) => $(sel).addEventListener("input", () => { if (lab.csvRawText) parseCsvText(lab.csvRawText); }));
    $("#f-edf-channel").addEventListener("change", (e) => selectEdfChannel(parseInt(e.target.value, 10)));
    ["#p-age", "#p-sex", "#p-dose"].forEach((sel) => $(sel).addEventListener("input", updatePatientContext));
    $("#b-fhir").addEventListener("click", exportFHIR);
    syncOutputs(); updatePatientContext(); runLab();
  }
  function clearPresets() { document.querySelectorAll("[data-preset]").forEach((b) => b.setAttribute("aria-pressed", "false")); }

  // ======================= boot / resize / theme =======================
  function redrawAll() { readColors(); if (lab.sig) drawLab(); }
  let rz = null;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(redrawAll, 120); });
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  function boot() {
    initLab();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
