/* CardioOncoPredict — page logic: hero monitor, TWA lab, charts, 3D heart, training status.
   All numbers are computed by CardioDSP (assets/js/dsp.js). Nothing is hard-coded. */
(function () {
  "use strict";
  const D = window.CardioDSP;
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
      fileErr: "В файле не найдено чисел. Нужен CSV/TXT: один столбец значений в мВ (или мкВ).",
      fileOk: (n, fs) => `Загружено ${n} отсчётов, ${fs} Гц`,
      pending: "Модель ещё не обучена на реальных данных. Запустите обучение — и сюда автоматически подтянутся метрики из assets/metrics.json.",
      trained: (m) => `Обучено на PTB-XL: лучшая эпоха ${m.best_epoch}, тестовая выборка ${m.n_test} ЭКГ, macro-AUC ${fmt(m.test_macro_auc, 3)}.`,
      cls: "Класс", auc: "AUC (95% ДИ)", sens: "Чувств.", spec: "Специф.", prev: "Доля",
      heartIdle: "Выберите группу отведений", dose: "Кумулятивная доза доксорубицина, мг/м²", hf: "Сердечная недостаточность",
      heartHint: "ПОТЯНИТЕ, ЧТОБЫ ПОВЕРНУТЬ", mv: "мВ", paperScale: "25 мм/с · 10 мм/мВ",
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
      fileErr: "No numbers found. Use CSV/TXT with one column of values in mV (or µV).",
      fileOk: (n, fs) => `Loaded ${n} samples at ${fs} Hz`,
      pending: "The model has not been trained on real data yet. Run training and the metrics from assets/metrics.json will appear here automatically.",
      trained: (m) => `Trained on PTB-XL: best epoch ${m.best_epoch}, test set ${m.n_test} ECGs, macro-AUC ${fmt(m.test_macro_auc, 3)}.`,
      cls: "Class", auc: "AUC (95% CI)", sens: "Sens.", spec: "Spec.", prev: "Prevalence",
      heartIdle: "Choose a lead group", dose: "Cumulative doxorubicin dose, mg/m²", hf: "Heart failure",
      heartHint: "DRAG TO ROTATE", mv: "mV", paperScale: "25 mm/s · 10 mm/mV",
    },
  }[LANG === "ru" ? "ru" : "en"];

  function fmt(v, d) {
    if (!isFinite(v)) return "—";
    const s = v.toFixed(d);
    return LANG === "ru" ? s.replace(".", ",") : s;
  }
  const $ = (s) => document.querySelector(s);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- theme-aware colours ----------
  let C = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["paper", "surface", "ink", "muted", "line", "grid-minor", "grid-major", "trace", "accent", "blue", "ok", "accent-soft", "blue-soft"])
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

  // ======================= HERO MONITOR =======================
  const hero = { sig: null, res: null, t0: performance.now() - 20000 };  // start mid-recording: the first frame is a full strip
  function initHero() {
    hero.sig = D.synth({ altUv: 12, hr: 72, noiseUv: 12, tAmp: 0.3, seed: 2026, wanderMv: 0.05 });
    try { hero.res = D.analyzeTWA(hero.sig.x, hero.sig.fs); } catch (e) { hero.res = null; }
    const r = hero.res;
    if (r) {
      $("#ro-hr").innerHTML = `${fmt(r.hr, 0)} <small>${T.bpm}</small>`;
      $("#ro-valt").innerHTML = `${fmt(r.vPeak, 1)} <small>${T.uv}</small>`;
      $("#ro-k").innerHTML = `${fmt(r.k, 0)}`;
      $("#ro-mm").innerHTML = `${fmt(r.vPeak / 100, 2)} <small>${T.mm}</small>`;
    }
    drawHero();
    if (!reduceMotion) requestAnimationFrame(loopHero);
  }
  function drawHero(now) {
    const cv = $("#hero-ecg"); if (!cv) return;
    const { ctx, w, h } = setupCanvas(cv);
    const pxmm = Math.max(4, w / 150), pps = 25 * pxmm, pxmv = 10 * pxmm;
    const calW = 8 * pxmm, plotW = w - calW, win = plotW / pps;
    paperGrid(ctx, w, h, pxmm);
    const base = h * 0.62;
    // 1 mV calibration pulse: 10 mm tall, 5 mm wide
    ctx.strokeStyle = C.trace; ctx.lineWidth = 1.6; ctx.beginPath();
    ctx.moveTo(pxmm, base); ctx.lineTo(2 * pxmm, base); ctx.lineTo(2 * pxmm, base - pxmv);
    ctx.lineTo(7 * pxmm, base - pxmv); ctx.lineTo(7 * pxmm, base); ctx.lineTo(8 * pxmm, base); ctx.stroke();
    const { x, fs } = hero.sig, n = x.length, total = n / fs;
    const elapsed = now ? (now - hero.t0) / 1000 : win * 0.999;
    const sweep = Math.floor(elapsed / win), cursor = (elapsed % win) * pps, gap = 12;
    ctx.beginPath(); ctx.strokeStyle = C.trace; ctx.lineWidth = 1.5; ctx.lineJoin = "round";
    let pen = false;
    for (let px = 0; px < plotW; px += 1) {
      if (px > cursor && px < cursor + gap) { pen = false; continue; }
      const s = px <= cursor ? sweep : sweep - 1;
      if (s < 0 && now) { pen = false; continue; }
      let ts = (s * win + px / pps) % total; if (ts < 0) ts += total;
      const i = Math.min(n - 1, Math.floor(ts * fs));
      const y = base - x[i] * pxmv;
      if (!pen) { ctx.moveTo(calW + px, y); pen = true; } else ctx.lineTo(calW + px, y);
    }
    ctx.stroke();
    if (now) { ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(calW + cursor, base - x[Math.min(n - 1, Math.floor(((sweep * win + cursor / pps) % total) * fs))] * pxmv, 3, 0, 7); ctx.fill(); }
  }
  let heroVisible = true;
  function loopHero(now) { if (heroVisible) drawHero(now); requestAnimationFrame(loopHero); }
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((e) => { heroVisible = e[0].isIntersecting; }).observe($("#hero-ecg"));
  }

  // ======================= DOSE CHART (Swain et al., Cancer 2003) =======================
  function drawDose() {
    const cv = $("#dose-chart"); if (!cv) return;
    const { ctx, w, h } = setupCanvas(cv, 250);
    const pad = { l: 44, r: 16, t: 26, b: 34 };
    const { X, Y } = axes(ctx, w, h, pad, [300, 750], [0, 50], { xt: [300, 400, 500, 600, 700], yt: [0, 10, 20, 30, 40, 50],
      yf: (v) => v + "%", xl: T.dose, yl: T.hf });
    const pts = [[400, 5], [550, 26], [700, 48]];
    const bw = Math.min(46, (w - pad.l - pad.r) / 9);
    for (const [d, p] of pts) {
      ctx.fillStyle = C.accent; ctx.globalAlpha = 0.85;
      ctx.fillRect(X(d) - bw / 2, Y(p), bw, Y(0) - Y(p)); ctx.globalAlpha = 1;
      ctx.fillStyle = C.ink; ctx.font = "600 13px 'JetBrains Mono', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(p + "%", X(d), Y(p) - 4);
    }
  }

  // ======================= TWA LAB =======================
  const lab = { res: null, sig: null, seed: 7, source: "synth", fileSig: null, fileFs: 500, fileName: "" };
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
      box.querySelector("p").textContent = (lab.source === "file" ? lab.fileName + ". " : "") + (r.positive ? T.posText(r) : T.negText(r));
      $("#m-hr").innerHTML = `${fmt(r.hr, 0)} <small>${T.bpm}</small>`; $("#d-hr").textContent = T.beats(r.nBeats);
      $("#m-valt").innerHTML = `${fmt(r.vAlt, 2)} <small>${T.uv}</small>`;
      $("#m-peak").innerHTML = `${fmt(r.vPeak, 1)} <small>${T.uv}</small>`;
      $("#m-k").innerHTML = fmt(r.k, 1);
      $("#m-mma").innerHTML = `${fmt(r.mma, 1)} <small>${T.uv}</small>`;
    } catch (e) {
      lab.res = null;
      box.className = "verdict err"; box.querySelector(".pill").textContent = T.err;
      box.querySelector("p").textContent = e.message;
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

  // file upload
  function onFile(ev) {
    const file = ev.target.files && ev.target.files[0]; if (!file) return;
    const rd = new FileReader();
    rd.onload = () => {
      const col = Math.max(0, parseInt($("#f-col").value || "0", 10));
      const unit = $("#f-unit").value, fs = Math.max(50, parseFloat($("#f-fs").value) || 500);
      const vals = [];
      for (const line of String(rd.result).split(/\r?\n/)) {
        const parts = line.split(/[,;\t ]+/).filter(Boolean);
        const v = parseFloat(parts[col]);
        if (isFinite(v)) vals.push(unit === "uv" ? v / 1000 : v);
      }
      const status = $("#f-status");
      if (vals.length < fs * 10) { status.textContent = T.fileErr; return; }
      lab.fileSig = Float64Array.from(vals); lab.fileFs = fs; lab.fileName = file.name; lab.source = "file";
      status.textContent = T.fileOk(vals.length, fs);
      runLab();
    };
    rd.readAsText(file);
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
    syncOutputs(); runLab();
  }
  function clearPresets() { document.querySelectorAll("[data-preset]").forEach((b) => b.setAttribute("aria-pressed", "false")); }

  // ======================= 3D LEFT VENTRICLE =======================
  const heart = { ax: -0.35, ay: 0.5, sel: null, pts: [], drag: false };
  const TERR = { septal: [45, 135], inferior: [135, 225], lateral: [225, 315], anterior: [315, 45] };
  function territory(phiDeg, z) {
    if (z < -0.72) return "apex";
    for (const [k, [a, b]] of Object.entries(TERR)) {
      if (a < b ? phiDeg >= a && phiDeg < b : phiDeg >= a || phiDeg < b) return k;
    }
    return "anterior";
  }
  function initHeartGeom() {
    // half prolate ellipsoid ("bullet"): open, widest at the base, closing to the apex
    const rings = 16, seg = 28, pts = [];
    for (let i = 0; i <= rings; i++) {
      const u = i / rings, z = 0.85 - u * 1.8;         // base (0.85) → apex (−0.95)
      const rad = 0.62 * Math.sqrt(Math.max(0, 1 - u * u));
      const row = [];
      for (let j = 0; j < seg; j++) {
        const phi = (j / seg) * 2 * Math.PI;
        row.push({ x: rad * Math.sin(phi), y: z, z: rad * Math.cos(phi), t: territory((phi * 180) / Math.PI, z) });
      }
      pts.push(row);
    }
    heart.pts = pts;
  }
  const LEADS = { septal: ["septal"], anterior: ["anterior", "apex"], lateral: ["lateral"], inferior: ["inferior", "apex"] };
  function drawHeart() {
    const cv = $("#c-heart"); if (!cv) return;
    const { ctx, w, h } = setupCanvas(cv);
    ctx.fillStyle = C.surface; ctx.fillRect(0, 0, w, h);
    const s = Math.min(w, h) * 0.42, cx = w / 2, cy = h / 2;
    const ca = Math.cos(heart.ax), sa = Math.sin(heart.ax), cb = Math.cos(heart.ay), sb = Math.sin(heart.ay);
    const P = heart.pts.map((row) => row.map((p) => {
      const x1 = p.x * cb + p.z * sb, z1 = -p.x * sb + p.z * cb;
      const y2 = p.y * ca - z1 * sa, z2 = p.y * sa + z1 * ca;
      return { X: cx + x1 * s, Y: cy - y2 * s, d: z2, t: p.t };
    }));
    const active = heart.sel ? LEADS[heart.sel] : [];
    const segs = [];
    for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
      const a = P[i][j], b = P[i][(j + 1) % P[i].length];
      segs.push([a, b]);
      if (i + 1 < P.length) segs.push([a, P[i + 1][j]]);
    }
    segs.sort((u, v) => (u[0].d + u[1].d) - (v[0].d + v[1].d));
    for (const [a, b] of segs) {
      const depth = (a.d + b.d) / 2, front = depth > 0;
      const on = active.includes(a.t) && active.includes(b.t);
      ctx.strokeStyle = on ? C.accent : C.trace;
      ctx.globalAlpha = on ? (front ? 0.95 : 0.3) : (front ? 0.35 : 0.08);
      ctx.lineWidth = on && front ? 1.8 : 1;
      ctx.beginPath(); ctx.moveTo(a.X, a.Y); ctx.lineTo(b.X, b.Y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.font = MONO; ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(LANG === "ru" ? "ОСНОВАНИЕ" : "BASE", 12, 12);
    ctx.textBaseline = "bottom"; ctx.fillText(LANG === "ru" ? "ВЕРХУШКА ↓" : "APEX ↓", 12, h - 10);
  }
  function initHeart() {
    initHeartGeom(); drawHeart();
    const st = $("#heart-stage");
    let px = 0, py = 0;
    st.addEventListener("pointerdown", (e) => { heart.drag = true; px = e.clientX; py = e.clientY; st.setPointerCapture(e.pointerId); });
    st.addEventListener("pointermove", (e) => { if (!heart.drag) return; heart.ay += (e.clientX - px) * 0.01; heart.ax += (e.clientY - py) * 0.01; px = e.clientX; py = e.clientY; drawHeart(); });
    st.addEventListener("pointerup", () => (heart.drag = false));
    document.querySelectorAll("[data-lead]").forEach((b) => b.addEventListener("click", () => {
      const on = b.getAttribute("aria-pressed") === "true";
      document.querySelectorAll("[data-lead]").forEach((x) => x.setAttribute("aria-pressed", "false"));
      heart.sel = on ? null : b.dataset.lead;
      if (!on) b.setAttribute("aria-pressed", "true");
      drawHeart();
    }));
    if (!reduceMotion) {
      let last = performance.now(), vis = true;
      if ("IntersectionObserver" in window) new IntersectionObserver((e) => (vis = e[0].isIntersecting)).observe(st);
      const spin = (now) => { if (vis && !heart.drag) { heart.ay += (now - last) * 0.00025; drawHeart(); } last = now; requestAnimationFrame(spin); };
      requestAnimationFrame(spin);
    }
  }

  // ======================= TRAINING STATUS =======================
  function initStatus() {
    const box = $("#train-status"); if (!box) return;
    box.querySelector("p").textContent = T.pending;
    fetch("assets/metrics.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((m) => {
      if (!m || !m.per_class) return;
      box.querySelector("p").textContent = T.trained(m);
      box.classList.add("done");
      const rows = Object.entries(m.per_class).map(([c, v]) =>
        `<tr><td>${c}</td><td>${fmt(v.auc, 3)} (${fmt(v.auc_ci95[0], 3)}–${fmt(v.auc_ci95[1], 3)})</td><td>${fmt(v.sensitivity * 100, 0)}%</td><td>${fmt(v.specificity * 100, 0)}%</td><td>${fmt(v.prevalence * 100, 0)}%</td></tr>`).join("");
      const tw = document.createElement("div"); tw.className = "table-wrap";
      tw.innerHTML = `<table><thead><tr><th>${T.cls}</th><th>${T.auc}</th><th>${T.sens}</th><th>${T.spec}</th><th>${T.prev}</th></tr></thead><tbody>${rows}</tbody></table>`;
      box.appendChild(tw);
      const cmd = box.querySelector("pre"); if (cmd) cmd.hidden = true;
    }).catch(() => {});
  }

  // ======================= boot / resize / theme =======================
  function redrawAll() { readColors(); drawHero(); drawDose(); if (lab.sig) drawLab(); drawHeart(); }
  let rz = null;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(redrawAll, 120); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  function boot() {
    initHero(); drawDose(); initLab(); initHeart(); initStatus();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
