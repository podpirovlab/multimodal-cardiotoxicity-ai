/* CardioOncoPredict — page logic: upload/synthesise a recording, run the TWA analysis, draw the charts.
   All numbers are computed by CardioDSP (assets/js/dsp.js) and CardioEDF (assets/js/edf.js). Nothing is hard-coded. */
(function () {
  "use strict";
  const D = window.CardioDSP;
  const EDF = window.CardioEDF;
  const LANG = (document.documentElement.lang || "en").slice(0, 2);
  const T = {
    ru: {
      pos: "Критерий альтернации выполнен", neg: "Значимой альтернации нет", ind: "Не определено", err: "Анализ невозможен",
      busy: "Анализ…", busyText: "Считаю окна по всей записи.",
      none: "Запись не загружена", noneText: "Загрузите запись или откройте пример ниже.",
      synthName: "синтетический пример", version: (v) => `версия ${v}`,
      beatsWindows: (n, w) => `${n} ${plural(n, "удар", "удара", "ударов")}, ${w} ${plural(w, "окно", "окна", "окон")}`,
      posText: (r) => `Альтернация ${fmt(r.vAlt, 1)} мкВ при ${kEq(r.k)} в окне с ЧСС ${fmt(r.windowHr, 0)} уд/мин: выполнен критерий спектрального метода (не меньше 1,9 мкВ, K не меньше 3, ЧСС не выше 110, шум не выше 1,8 мкВ). На бумажной ЭКГ это ${fmt(r.vPeak / 100, 2)} мм — глазом не увидеть.`,
      negText: (r) => `Значимой альтернации нет ни в одном окне (наибольшая ${fmt(r.vAlt, 1)} мкВ, ${kEq(r.k)}), в том числе в чистых окнах с ЧСС до ${fmt(r.hrMaxClean, 0)} уд/мин. Это отрицательный результат по правилам метода: для него ЧСС должна дойти до 105.`,
      indText: {
        hr_too_low: (r) => `Значимой альтернации нет, но ЧСС в чистых окнах не поднималась до 105 уд/мин (максимум ${fmt(r.hrMaxClean, 0)}). По правилам спектрального метода отрицательный результат без этого не выдаётся: альтернация часто появляется только при нагрузке.`,
        hr_too_high: (r) => `Альтернация ${fmt(r.vAlt, 1)} мкВ (${kEq(r.k)}) есть только в окнах с ЧСС выше 110 уд/мин. По правилам метода такое начало положительным результатом не считается.`,
        noise: (r) => `Шум в полосе 0,44–0,49 цикла на удар — ${fmt(r.noise, 1)} мкВ, а допустимо не больше 1,8 мкВ. Сказать ни «да», ни «нет» нельзя: нужна более чистая запись.`,
        ectopy: () => "В каждом окне больше 10% экстрасистол или артефактов, поэтому альтернацию оценить нельзя.",
      },
      windows: (n) => `${n} ${plural(n, "окно", "окна", "окон")}`,
      tooShort: (n, min) => `Запись слишком короткая: ${n} пригодных ударов, а нужно не меньше ${min} (около минуты; стандарт метода — 128 ударов, около 2 минут). Результат «норма» здесь был бы неправдой.`,
      noBeats: "В записи не удалось найти удары сердца. Проверьте частоту дискретизации, единицы и выбранный канал.",
      noNoise: "Не удалось оценить уровень шума — результат был бы бессмысленным.",
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
      unitUnknown: (u) => ` Единицы канала «${u || "не указаны"}» не распознаны как напряжение: считаю, что значения в мВ.`,
      edfChannel: "Канал (отведение)", edfAnnotations: "(служебный канал, пропущен)",
      shortWarn: (n) => `Проанализировано ${n} ударов — меньше стандартных 128, поэтому к результату стоит относиться осторожнее.`,
    },
    en: {
      pos: "Alternans criterion met", neg: "No significant alternans", ind: "Indeterminate", err: "Cannot analyse",
      busy: "Analysing…", busyText: "Scanning windows across the whole recording.",
      none: "No recording loaded", noneText: "Upload a recording or open an example below.",
      synthName: "synthetic example", version: (v) => `version ${v}`,
      beatsWindows: (n, w) => `${n} beats, ${w} window${w === 1 ? "" : "s"}`,
      posText: (r) => `Alternans ${fmt(r.vAlt, 1)} µV with ${kEq(r.k)} in a window at ${fmt(r.windowHr, 0)} bpm: meets the Spectral Method criterion (at least 1.9 µV, K at least 3, heart rate at most 110, noise at most 1.8 µV). On paper ECG that is ${fmt(r.vPeak / 100, 2)} mm, invisible to the eye.`,
      negText: (r) => `No significant alternans in any window (largest ${fmt(r.vAlt, 1)} µV, ${kEq(r.k)}), including clean windows at up to ${fmt(r.hrMaxClean, 0)} bpm. That is a negative result under the method's rules, which require the heart rate to reach 105.`,
      indText: {
        hr_too_low: (r) => `No significant alternans, but the heart rate in clean windows never reached 105 bpm (highest ${fmt(r.hrMaxClean, 0)}). The Spectral Method does not call a test negative without it: alternans often appears only on exertion.`,
        hr_too_high: (r) => `Alternans of ${fmt(r.vAlt, 1)} µV (${kEq(r.k)}) appears only in windows above 110 bpm. Under the method's rules an onset that late does not count as positive.`,
        noise: (r) => `Noise in the 0.44–0.49 cycles/beat band is ${fmt(r.noise, 1)} µV, above the 1.8 µV limit. Neither yes nor no can be said; a cleaner recording is needed.`,
        ectopy: () => "Every window has more than 10% ectopic or artefactual beats, so alternans cannot be assessed.",
      },
      windows: (n) => `${n} window${n === 1 ? "" : "s"}`,
      tooShort: (n, min) => `Recording too short: ${n} usable beats, at least ${min} are needed (about a minute; the method's standard is 128 beats, about 2 minutes). Calling this "normal" would be untrue.`,
      noBeats: "No heartbeats found in this recording. Check the sampling rate, units and selected channel.",
      noNoise: "The noise level could not be estimated, so a result would be meaningless.",
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
      unitUnknown: (u) => ` The channel's unit "${u || "none"}" is not a voltage I recognise, so the values are read as mV.`,
      edfChannel: "Channel (lead)", edfAnnotations: "(service channel, skipped)",
      shortWarn: (n) => `Analysed ${n} beats — fewer than the standard 128, so treat the result with more caution.`,
    },
  }[LANG === "ru" ? "ru" : "en"];

  function plural(n, one, few, many) {
    const m10 = n % 10, m100 = n % 100;
    return m10 === 1 && m100 !== 11 ? one : m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14) ? few : many;
  }
  function fmt(v, d) {
    if (!isFinite(v)) return "—";
    const s = v.toFixed(d);
    return LANG === "ru" ? s.replace(".", ",") : s;
  }
  // Clinical K-scores are single or double digits; clean synthetic noise pushes K into the
  // hundreds, which reads as a malfunction, so anything above 100 is shown as ">100".
  const fmtK = (k) => (isFinite(k) && k > 100 ? ">100" : fmt(k, 1));
  const kEq = (k) => (isFinite(k) && k > 100 ? "K > 100" : `K = ${fmt(k, 1)}`);
  const $ = (s) => document.querySelector(s);

  // ---------- theme-aware colours (read once from CSS custom properties) ----------
  let C = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["paper", "surface", "ink", "muted", "line", "grid-minor", "grid-major", "trace", "rpeak", "window", "band", "even", "odd", "finding"])
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
  const MONO = "400 12px 'PT Mono', ui-monospace, monospace";

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
  // ======================= RESEARCH METADATA (age / sex / dose) =======================
  // Never used by the analysis and never turned into a risk figure; it only goes into the
  // downloaded JSON, and only when the user ticks "include".
  function metadata() {
    const age = parseFloat($("#p-age").value);
    const sex = $("#p-sex").value;
    const dose = parseFloat($("#p-dose").value);
    return { age: isFinite(age) ? age : null, sex: sex || null, dose: isFinite(dose) && dose > 0 ? dose : null };
  }

  // ======================= TWA LAB =======================
  // source: "none" until the user uploads a file or opens a synthetic example — the page
  // never opens on a result about a person who does not exist.
  const lab = { res: null, sig: null, seed: 7, source: "none", fileSig: null, fileFs: 500, fileName: "", edf: null, csvRawText: null };
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

  function clearMetrics() {
    for (const id of ["#f-rec", "#f-beats", "#m-hr", "#f-ver", "#m-valt", "#m-k", "#m-whr", "#m-noise", "#m-peak", "#m-mma"]) {
      $(id).textContent = "—"; $(id).classList.remove("met", "warn");
    }
  }
  function showEmpty() {
    runSeq++; lab.res = null; lab.sig = null;
    const box = $("#verdict");
    box.className = "verdict none";
    box.querySelector(".pill").textContent = T.none;
    box.querySelector("p").textContent = T.noneText;
    clearMetrics();
    $("#b-fhir").disabled = true;
    drawLab();
  }
  function errorText(e) {
    if (e && e.code === "too_short") return T.tooShort(e.nBeats, e.minBeats);
    if (e && e.code === "no_beats") return T.noBeats;
    if (e && e.code === "no_noise_estimate") return T.noNoise;
    return e && e.message ? e.message : String(e);
  }

  // A new result (a file, an example, a new seed) settles in; slider drags redraw without motion.
  function animateResult() {
    for (const el of [$("#verdict"), $(".criteria"), $(".panels")]) {
      el.classList.remove("fresh"); void el.offsetWidth; el.classList.add("fresh");
    }
    clearTimeout(animateResult.t);
    animateResult.t = setTimeout(() => document.querySelectorAll(".fresh").forEach((el) => el.classList.remove("fresh")), 1200);
  }
  function verdictText(r) {
    if (r.outcome === "positive") return T.posText(r);
    if (r.outcome === "negative") return T.negText(r);
    return (T.indText[r.reason] || T.indText.noise)(r);
  }
  const VERDICT_CLASS = { positive: "pos", negative: "neg", indeterminate: "ind" };
  const VERDICT_PILL = { positive: "pos", negative: "neg", indeterminate: "ind" };

  // The analysis runs in a Web Worker so that an hour-long file does not freeze the page.
  // Where workers are unavailable (e.g. the page opened straight from disk) it runs inline.
  let worker = null, jobSeq = 0, runSeq = 0;
  const jobs = new Map();
  try {
    worker = new Worker("assets/js/twa-worker.js");
    worker.onmessage = (e) => {
      const job = jobs.get(e.data.id); if (!job) return;
      jobs.delete(e.data.id);
      e.data.ok ? job.resolve(e.data.r) : job.reject(Object.assign(new Error(e.data.err.message), e.data.err));
    };
    worker.onerror = (e) => { e.preventDefault(); worker = null; for (const job of jobs.values()) job.inline(); jobs.clear(); };
  } catch (e) { worker = null; }
  function analyse(x, fs) {
    return new Promise((resolve, reject) => {
      const inline = () => { try { resolve(D.analyzeTWA(x, fs)); } catch (err) { reject(err); } };
      if (!worker) { inline(); return; }
      const id = ++jobSeq;
      jobs.set(id, { resolve, reject, inline });
      worker.postMessage({ id, x, fs });
    });
  }

  function runLab() {
    if (lab.source === "none") { showEmpty(); return; }
    let x, fs;
    if (lab.source === "file" && lab.fileSig) { x = lab.fileSig; fs = lab.fileFs; }
    else { const s = D.synth(Object.assign(params(), { seed: lab.seed })); x = s.x; fs = s.fs; }
    const box = $("#verdict"), run = ++runSeq;
    const record = lab.source === "file" ? lab.fileName : T.synthName;
    if (lab.source === "file") {
      box.className = "verdict none"; box.querySelector(".pill").textContent = T.busy;
      box.querySelector("p").textContent = T.busyText;
    }
    analyse(x, fs).then((r) => {
      if (run !== runSeq) return;                 // a newer request has started
      lab.sig = { x, fs }; lab.res = r;
      box.className = "verdict " + VERDICT_CLASS[r.outcome];
      box.querySelector(".pill").textContent = T[VERDICT_PILL[r.outcome]];
      let msg = verdictText(r);
      if (r.windowBeats < D.STANDARD_BEATS) msg += " " + T.shortWarn(r.windowBeats);
      box.querySelector("p").textContent = msg;
      $("#f-rec").textContent = record;
      $("#f-beats").textContent = T.beatsWindows(r.nBeats, r.nWindows);
      $("#m-hr").textContent = `${fmt(r.hr, 0)} ${T.bpm}`;
      $("#f-ver").textContent = T.version(D.version);
      $("#m-valt").innerHTML = `${fmt(r.vAlt, 1)} <small>${T.uv}</small>`;
      $("#m-k").textContent = fmtK(r.k);
      $("#m-whr").innerHTML = `${fmt(r.windowHr, 0)} <small>${T.bpm}</small>`;
      $("#m-noise").innerHTML = `${fmt(r.noise, 1)} <small>${T.uv}</small>`;
      $("#m-peak").innerHTML = `${fmt(r.vPeak, 1)} <small>${T.uv}</small>`;
      $("#m-mma").innerHTML = `${fmt(r.mma, 1)} <small>${T.uv}</small>`;
      // red marks the finding only: the conditions that made this result positive
      $("#m-valt").classList.toggle("met", r.positive);
      $("#m-k").classList.toggle("met", r.positive);
      $("#m-noise").classList.toggle("warn", r.outcome === "indeterminate" && r.reason === "noise");
      $("#b-fhir").disabled = false;
      drawLab();
      if (lab.animate) { lab.animate = false; animateResult(); }
    }, (e) => {
      if (run !== runSeq) return;
      lab.sig = { x, fs }; lab.res = null; lab.animate = false;
      box.className = "verdict err"; box.querySelector(".pill").textContent = T.err;
      clearMetrics();
      box.querySelector("p").textContent = errorText(e);
      $("#f-rec").textContent = record;
      $("#b-fhir").disabled = true;
      drawLab();
    });
  }

  function drawLab() {
    if (!lab.sig) {                       // empty state: blank panels, no invented trace
      for (const [id, h] of [["#c-strip", 200], ["#c-overlay", 230], ["#c-series", 200], ["#c-spec", 200]]) {
        const { ctx, w, h: hh } = setupCanvas($(id), h);
        ctx.fillStyle = C.surface; ctx.fillRect(0, 0, w, hh);
      }
      return;
    }
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
      ctx.fillStyle = C.window;
      for (const ri of r.r) { const t = ri / fs; if (t > win) break; ctx.fillRect((t + a) * pps, 0, (b - a) * pps, h); }
    }
    ctx.beginPath(); ctx.strokeStyle = C.trace; ctx.lineWidth = 1.3;
    for (let i = 0; i < n; i++) { const X = i / fs * pps, Y = base - (sl[i] - med) * pxmv; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke();
    if (r) {
      ctx.fillStyle = C.rpeak;
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
    ctx.fillStyle = C.window; ctx.fillRect(X(a), pad.t, X(b) - X(a), h - pad.t - pad.b);
    const line = (arr, col, lw, f = 1) => { ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = lw;
      for (let k = 0; k < L; k++) { const px = X(k / fs - 0.25), py = Y(arr[k] * f); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke(); };
    const diff = A.map((v, k) => v - Bo[k]);
    ctx.setLineDash([4, 3]); line(diff, C.muted, 1.2, 10); ctx.setLineDash([]);
    line(A, C.even, 2); line(Bo, C.odd, 2);
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
    uv.forEach((v, n) => { ctx.fillStyle = n % 2 ? C.odd : C.even; const y0 = Y(0), y1 = Y(v);
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
    ctx.fillStyle = C.band; ctx.fillRect(X(0.44), pad.t, X(0.49) - X(0.44), h - pad.t - pad.b);
    ctx.beginPath(); ctx.strokeStyle = C.trace; ctx.lineWidth = 1.4;
    for (let i = 1; i < f.length; i++) { const px = X(f[i]), py = Y(Math.log10(P[i])); i > 1 ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    const pa = P[P.length - 1];
    ctx.fillStyle = r.positive ? C.finding : C.ink; ctx.beginPath(); ctx.arc(X(0.5) - 2, Y(Math.log10(pa)), 4.5, 0, 7); ctx.fill();
  }

  // ---------- file upload: plain CSV/TXT, or binary EDF/EDF+ with a channel picker ----------
  function isEdf(file) {
    return /\.edf$/i.test(file.name);
  }
  function loadFromChannel(sig, fs, name) {
    lab.fileSig = sig; lab.fileFs = fs; lab.fileName = name; lab.source = "file"; lab.animate = true;
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
    $("#f-status").textContent = T.edfOk(sig.length, Math.round(s.fs), s.label) + (s.unitToMv === null ? T.unitUnknown(s.unit) : "");
    loadFromChannel(sig, s.fs, lab.fileName);
  }
  function parseCsvText(text) {
    const col = Math.max(0, parseInt($("#f-col").value || "0", 10));
    const unit = $("#f-unit").value, fs = Math.max(50, parseFloat($("#f-fs").value) || 500);
    const vals = EDF.parseCSV(text, col);
    if (unit === "uv") for (let i = 0; i < vals.length; i++) vals[i] /= 1000;
    const status = $("#f-status");
    if (vals.length < fs * 10) { status.textContent = T.fileErr; return; }
    status.textContent = T.fileOk(vals.length, fs);
    loadFromChannel(vals, fs, lab.fileName);
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

  // ---------- research JSON in FHIR R4 format (mirrors cardioonco/fhir.py) ----------
  // Every Observation points to a contained Device carrying the algorithm version, so a number
  // can always be traced to the code that produced it. The recording time is unknown here, so
  // effective[x] is left out rather than filled with the export time.
  const FHIR_CS = "https://github.com/podpirovlab/multimodal-cardiotoxicity-ai/fhir/CodeSystem/cardioonco";
  const UCUM = "http://unitsofmeasure.org";
  const LOINC = "http://loinc.org";
  function fhirObs(id, code, display, value, unit, loinc) {
    const coding = [{ system: FHIR_CS, code, display }];
    if (loinc) coding.unshift({ system: LOINC, code: loinc, display });
    return { resourceType: "Observation", id, status: "preliminary",
      code: { coding, text: display }, device: { reference: "#software" },
      valueQuantity: { value: +value.toFixed(4), unit, system: UCUM, code: unit } };
  }
  function exportFHIR() {
    const r = lab.res; if (!r) return;
    const observations = [
      fhirObs("hr", "heart-rate", "Heart rate (from R-R intervals)", r.hr, "/min", "8867-4"),
      fhirObs("valt", "twa-valt", "T-wave alternans voltage (Spectral Method)", r.vAlt, "uV"),
      fhirObs("kscore", "twa-k", "T-wave alternans K-score", r.k, "1"),
      fhirObs("mma", "twa-mma", "T-wave alternans (Modified Moving Average)", r.mma, "uV"),
      { resourceType: "Observation", id: "outcome", status: "preliminary",
        code: { coding: [{ system: FHIR_CS, code: "twa-outcome", display: "T-wave alternans outcome (Spectral Method rules)" }],
                text: "T-wave alternans outcome (Spectral Method rules)" },
        device: { reference: "#software" },
        valueCodeableConcept: { coding: [{ system: FHIR_CS, code: r.outcome, display: r.outcome }], text: r.reason } },
    ];
    const device = { resourceType: "Device", id: "software",
      deviceName: [{ name: "CardioOncoPredict TWA analysis (research software)", type: "model-name" }],
      version: [{ value: D.version }] };
    const report = {
      resourceType: "DiagnosticReport", id: crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2),
      meta: { tag: [{ system: FHIR_CS, code: "research-only", display: "Research prototype output - not for clinical use" }] },
      contained: [device, ...observations],
      status: "preliminary",
      category: [{ coding: [{ system: "http://terminology.hl7.org/CodeSystem/v2-0074", code: "EC", display: "Electrocardiac (e.g., EKG, EEC, Holter)" }] }],
      code: { coding: [{ system: LOINC, code: "11524-6", display: "EKG study" }], text: "CardioOncoPredict TWA analysis (research prototype)" },
      issued: new Date().toISOString(),
      result: observations.map((o) => ({ reference: `#${o.id}` })),
      conclusion: "RESEARCH USE ONLY — NOT FOR CLINICAL RECORDS. " + verdictText(r),
    };
    if ($("#p-include").checked) {           // research metadata only on explicit opt-in
      const m = metadata(), ext = [];
      if (m.age !== null) ext.push({ url: `${FHIR_CS}/age-at-recording`,
        valueQuantity: { value: Math.round(m.age), unit: "a", system: UCUM, code: "a" } });
      if (m.dose !== null) ext.push({ url: `${FHIR_CS}/cumulative-doxorubicin-dose`,
        valueQuantity: { value: m.dose, unit: "mg/m2", system: UCUM, code: "mg/m2" } });
      const patient = { resourceType: "Patient", id: "subject",
        gender: m.sex === "male" ? "male" : m.sex === "female" ? "female" : "unknown" };
      if (ext.length) patient.extension = ext;
      report.contained.unshift(patient);
      report.subject = { reference: "#subject" };
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "cardioonco_research_twa.json";
    document.body.appendChild(a); a.click(); a.remove();
  }

  function initLab() {
    sliders.forEach((id) => $("#s-" + id).addEventListener("input", () => { lab.source = "synth"; clearPresets(); schedule(); }));
    document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const p = JSON.parse(b.dataset.preset);
      for (const k in p) $("#s-" + k).value = p[k];
      clearPresets(); b.setAttribute("aria-pressed", "true"); lab.source = "synth"; lab.animate = true; schedule();
    }));
    $("#b-reseed").addEventListener("click", () => { lab.seed = (lab.seed * 48271) % 2147483647; lab.source = "synth"; lab.animate = true; schedule(); });
    $("#f-file").addEventListener("change", onFile);
    ["#f-fs", "#f-col", "#f-unit"].forEach((sel) => $(sel).addEventListener("input", () => { if (lab.csvRawText) parseCsvText(lab.csvRawText); }));
    $("#f-edf-channel").addEventListener("change", (e) => selectEdfChannel(parseInt(e.target.value, 10)));
    $("#b-fhir").addEventListener("click", exportFHIR);
    // opening the example panel is an explicit request to see a synthetic result
    $(".demo").addEventListener("toggle", (e) => {
      if (e.target.open && lab.source === "none") { lab.source = "synth"; lab.animate = true; schedule(); }
    });
    syncOutputs(); showEmpty();
  }
  function clearPresets() { document.querySelectorAll("[data-preset]").forEach((b) => b.setAttribute("aria-pressed", "false")); }

  // ======================= cover: a quiet field of dots =======================
  // Ripples spread from the centre about once a second, like a pulse; every other ripple is a
  // little weaker, which is alternans drawn in dots. Purely decorative: it stops when the cover
  // is off screen or the tab is hidden, and stays still for people who prefer reduced motion.
  function coverDots() {
    const cv = document.querySelector(".cover-dots");
    if (!cv || !cv.getContext) return;
    const still = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ctx = cv.getContext("2d"), BEAT = 1.05, STEP = 20;
    let w = 0, h = 0, raf = 0, visible = true;
    function size() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      w = cv.clientWidth; h = cv.clientHeight;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function frame(ms) {
      const s = ms / 1000, k = Math.floor(s / BEAT), phase = (s % BEAT) / BEAT;
      const strength = k % 2 ? 0.62 : 1, cx = w * 0.55, cy = h * 0.5, maxR = Math.hypot(w, h) * 0.55;
      const r = phase * maxR, fade = 1 - phase;
      ctx.clearRect(0, 0, w, h);
      for (let y = STEP / 2; y < h; y += STEP) {
        for (let x = STEP / 2; x < w; x += STEP) {
          const d = Math.hypot(x - cx, y - cy);
          const edge = Math.max(0, 1 - d / maxR);                       // soft vignette
          const ring = still ? 0 : Math.exp(-(((d - r) / 34) ** 2)) * strength * fade;
          const a = (0.2 + 0.7 * ring) * (0.45 + 0.55 * edge);
          ctx.fillStyle = ring > 0.15 ? `rgba(169, 205, 191, ${a})` : `rgba(239, 233, 223, ${a})`;
          ctx.beginPath(); ctx.arc(x, y, 1.3 + 1.9 * ring, 0, 6.2832); ctx.fill();
        }
      }
      if (!still && visible && !document.hidden && w > 0) raf = requestAnimationFrame(frame);
    }
    function start() { cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); }
    size(); start();
    window.addEventListener("resize", () => { size(); start(); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) start(); });
    if ("IntersectionObserver" in window)
      new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) start(); }).observe(cv);
  }

  // ======================= boot / resize / theme =======================
  function redrawAll() { readColors(); if (lab.sig) drawLab(); }
  let rz = null;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(redrawAll, 120); });
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  function boot() {
    initLab();
    coverDots();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
