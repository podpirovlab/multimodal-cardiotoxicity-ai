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
      synthName: "синтетический пример", leadNone: "не указано",
      sampleName: "twa36, PhysioNet TWADB", sampleFail: "Не удалось загрузить пример: нет связи с сайтом.",
      sampleNote: "Настоящая запись из базы PhysioNet T-Wave Alternans Challenge (из PTB), лицензия ODC-By.", version: (v) => `версия ${v}`,
      beatsWindows: (n, w) => `${n} ${plural(n, "удар", "удара", "ударов")}, ${w} ${plural(w, "окно", "окна", "окон")}`,
      posText: (r) => `Альтернация ${fmt(r.vAlt, 1)} мкВ при ${kEq(r.k)} в окне с ЧСС ${fmt(r.windowHr, 0)} уд/мин: выполнен критерий спектрального метода (не меньше 1,9 мкВ, K не меньше 3, ЧСС не выше 110, шум не выше 1,8 мкВ). В точке ST-T, где альтернация наибольшая, она достигает ${fmt(r.vPeak, 1)} мкВ — на бумажной ЭКГ это ${fmt(r.vPeak / 100, 2)} мм, глазом не увидеть.`,
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
      edfOk: (n, fs, label, fmt) => `${fmt}: канал «${label}», ${n} отсчётов, ${fs} Гц`,
      unitUnknown: (u) => ` Единицы канала «${u || "не указаны"}» не распознаны как напряжение: считаю, что значения в мВ.`,
      edfChannel: "Канал (отведение)", edfAnnotations: "(служебный канал, пропущен)",
      shortWarn: (n) => `Проанализировано ${n} ударов — меньше стандартных 128, поэтому к результату стоит относиться осторожнее.`,
      ms: "мс",
      qtBusy: "Измеряю QT по всем отведениям…",
      qtNone: "QT измеряется по файлу EDF или BDF, где не меньше 6 стандартных отведений (I, II … V6). По одному отведению наш QT на ЭКГ PTB-XL ошибался примерно на 46 мс, а по двенадцати — на 9 мс, поэтому по одному отведению QT не показывается.",
      qtFew: (n) => `В файле ${n} ${plural(n, "стандартное отведение", "стандартных отведения", "стандартных отведений")} ЭКГ с известными единицами, а для QT нужно не меньше 6: по одному отведению наш QT ошибался примерно на 46 мс. Отведение узнаётся по подписи канала, например «II» или «ECG V5».`,
      qtErr: "QT измерить не удалось: в записи не нашлось ровных ударов с заметной волной T.",
      qtWhere: (q) => q.nSegments > 1
        ? `Показан отрезок ${fmt(q.segmentStart, 0)}–${fmt(q.segmentStart + D.QT.segmentS, 0)} с — с медианным QTcF среди ${q.nSegmentsReliable ? `${q.nSegmentsReliable} ${plural(q.nSegmentsReliable, "надёжного", "надёжных", "надёжных")} десятисекундных отрезков (всего ${q.nSegments})` : `${q.nSegments} десятисекундных отрезков (надёжных нет)`}; ЧСС ${fmt(q.hr, 0)} уд/мин.`
        : `По ${q.nBeats} ${plural(q.nBeats, "удару", "ударам", "ударам")}, ЧСС ${fmt(q.hr, 0)} уд/мин.`,
      qtOk: (q, n) => `Отведения согласны: конец T найден в ${q.nLeadsUsed} из ${n}, разброс ${fmt(q.iqrMs, 0)} мс. На тестовых ЭКГ PTB-XL с такой пометкой наш QT отличался от алгоритма GE 12SL в среднем на 9 мс. Автоматический QT врач всё равно проверяет вручную.`,
      batchBusy: (k, n, name) => `Файл ${k} из ${n}: ${name}…`,
      batchDone: (n, pos, neg, ind, err) => `Готово: ${n} ${plural(n, "файл", "файла", "файлов")}. Критерий выполнен: ${pos}, альтернации нет: ${neg}, не определено: ${ind}, не удалось проанализировать: ${err}. Таблица ждёт скачивания.`,
      anonHas: (items) => `В заголовке файла есть: ${items}. Обезличенная копия заменит их на «X» и сотрёт текст аннотаций; сигнал не меняется.`,
      anonNone: "Имени и кода пациента в заголовке нет. Копия всё равно сотрёт дату и время записи и текст аннотаций.",
      anonItems: { patient: "имя или код пациента", recording: "сведения о записи (врач, аппарат)", date: "дата и время записи" },
      anonDone: (name, n) => `Сохранено: ${name}. Стёрто аннотаций с текстом: ${n}. Сама ЭКГ — тоже биометрические данные: передавайте её только тем, кому это разрешено.`,
      leadsBusy: "Считаю все отведения…",
      leadsSub: (n, capped) => `${n} ${plural(n, "отведение", "отведения", "отведений")}, общие R-пики${capped ? ", первые 10 минут записи" : ""}. Нажмите на отведение, чтобы открыть его подробно.`,
      leadsPos: (pairs) => `Альтернация есть в соседних грудных отведениях ${pairs}. Так микровольтную альтернацию определяют в клинике: в двух соседних грудных отведениях или в ортогональных отведениях X, Y, Z (FDA, 2001).`,
      leadsIsolated: (leads) => `Значимая альтернация есть только в ${leads}, но не в двух соседних грудных отведениях. По клиническому правилу этого мало: одно отведение может «качаться» из-за электрода или помехи.`,
      leadsNeg: "Ни в одном грудном отведении нет значимой альтернации, а пульс доходил до 105: по правилу двух соседних грудных отведений результат отрицательный.",
      leadsInd: (why) => `По всем отведениям сказать «да» или «нет» нельзя: в грудных отведениях чаще всего ${why}.`,
      leadsWhy: { noise: "слишком сильный шум", ectopy: "много экстрасистол или артефактов", hr_too_high: "альтернация только при пульсе выше 110", hr_too_low: "пульс не доходил до 105", not_analysable: "запись слишком короткая для альтернации" },
      leadsErr: "Отведения вместе посчитать не удалось.",
      and: (a, b) => `${a} и ${b}`, cellNone: "нет в файле",
      cellState: { pos: "критерий выполнен", neg: "альтернации нет", ind: "не определено", err: "не удалось посчитать", none: "нет в файле" },
      qtCheck: (q, n) => `Проверьте вручную: ${q.nLeadsUsed < 6 ? `конец T найден только в ${q.nLeadsUsed} из ${n} отведений` : `отведения расходятся в том, где кончается T (разброс ${fmt(q.iqrMs, 0)} мс)`}. Так бывает при плоской или двугорбой волне T, трепетании и мерцании предсердий.`,
    },
    en: {
      pos: "Alternans criterion met", neg: "No significant alternans", ind: "Indeterminate", err: "Cannot analyse",
      busy: "Analysing…", busyText: "Scanning windows across the whole recording.",
      none: "No recording loaded", noneText: "Upload a recording or open an example below.",
      synthName: "synthetic example", leadNone: "not given",
      sampleName: "twa36, PhysioNet TWADB", sampleFail: "Could not load the example: no connection to the site.",
      sampleNote: "A real recording from the PhysioNet T-Wave Alternans Challenge database (from PTB), ODC-By licence.", version: (v) => `version ${v}`,
      beatsWindows: (n, w) => `${n} beats, ${w} window${w === 1 ? "" : "s"}`,
      posText: (r) => `Alternans ${fmt(r.vAlt, 1)} µV with ${kEq(r.k)} in a window at ${fmt(r.windowHr, 0)} bpm: meets the Spectral Method criterion (at least 1.9 µV, K at least 3, heart rate at most 110, noise at most 1.8 µV). At the point of ST-T where it is largest, the alternans reaches ${fmt(r.vPeak, 1)} µV: ${fmt(r.vPeak / 100, 2)} mm on paper ECG, invisible to the eye.`,
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
      edfOk: (n, fs, label, fmt) => `${fmt}: channel "${label}", ${n} samples at ${fs} Hz`,
      unitUnknown: (u) => ` The channel's unit "${u || "none"}" is not a voltage I recognise, so the values are read as mV.`,
      edfChannel: "Channel (lead)", edfAnnotations: "(service channel, skipped)",
      shortWarn: (n) => `Analysed ${n} beats — fewer than the standard 128, so treat the result with more caution.`,
      ms: "ms",
      qtBusy: "Measuring QT on all leads…",
      qtNone: "QT is measured from an EDF or BDF file with at least 6 standard leads (I, II … V6). From one lead our QT was off by about 46 ms on PTB-XL ECGs, against 9 ms from twelve, so no single-lead QT is shown.",
      qtFew: (n) => `This file has ${n} standard ECG lead${n === 1 ? "" : "s"} with known units; QT needs at least 6: from one lead our QT was off by about 46 ms. A lead is recognised by its channel label, such as "II" or "ECG V5".`,
      qtErr: "QT could not be measured: no regular beats with a visible T wave were found.",
      qtWhere: (q) => q.nSegments > 1
        ? `Shown: the ${fmt(q.segmentStart, 0)}–${fmt(q.segmentStart + D.QT.segmentS, 0)} s segment, the one with the median QTcF of ${q.nSegmentsReliable ? `the ${q.nSegmentsReliable} reliable ten-second segments (${q.nSegments} in all)` : `${q.nSegments} ten-second segments (none reliable)`}; heart rate ${fmt(q.hr, 0)} bpm.`
        : `From ${q.nBeats} beats, heart rate ${fmt(q.hr, 0)} bpm.`,
      qtOk: (q, n) => `The leads agree: ${q.nLeadsUsed} of ${n} give a T end, spread ${fmt(q.iqrMs, 0)} ms. On PTB-XL test ECGs flagged like this, our QT differed from the GE 12SL algorithm by 9 ms on average. A doctor still checks an automatic QT by hand.`,
      batchBusy: (k, n, name) => `File ${k} of ${n}: ${name}…`,
      batchDone: (n, pos, neg, ind, err) => `Done: ${n} file${n === 1 ? "" : "s"}. Criterion met: ${pos}, no alternans: ${neg}, indeterminate: ${ind}, could not be analysed: ${err}. The table is ready to download.`,
      anonHas: (items) => `The file header holds: ${items}. The de-identified copy replaces them with "X" and clears the annotation text; the signal is unchanged.`,
      anonNone: "The header holds no patient name or code. The copy still clears the recording date and time and the annotation text.",
      anonItems: { patient: "the patient's name or code", recording: "recording details (staff, equipment)", date: "the recording date and time" },
      anonDone: (name, n) => `Saved: ${name}. Annotation texts cleared: ${n}. The ECG itself is biometric data too: share it only with those allowed to see it.`,
      leadsBusy: "Analysing every lead…",
      leadsSub: (n, capped) => `${n} leads, shared R peaks${capped ? ", the first 10 minutes" : ""}. Choose a lead to open it in detail.`,
      leadsPos: (pairs) => `Alternans in the adjacent chest leads ${pairs}. That is how microvolt alternans is defined in the clinic: in two adjacent chest leads or in the orthogonal leads X, Y, Z (FDA, 2001).`,
      leadsIsolated: (leads) => `Significant alternans only in ${leads}, not in two adjacent chest leads. Under the clinical rule that is not enough: one lead can alternate because of an electrode or interference.`,
      leadsNeg: "No chest lead shows significant alternans and the heart rate reached 105: under the two-adjacent-chest-leads rule the result is negative.",
      leadsInd: (why) => `Across all leads neither yes nor no can be said: in the chest leads the commonest reason is ${why}.`,
      leadsWhy: { noise: "too much noise", ectopy: "many ectopic beats or artefacts", hr_too_high: "alternans only above 110 bpm", hr_too_low: "a heart rate that never reached 105", not_analysable: "a recording too short for alternans" },
      leadsErr: "The leads could not be analysed together.",
      and: (a, b) => `${a} and ${b}`, cellNone: "not in file",
      cellState: { pos: "criterion met", neg: "no alternans", ind: "indeterminate", err: "could not be analysed", none: "not in file" },
      qtCheck: (q, n) => `Check by hand: ${q.nLeadsUsed < 6 ? `only ${q.nLeadsUsed} of ${n} leads give a T end` : `the leads disagree on where T ends (spread ${fmt(q.iqrMs, 0)} ms)`}. This happens with flat or notched T waves, atrial flutter and fibrillation.`,
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
  const lab = { res: null, sig: null, seed: 7, source: "none", lead: "", fileSig: null, fileFs: 500, fileName: "", edf: null, edfBuf: null, csvRawText: null, qt: null, multi: null };
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
    for (const id of ["#f-rec", "#f-beats", "#m-hr", "#f-ver", "#f-lead-out", "#m-valt", "#m-k", "#m-whr", "#m-noise", "#m-peak", "#m-mma"]) {
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
    qtSeq++; lab.qt = null; renderQT();
    setLeads(null);
    $("#b-fhir").disabled = true; $("#b-print").disabled = true;
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
  function runJob(msg, compute) {
    return new Promise((resolve, reject) => {
      const inline = () => { try { resolve(compute()); } catch (err) { reject(err); } };
      if (!worker) { inline(); return; }
      const id = ++jobSeq;
      jobs.set(id, { resolve, reject, inline });
      worker.postMessage(Object.assign({ id }, msg));
    });
  }
  const analyse = (x, fs) => runJob({ kind: "twa", x, fs }, () => D.analyzeTWA(x, fs));
  const measureQT = (segments, starts, fs) => runJob({ kind: "qt", segments, starts, fs },
    () => D.measureQTOnSegments(segments, starts, fs));

  // ---------- QT: all standard leads of an EDF file together (cardioonco/qt.py) ----------
  // QT is shown only from at least 6 standard leads: from one lead it was off by ~46 ms on PTB-XL.
  let qtSeq = 0;
  const qtOk = () => !!(lab.qt && lab.qt.state === "ok");
  function updateExport() { $("#b-fhir").disabled = !(lab.res || qtOk() || multiOk()); }
  // the standard leads (one channel each, known voltage unit) sharing the most common sampling rate
  function standardLeads(edf) {
    const byFs = new Map();
    edf.signals.forEach((s, i) => {
      const lead = EDF.leadFromLabel(s.label);
      if (s.annotations || s.unitToMv === null || !lead) return;
      if (!byFs.has(s.fs)) byFs.set(s.fs, new Map());
      if (!byFs.get(s.fs).has(lead)) byFs.get(s.fs).set(lead, i);
    });
    let best = { fs: 0, idx: [], names: [] };
    for (const [fs, m] of byFs) if (m.size > best.idx.length) best = { fs, idx: [...m.values()], names: [...m.keys()] };
    return best;
  }

  // ---------- TWA in every lead and the two-adjacent-chest-leads rule (cardioonco.twa.analyze_leads) ----------
  // Shown when the file has at least two chest leads; at most the first 10 minutes are read.
  const GRID = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"];   // printed 12-lead layout
  const MULTI_MAX_S = 600;
  let multiSeq = 0;
  const multiOk = () => !!(lab.multi && lab.multi.state === "ok");
  function setLeads(state) { multiSeq++; lab.multi = state; renderLeads(); updateExport(); }
  function startLeads(edf) {
    const g = standardLeads(edf);
    if (g.names.filter((n) => D.PRECORDIAL.includes(n)).length < 2) { setLeads(null); return; }
    const { leads, capped } = leadsInput(edf, g);
    setLeads({ state: "busy" });
    const seq = multiSeq;
    runJob({ kind: "leads", leads, names: g.names, fs: g.fs }, () => D.analyzeLeads(leads, g.names, g.fs)).then(
      (res) => { if (seq === multiSeq) { lab.multi = { state: "ok", res, names: g.names, idx: g.idx, capped }; renderLeads(); updateExport(); } },
      () => { if (seq === multiSeq) { lab.multi = { state: "err" }; renderLeads(); updateExport(); } });
  }
  // all standard leads, at most the first MULTI_MAX_S seconds
  function leadsInput(edf, g) {
    const total = edf.nRecords * edf.signals[g.idx[0]].samplesPerRecord, len = Math.min(total, Math.round(MULTI_MAX_S * g.fs));
    return { leads: g.idx.map((i) => edf.getChannel(i, 0, len)), capped: len < total };
  }
  const CELL_CLASS = { positive: "pos", negative: "neg", indeterminate: "ind" };
  function renderLeads() {
    const st = lab.multi, block = $("#leads-block");
    block.hidden = !st;
    if (!st) return;
    const grid = $("#lead-grid"), note = $("#leads-note");
    grid.innerHTML = "";
    $("#leads-sub").textContent = st.state === "ok" ? T.leadsSub(st.names.length, st.capped) : "";
    for (const name of GRID) {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.lead = name;
      const r = st.state === "ok" ? st.res.leads[name] : null;
      const cls = !r ? "none" : r.error ? "err" : CELL_CLASS[r.outcome];
      b.className = "lead-cell " + cls;
      const v = r && !r.error ? `${fmt(r.vAlt, 1)} ${T.uv}` : st.state === "busy" ? "…" : "—";
      const k = r && !r.error ? `K ${fmtK(r.k)}` : "";
      b.innerHTML = `<b></b><span class="v"></span><span class="k"></span>`;
      b.querySelector("b").textContent = name; b.querySelector(".v").textContent = v; b.querySelector(".k").textContent = k;
      b.setAttribute("aria-label", `${name}: ${v}${k ? ", " + k : ""}, ${T.cellState[cls]}`);
      b.disabled = !r || !!r.error;
      b.setAttribute("aria-pressed", String(lab.source === "file" && lab.lead === name));
      b.addEventListener("click", () => {
        const i = st.idx[st.names.indexOf(name)];
        $("#f-edf-channel").value = String(i); selectEdfChannel(i);
      });
      grid.appendChild(b);
    }
    if (st.state === "busy") { note.textContent = T.leadsBusy; return; }
    if (st.state === "err") { note.textContent = T.leadsErr; return; }
    const res = st.res, list = (xs) => xs.length > 1 ? T.and(xs.slice(0, -1).join(", "), xs[xs.length - 1]) : xs[0] || "";
    if (res.outcome === "positive") note.textContent = T.leadsPos(list(res.positivePairs.map(([a, b]) => `${a}–${b}`)));
    else if (res.reason === "isolated") note.textContent = T.leadsIsolated(list(GRID.filter((n) => res.leads[n] && res.leads[n].outcome === "positive")));
    else if (res.outcome === "negative") note.textContent = T.leadsNeg;
    else note.textContent = T.leadsInd(T.leadsWhy[res.reason] || res.reason);
  }
  function setQT(state) { qtSeq++; lab.qt = state; renderQT(); updateExport(); }
  function startQT(edf) {
    const g = standardLeads(edf);
    if (g.idx.length < D.QT.minLeads) { setQT({ state: "few", n: g.idx.length }); return; }
    setQT({ state: "busy" });
    const seq = qtSeq;
    const { segments, starts } = qtInput(edf, g);
    measureQT(segments, starts, g.fs).then(
      (q) => { if (seq === qtSeq) { lab.qt = { state: "ok", q, n: g.idx.length }; renderQT(); updateExport(); } },
      () => { if (seq === qtSeq) { lab.qt = { state: "err" }; renderQT(); updateExport(); } });
  }
  // only the 10-s segments that are measured are read from the file (at most 60 of them)
  function qtInput(edf, g) {
    const total = edf.nRecords * edf.signals[g.idx[0]].samplesPerRecord;
    const starts = D.qtSegmentStarts(total, g.fs);
    const len = starts.length === 1 ? total : Math.round(D.QT.segmentS * g.fs);
    return { starts, segments: starts.map((s) => g.idx.map((i) => edf.getChannel(i, s, s + len))) };
  }
  function renderQT() {
    const st = lab.qt, cells = [["#m-qt", "qtMs"], ["#m-qtcf", "qtcfMs"], ["#m-qtcb", "qtcbMs"]];
    for (const [id] of cells) { $(id).textContent = st && st.state === "busy" ? "…" : "—"; $(id).classList.remove("met", "warn"); }
    const note = $("#qt-note");
    if (!st || st.state === "none") { note.textContent = T.qtNone; return; }
    if (st.state === "busy") { note.textContent = T.qtBusy; return; }
    if (st.state === "few") { note.textContent = T.qtFew(st.n); return; }
    if (st.state === "err") { note.textContent = T.qtErr; return; }
    const q = st.q;
    for (const [id, k] of cells) {
      $(id).innerHTML = `${fmt(q[k], 0)} <small>${T.ms}</small>`;
      $(id).classList.toggle("warn", !q.reliable);
    }
    // red marks a finding only: a reliable QTcF above the ESC 2022 definition of prolonged
    $("#m-qtcf").classList.toggle("met", q.reliable && q.qtcfMs > 500);
    note.textContent = T.qtWhere(q) + " " + (q.reliable ? T.qtOk(q, st.n) : T.qtCheck(q, st.n));
  }

  function runLab() {
    if (lab.source === "none") { showEmpty(); return; }
    let x, fs;
    if (lab.source === "file" && lab.fileSig) { x = lab.fileSig; fs = lab.fileFs; }
    else { const s = D.synth(Object.assign(params(), { seed: lab.seed })); x = s.x; fs = s.fs; }
    const box = $("#verdict"), run = ++runSeq;
    const record = lab.source === "file" ? lab.fileName : T.synthName;
    if (lab.source !== "file" && lab.qt) setQT(null);         // the synthetic example is one lead
    if (lab.source !== "file" && lab.multi) setLeads(null);
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
      const lead = lab.source === "file" ? lab.lead : "";
      if (lab.source === "file") leadField(lead); else $("#f-lead-out").textContent = "—";
      if (lead) linkHeart(lead); else if (heart3d.fromRecording) linkHeart("");
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
      $("#b-print").disabled = false; updateExport();
      drawLab();
      if (lab.animate) { lab.animate = false; animateResult(); }
    }, (e) => {
      if (run !== runSeq) return;
      lab.sig = { x, fs }; lab.res = null; lab.animate = false;
      box.className = "verdict err"; box.querySelector(".pill").textContent = T.err;
      clearMetrics();
      box.querySelector("p").textContent = errorText(e);
      $("#f-rec").textContent = record;
      $("#b-print").disabled = true; updateExport();
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
  // EDF starts with "0       ", BDF with 0xFF "BIOSEMI": decide by content, not by the file name
  function isEdfBytes(head) {
    const ascii = String.fromCharCode(...head.slice(0, 8));
    return ascii === "0       " || (head[0] === 0xff && ascii.slice(1) === "BIOSEMI");
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
    sel.value = String(guessChannel(edf));
    selectEdfChannel(parseInt(sel.value, 10));
  }
  // the channel analysed by default: the first labelled as ECG or lead II, else the first signal
  function guessChannel(edf) {
    const guess = edf.signals.findIndex((s) => !s.annotations && /ecg|ekg|\bii\b|lead/i.test(s.label));
    const first = edf.signals.findIndex((s) => !s.annotations);
    return guess >= 0 ? guess : Math.max(0, first);
  }
  function selectEdfChannel(idx) {
    const edf = lab.edf; if (!edf) return;
    const s = edf.signals[idx];
    const sig = edf.getChannel(idx);
    $("#f-status").textContent = T.edfOk(sig.length, Math.round(s.fs), s.label, edf.format) + (s.unitToMv === null ? T.unitUnknown(s.unit) : "");
    lab.lead = EDF.leadFromLabel(s.label); $("#f-lead").value = lab.lead;
    document.querySelectorAll(".lead-cell").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lead === lab.lead)));
    loadFromChannel(sig, s.fs, lab.fileName);
  }
  // ---------- batch: many files -> one CSV table (the same columns as scripts/batch_analyse.py) ----------
  const BATCH_COLS = ["file", "format", "fs_hz", "duration_s", "n_leads", "lead", "n_beats", "heart_rate_bpm",
    "v_alt_uv", "k_score", "noise_uv", "twa_outcome", "twa_reason", "all_leads_outcome", "all_leads_reason",
    "positive_pairs", "qt_ms", "qtcf_ms", "qtcb_ms", "qt_reliable", "error"];
  let batchRows = [];
  const num = (v, d) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "");
  async function batchRow(file) {
    const row = { file: file.name };
    const buf = await file.arrayBuffer();
    const twa = async (x, fs) => {
      try {
        const r = await analyse(x, fs);
        Object.assign(row, { n_beats: r.nBeats, heart_rate_bpm: num(r.hr, 1), v_alt_uv: num(r.vAlt, 2), k_score: num(r.k, 2),
          noise_uv: num(r.noise, 2), twa_outcome: r.outcome, twa_reason: r.reason });
        return true;
      } catch (e) { Object.assign(row, { twa_outcome: "not_analysed", twa_reason: e.code || "error", error: e.message }); return false; }
    };
    if (!isEdfBytes(new Uint8Array(buf, 0, Math.min(8, buf.byteLength)))) {
      const col = Math.max(0, parseInt($("#f-col").value || "0", 10)), fs = Math.max(50, parseFloat($("#f-fs").value) || 500);
      const x = EDF.parseCSV(new TextDecoder().decode(buf), col);
      if ($("#f-unit").value === "uv") for (let i = 0; i < x.length; i++) x[i] /= 1000;
      Object.assign(row, { format: "CSV", fs_hz: num(fs, 1), duration_s: num(x.length / fs, 1), n_leads: 1, lead: `column ${col}` });
      await twa(x, fs);
      return row;
    }
    const edf = EDF.parseEDF(buf), ci = guessChannel(edf), s = edf.signals[ci];
    Object.assign(row, { format: edf.format, fs_hz: num(s.fs, 1), duration_s: num(edf.nRecords * edf.recordDuration, 1),
      n_leads: edf.signals.filter((q) => !q.annotations).length, lead: s.label });
    const ok = await twa(edf.getChannel(ci), s.fs);
    const g = standardLeads(edf);
    if (ok && g.names.filter((n) => D.PRECORDIAL.includes(n)).length >= 2) {
      const { leads } = leadsInput(edf, g);
      const m = await runJob({ kind: "leads", leads, names: g.names, fs: g.fs }, () => D.analyzeLeads(leads, g.names, g.fs));
      Object.assign(row, { all_leads_outcome: m.outcome, all_leads_reason: m.reason,
        positive_pairs: m.positivePairs.map((p) => p.join("-")).join(" ") });
    }
    if (g.idx.length >= D.QT.minLeads) {
      try {
        const { segments, starts } = qtInput(edf, g);
        const q = await measureQT(segments, starts, g.fs);
        Object.assign(row, { qt_ms: num(q.qtMs, 1), qtcf_ms: num(q.qtcfMs, 1), qtcb_ms: num(q.qtcbMs, 1), qt_reliable: q.reliable ? 1 : 0 });
      } catch (e) { /* QT left empty */ }
    }
    return row;
  }
  async function runBatch(files) {
    const btn = $("#b-batch-csv"), status = $("#f-batch-status");
    btn.disabled = true; batchRows = [];
    for (const [k, file] of files.entries()) {
      status.textContent = T.batchBusy(k + 1, files.length, file.name);
      try { batchRows.push(await batchRow(file)); }
      catch (e) { batchRows.push({ file: file.name, twa_outcome: "not_analysed", error: e.message }); }
    }
    const count = (o) => batchRows.filter((r) => r.twa_outcome === o).length;
    status.textContent = T.batchDone(batchRows.length, count("positive"), count("negative"), count("indeterminate"), count("not_analysed"));
    btn.disabled = !batchRows.length;
  }
  function downloadBatch() {
    const cell = (v) => { const t = v === undefined || v === null ? "" : String(v); return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t; };
    const text = [BATCH_COLS.join(","), ...batchRows.map((r) => BATCH_COLS.map((c) => cell(r[c])).join(","))].join("\n") + "\n";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" })); a.download = "cardioonco_batch.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }

  // ---------- de-identified copy of the loaded EDF/BDF (cardioonco/edf.py) ----------
  // The page lists what kinds of personal data the header holds without showing them, so that
  // a printed or shared screen never carries a name.
  function describeIdentity(buf) {
    const b = new Uint8Array(buf, 0, 184), txt = (o, n) => String.fromCharCode(...b.subarray(o, o + n)).trim();
    const items = [];
    if (!/^[X\s]*$/.test(txt(8, 80))) items.push(T.anonItems.patient);
    if (!/^(Startdate)?[X\s]*$/.test(txt(88, 80))) items.push(T.anonItems.recording);
    if (txt(168, 8) !== "01.01.85" || txt(176, 8) !== "00.00.00") items.push(T.anonItems.date);
    $("#f-anon-text").textContent = items.some((i) => i !== T.anonItems.date) ? T.anonHas(items.join(", ")) : T.anonNone;
  }
  function downloadAnonymized() {
    if (!lab.edfBuf) return;
    const { buffer, removed } = EDF.anonymizeEDF(lab.edfBuf);
    const name = (lab.fileName || "recording.edf").replace(/(\.[^.]+)?$/, (ext) => "_anon" + (ext || ".edf"));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([buffer], { type: "application/octet-stream" })); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    $("#f-anon-text").textContent = T.anonDone(name, removed.annotations);
  }
  function parseCsvText(text) {
    const col = Math.max(0, parseInt($("#f-col").value || "0", 10));
    const unit = $("#f-unit").value, fs = Math.max(50, parseFloat($("#f-fs").value) || 500);
    const vals = EDF.parseCSV(text, col);
    lab.lead = $("#f-lead").value;
    if (unit === "uv") for (let i = 0; i < vals.length; i++) vals[i] /= 1000;
    const status = $("#f-status");
    if (vals.length < fs * 10) { status.textContent = T.fileErr; return; }
    status.textContent = T.fileOk(vals.length, fs);
    loadFromChannel(vals, fs, lab.fileName);
  }
  function openEdf(buf, name) {
    lab.fileName = name; lab.csvRawText = null;
    try {
      const edf = EDF.parseEDF(buf);
      lab.edf = edf; lab.edfBuf = buf;
      describeIdentity(buf);
      $("#f-csv-wrap").hidden = true; $("#f-edf-wrap").hidden = false;
      populateEdfChannels(edf);           // the TWA job goes to the worker first, then all leads, then QT
      startLeads(edf);
      startQT(edf);
    } catch (e) {
      $("#f-status").textContent = T.fileErrEdf(e.message);
    }
  }
  function openFile(file) {
    if (!file) return;
    file.arrayBuffer().then((buf) => {
      if (isEdfBytes(new Uint8Array(buf, 0, Math.min(8, buf.byteLength)))) { openEdf(buf, file.name); return; }
      lab.fileName = file.name;
      $("#f-csv-wrap").hidden = false; $("#f-edf-wrap").hidden = true; lab.edf = null; lab.edfBuf = null;
      setQT({ state: "none" }); setLeads(null);
      lab.csvRawText = new TextDecoder().decode(buf);
      parseCsvText(lab.csvRawText);
    });
  }
  function onFile(ev) { openFile(ev.target.files && ev.target.files[0]); }
  // a real 12-lead recording from PhysioNet, one click away (assets/samples/README.md)
  function openSample() {
    const b = $("#b-sample"); b.disabled = true;
    fetch("assets/samples/twadb-twa36.edf").then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
      .then((buf) => { openEdf(buf, T.sampleName); $("#f-status").textContent += " " + T.sampleNote; })
      .catch(() => { $("#f-status").textContent = T.sampleFail; })
      .finally(() => { b.disabled = false; });
  }
  // dropping a file anywhere on the upload block opens it
  function initDrop() {
    const zone = $(".file"); let depth = 0;
    zone.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; zone.classList.add("dragover"); });
    zone.addEventListener("dragover", (e) => e.preventDefault());
    zone.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; zone.classList.remove("dragover"); } });
    zone.addEventListener("drop", (e) => {
      e.preventDefault(); depth = 0; zone.classList.remove("dragover");
      openFile(e.dataTransfer.files && e.dataTransfer.files[0]);
    });
  }

  // ---------- research JSON in FHIR R4 format (mirrors cardioonco/fhir.py) ----------
  // Every Observation points to a contained Device carrying the algorithm version, so a number
  // can always be traced to the code that produced it. The recording time is unknown here, so
  // effective[x] is left out rather than filled with the export time.
  const FHIR_CS = "https://github.com/podpirovlab/multimodal-cardiotoxicity-ai/fhir/CodeSystem/cardioonco";
  const UCUM = "http://unitsofmeasure.org";
  const LOINC = "http://loinc.org";
  // loinc: [code, display], the display exactly as LOINC names the code (checked on tx.fhir.org)
  function fhirObs(id, code, display, value, unit, loinc) {
    const coding = [{ system: FHIR_CS, code, display }];
    if (loinc) coding.unshift({ system: LOINC, code: loinc[0], display: loinc[1] });
    return { resourceType: "Observation", id, status: "preliminary",
      code: { coding, text: display }, device: { reference: "#software" },
      valueQuantity: { value: +value.toFixed(4), unit, system: UCUM, code: unit } };
  }
  function exportFHIR() {
    const r = lab.res, q = qtOk() ? lab.qt.q : null, m = multiOk() ? lab.multi.res : null;
    if (!r && !q && !m) return;
    const observations = r ? [
      fhirObs("hr", "heart-rate", "Heart rate (from R-R intervals)", r.hr, "/min", ["8867-4", "Heart rate"]),
      fhirObs("valt", "twa-valt", "T-wave alternans voltage (Spectral Method)", r.vAlt, "uV"),
      fhirObs("kscore", "twa-k", "T-wave alternans K-score", r.k, "1"),
      fhirObs("mma", "twa-mma", "T-wave alternans (Modified Moving Average)", r.mma, "uV"),
      { resourceType: "Observation", id: "outcome", status: "preliminary",
        code: { coding: [{ system: FHIR_CS, code: "twa-outcome", display: "T-wave alternans outcome (Spectral Method rules)" }],
                text: "T-wave alternans outcome (Spectral Method rules)" },
        device: { reference: "#software" },
        valueCodeableConcept: { coding: [{ system: FHIR_CS, code: r.outcome, display: r.outcome }], text: r.reason } },
    ] : [];
    if (m) observations.push({ resourceType: "Observation", id: "twa-leads", status: "preliminary",   // mirrors cardioonco/fhir.py
      code: { coding: [{ system: FHIR_CS, code: "twa-multilead-outcome", display: "T-wave alternans outcome, two adjacent precordial leads rule" }],
              text: "T-wave alternans outcome, two adjacent precordial leads rule" },
      device: { reference: "#software" },
      valueCodeableConcept: { coding: [{ system: FHIR_CS, code: m.outcome, display: m.outcome }],
                              text: m.reason + (m.positivePairs.length ? ": " + m.positivePairs.map((p) => p.join("-")).join(", ") : "") } });
    if (q) observations.push(        // mirrors cardioonco/fhir.py
      fhirObs("qt", "qt-interval", "QT interval (tangent method, 12SL scale)", q.qtMs, "ms", ["8634-8", "Q-T interval"]),
      fhirObs("qtcf", "qtc-fridericia", "QTc, Fridericia correction", q.qtcfMs, "ms", ["8636-3", "Q-T interval corrected"]),
      fhirObs("qtcb", "qtc-bazett", "QTc, Bazett correction", q.qtcbMs, "ms"),
      { resourceType: "Observation", id: "qt-reliability", status: "preliminary",
        code: { coding: [{ system: FHIR_CS, code: "qt-reliability", display: "QT measurement reliability (leads agree on the T end)" }],
                text: "QT measurement reliability (leads agree on the T end)" },
        device: { reference: "#software" },
        valueCodeableConcept: { coding: [{ system: FHIR_CS, code: q.reliable ? "reliable" : "check-by-hand", display: q.reliable ? "reliable" : "check-by-hand" }],
                                text: `${q.nLeadsUsed} leads, T-end spread ${Math.round(q.iqrMs)} ms` } });
    const device = { resourceType: "Device", id: "software",
      deviceName: [{ name: "CardioOncoPredict ECG analysis (research software)", type: "model-name" }],
      version: [{ value: D.version }] };
    const report = {
      resourceType: "DiagnosticReport", id: crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2),
      meta: { tag: [{ system: FHIR_CS, code: "research-only", display: "Research prototype output - not for clinical use" }] },
      contained: [device, ...observations],
      status: "preliminary",
      category: [{ coding: [{ system: "http://terminology.hl7.org/CodeSystem/v2-0074", code: "EC", display: "Electrocardiac (e.g., EKG, EEC, Holter)" }] }],
      code: { coding: [{ system: LOINC, code: "11524-6", display: "EKG study" }], text: "CardioOncoPredict ECG analysis (research prototype)" },
      issued: new Date().toISOString(),
      result: observations.map((o) => ({ reference: `#${o.id}` })),
      conclusion: "RESEARCH USE ONLY — NOT FOR CLINICAL RECORDS. " + [r ? verdictText(r) : "",
        q ? `QT ${Math.round(q.qtMs)} ms, QTcF ${Math.round(q.qtcfMs)} ms (${q.reliable ? "leads agree" : "check by hand"}).` : ""].filter(Boolean).join(" "),
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
    $("#b-anon").addEventListener("click", downloadAnonymized);
    $("#f-batch").addEventListener("change", (e) => { const fl = Array.from(e.target.files || []); if (fl.length) runBatch(fl); });
    $("#b-batch-csv").addEventListener("click", downloadBatch);
    $("#f-lead").addEventListener("change", (e) => {
      lab.lead = e.target.value;
      if (lab.res && lab.source === "file") { leadField(lab.lead); linkHeart(lab.lead); }
    });
    $("#b-fhir").addEventListener("click", exportFHIR);
    $("#b-print").addEventListener("click", () => window.print());
    $("#b-sample").addEventListener("click", openSample);
    initDrop();
    // opening the example panel is an explicit request to see a synthetic result
    $(".demo").addEventListener("toggle", (e) => {
      if (e.target.open && lab.source === "none") { lab.source = "synth"; lab.animate = true; schedule(); }
    });
    syncOutputs(); showEmpty();
  }
  function clearPresets() { document.querySelectorAll("[data-preset]").forEach((b) => b.setAttribute("aria-pressed", "false")); }

  // ======================= 3D anatomy: which wall each lead sees =======================
  // Body frame: x = patient's left, y = up, z = anterior. The ventricles are drawn as shaded
  // half-ellipsoids along the anatomical long axis (apex down, left and forward). Selecting a
  // lead lights the left-ventricular wall it faces and draws its axis; "whole LV" lights the
  // entire left-ventricular myocardium, because anthracycline injury is diffuse. Nothing here
  // comes from a recording: it is ECG anatomy, not a map of damage.
  const HEART_TEXT = {
    ru: {
      idle: "Нажмите на отведение: подсветится стенка левого желудочка, на которую оно смотрит, и его ось.",
      lv: "Левый желудочек — главная мишень антрациклинов. Они повреждают его миокард диффузно, по всей стенке, а не в одном месте, поэтому подсвечен весь желудочек. По ЭКГ из одного отведения место повреждения определить нельзя, и этот инструмент ничего не локализует.",
      wall: { septal: "межжелудочковую перегородку", anterior: "переднюю стенку", lateral: "боковую стенку", inferior: "нижнюю стенку" },
      sees: (lead, wall, axis) => `${lead} смотрит на ${wall} левого желудочка (ось ${axis}). Это анатомия ЭКГ, а не карта повреждения.`,
      aVR: "aVR смотрит в полость сердца справа и сверху и ни одной стенки не локализует, поэтому подсветки нет.",
      wallShort: { septal: "перегородка", anterior: "передняя стенка", lateral: "боковая стенка", inferior: "нижняя стенка" },
      fromRecording: (lead, wall) => `Загруженная запись сделана в отведении ${lead}. Оно смотрит на ${wall} левого желудочка, поэтому альтернация измерена «со стороны» этой стенки. Это не место повреждения: по одному отведению его определить нельзя.`,
      fromRecordingAVR: "Загруженная запись сделана в отведении aVR. Оно смотрит в полость сердца и ни одной стенки не локализует.",
      LV: "ЛЖ", RV: "ПЖ", apex: "верхушка", base: "основание",
    },
    en: {
      idle: "Choose a lead to light the left-ventricular wall it faces and draw its axis.",
      lv: "The left ventricle is the main target of anthracyclines. They injure its muscle diffusely, across the whole wall rather than in one place, so the whole ventricle is lit. A single-lead ECG cannot tell where the injury is, and this tool does not try to localise anything.",
      wall: { septal: "septum", anterior: "anterior wall", lateral: "lateral wall", inferior: "inferior wall" },
      sees: (lead, wall, axis) => `${lead} faces the ${wall} of the left ventricle (axis ${axis}). This is ECG anatomy, not a map of damage.`,
      aVR: "aVR looks into the cavity from the upper right and does not localise any wall, so nothing is lit.",
      wallShort: { septal: "septum", anterior: "anterior wall", lateral: "lateral wall", inferior: "inferior wall" },
      fromRecording: (lead, wall) => `The uploaded recording is lead ${lead}. It faces the ${wall} of the left ventricle, so the alternans was measured from that side. That is not where any damage is: one lead cannot tell.`,
      fromRecordingAVR: "The uploaded recording is lead aVR. It looks into the cavity and does not localise any wall.",
      LV: "LV", RV: "RV", apex: "apex", base: "base",
    },
  }[LANG === "ru" ? "ru" : "en"];
  // lead axes: frontal plane (hexaxial, 0° = patient's left, +90° = down) and horizontal plane (0° = left, 90° = anterior)
  const LEAD_AXES = {
    I: ["f", 0], II: ["f", 60], III: ["f", 120], aVR: ["f", -150], aVL: ["f", -30], aVF: ["f", 90],
    V1: ["h", 120], V2: ["h", 90], V3: ["h", 75], V4: ["h", 60], V5: ["h", 30], V6: ["h", 0],
  };
  const LEAD_WALL = { I: "lateral", aVL: "lateral", V5: "lateral", V6: "lateral", II: "inferior", III: "inferior", aVF: "inferior",
    V1: "septal", V2: "septal", V3: "anterior", V4: "anterior", aVR: null };
  const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], vsc = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const vnorm = (a) => vsc(a, 1 / Math.hypot(a[0], a[1], a[2]));
  function leadDir(name) {
    const [plane, deg] = LEAD_AXES[name], t = (deg * Math.PI) / 180;
    return plane === "f" ? [Math.cos(t), -Math.sin(t), 0] : [Math.cos(t), 0, Math.sin(t)];
  }
  // Primitives are built once in the body frame as polygons with outward normals; each frame
  // they are rotated to the view, back faces are culled and the rest painted far to near.
  function heartModel() {
    const axis = vnorm([0.55, -0.62, 0.5]);                       // base -> apex
    const perp = (d) => vnorm(vadd(d, vsc(axis, -vdot(d, axis))));
    const ant = perp([0, 0, 1]), side = vcross(axis, ant);
    const dirs = { anterior: ant, lateral: perp([1, 0, 0]), septal: perp([-1, 0, 0]), inferior: perp([0, -1, 0]) };
    const polys = [];
    // half-ellipsoid with its equator (base) at c and its tip at c + ra*e1
    function halfEllipsoid(c, e1, e2, e3, ra, rb, rc, tag, beats) {
      const NP = 16, NT = 36, V = [];
      for (let i = 0; i <= NP; i++) {
        const ph = (i / NP) * (Math.PI / 2), row = [];
        for (let k = 0; k <= NT; k++) {
          const th = (k / NT) * 2 * Math.PI, sp = Math.sin(ph), cp = Math.cos(ph);
          const radial = vadd(vsc(e2, Math.cos(th)), vsc(e3, Math.sin(th)));
          const pt = vadd(vadd(c, vsc(e1, ra * cp)), vadd(vsc(e2, rb * sp * Math.cos(th)), vsc(e3, rc * sp * Math.sin(th))));
          const n = vnorm(vadd(vsc(e1, cp / ra), vadd(vsc(e2, sp * Math.cos(th) / rb), vsc(e3, sp * Math.sin(th) / rc))));
          let wall = null;
          if (tag === "lv") {
            if (ph < 0.42) wall = "apex";
            else { let best = -2; for (const [w, d] of Object.entries(dirs)) { const q = vdot(radial, d); if (q > best) { best = q; wall = w; } } }
          }
          row.push({ p: pt, n, wall });
        }
        V.push(row);
      }
      for (let i = 0; i < NP; i++) for (let k = 0; k < NT; k++) {
        const q = [V[i][k], V[i][k + 1], V[i + 1][k + 1], V[i + 1][k]];
        polys.push({ pts: q.map((v) => v.p), n: vnorm(q.reduce((acc, v) => vadd(acc, v.n), [0, 0, 0])), tag, wall: V[i + 1][k].wall, beats });
      }
      // valve plane: a lid over the base so the shell reads as solid from every side
      const ring = V[NP], lid = vsc(e1, -1);
      for (let k = 0; k < NT; k++) polys.push({ pts: [c, ring[k].p, ring[k + 1].p], n: lid, tag: "lid", beats });
    }
    function ellipsoid(c, e1, e2, e3, ra, rb, rc, tag) {
      halfEllipsoid(c, e1, e2, e3, ra, rb, rc, tag, false);
      halfEllipsoid(c, vsc(e1, -1), e3, e2, ra, rc, rb, tag, false);
    }
    function tube(p0, p1, r, tag) {
      const d = vnorm(vadd(p1, vsc(p0, -1))), u = vnorm(vcross(d, Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])), v = vcross(d, u), N = 20;
      for (let k = 0; k < N; k++) {
        const t0 = (k / N) * 2 * Math.PI, t1 = ((k + 1) / N) * 2 * Math.PI;
        const r0 = vadd(vsc(u, Math.cos(t0)), vsc(v, Math.sin(t0))), r1 = vadd(vsc(u, Math.cos(t1)), vsc(v, Math.sin(t1)));
        polys.push({ pts: [vadd(p0, vsc(r0, r)), vadd(p0, vsc(r1, r)), vadd(p1, vsc(r1, r)), vadd(p1, vsc(r0, r))], n: vnorm(vadd(r0, r1)), tag, beats: false });
      }
      const cap = [];
      for (let k = 0; k < N; k++) cap.push(vadd(p1, vsc(vadd(vsc(u, Math.cos((k / N) * 2 * Math.PI)), vsc(v, Math.sin((k / N) * 2 * Math.PI))), r)));
      for (let k = 0; k < N; k++) polys.push({ pts: [p1, cap[k], cap[(k + 1) % N]], n: d, tag, beats: false });
    }
    const base = [-0.2, 0.5, -0.15];
    halfEllipsoid(base, axis, ant, side, 1.7, 0.62, 0.62, "lv", true);
    // right ventricle: wraps the septal-anterior side, flatter and shorter than the left
    const rvBase = vadd(vadd(base, vsc(dirs.septal, 0.66)), vsc(ant, 0.16));
    halfEllipsoid(rvBase, axis, ant, side, 1.2, 0.55, 0.42, "rv", true);
    // atria behind and above the base, then the great arteries
    const up = [0, 1, 0], back = [0, 0, -1], right = [-1, 0, 0];
    ellipsoid(vadd(base, [0.18, 0.42, -0.42]), up, [1, 0, 0], back, 0.42, 0.5, 0.36, "atrium");
    ellipsoid(vadd(base, [-0.62, 0.35, -0.12]), up, right, [0, 0, 1], 0.45, 0.4, 0.38, "atrium");
    tube(vadd(base, [-0.12, 0.3, 0.12]), vadd(base, [-0.05, 1.25, 0.08]), 0.2, "aorta");
    tube(vadd(base, [-0.35, 0.25, 0.42]), vadd(base, [0.15, 1.05, 0.2]), 0.18, "pulm");
    return { axis, base, polys, centre: vadd(base, vsc(axis, 0.75)) };
  }
  const heart3d = { yaw: -0.35, pitch: 0.12, target: null, sel: null, lv: false, drag: false, model: null, still: false };
  // view from just beside the electrode, so the wall it faces is in front and its axis stays visible
  function leadView(name) {
    const d = leadDir(name), yaw = Math.atan2(-d[0], d[2]) + 0.55;
    const z1 = -d[0] * Math.sin(yaw) + d[2] * Math.cos(yaw);
    return { yaw, pitch: 0.55 * Math.atan2(d[1], Math.max(0.2, z1)) };
  }
  function drawHeart(nowMs) {
    const cv = $("#c-heart"); if (!cv || !heart3d.model) return;
    const { ctx, w, h } = setupCanvas(cv);
    ctx.fillStyle = C.surface; ctx.fillRect(0, 0, w, h);
    const m = heart3d.model, phase = ((nowMs || 0) / 1000) % 1;
    const beat = heart3d.still ? 0 : Math.exp(-(((phase - 0.18) / 0.09) ** 2)), squeeze = 1 - 0.035 * beat;
    const cy = Math.cos(heart3d.yaw), sy = Math.sin(heart3d.yaw), cp = Math.cos(heart3d.pitch), sp = Math.sin(heart3d.pitch);
    const rot = (v) => { const x1 = v[0] * cy + v[2] * sy, z1 = -v[0] * sy + v[2] * cy; return [x1, v[1] * cp - z1 * sp, v[1] * sp + z1 * cp]; };
    const scale = Math.min(w, h) * 0.27, cx0 = w / 2, cy0 = h / 2;
    const proj = (p0, beats) => {
      const p = beats ? vadd(m.centre, vsc(vadd(p0, vsc(m.centre, -1)), squeeze)) : p0;
      const r = rot(vadd(p, vsc(m.centre, -1))), f = 1 / (1 - r[2] * 0.1);
      return { X: cx0 + r[0] * scale * f, Y: cy0 - r[1] * scale * f, Z: r[2] };
    };
    const light = vnorm([-0.45, 0.55, 0.7]);
    const wall = heart3d.sel ? LEAD_WALL[heart3d.sel] : null;
    const COL = { lv: [207, 154, 166], rv: [143, 152, 182], lid: [120, 96, 108], atrium: [150, 128, 140], aorta: [196, 140, 152], pulm: [128, 138, 170] };
    const SEL = [169, 205, 191], LVON = [230, 211, 167];
    const draw = [];
    for (const poly of m.polys) {
      const nv = rot(poly.n);
      if (nv[2] < -0.02) continue;                                  // back face
      const pts = poly.pts.map((q) => proj(q, poly.beats));
      let col = COL[poly.tag], alpha = 1;
      if (poly.tag === "lv" && heart3d.lv) col = LVON;
      if (poly.tag === "lv" && wall && (poly.wall === wall || (poly.wall === "apex" && (heart3d.sel === "V4" || heart3d.sel === "V5")))) col = SEL;
      if (wall && poly.tag !== "lv") alpha = poly.tag === "rv" ? 0.28 : 0.5;
      if (heart3d.lv && poly.tag !== "lv") alpha = 0.55;
      const k = 0.34 + 0.66 * Math.max(0, vdot(nv, light));
      draw.push({ pts, z: pts.reduce((acc, q) => acc + q.Z, 0) / pts.length, fill: `rgba(${Math.round(col[0] * k)},${Math.round(col[1] * k)},${Math.round(col[2] * k)},${alpha})`, alpha });
    }
    draw.sort((a, b) => a.z - b.z);
    for (const d of draw) {
      ctx.fillStyle = d.fill; ctx.beginPath(); ctx.moveTo(d.pts[0].X, d.pts[0].Y);
      for (let i = 1; i < d.pts.length; i++) ctx.lineTo(d.pts[i].X, d.pts[i].Y);
      ctx.closePath(); ctx.fill();
      if (d.alpha === 1) { ctx.strokeStyle = d.fill; ctx.lineWidth = 0.7; ctx.stroke(); }   // hide seams
    }
    if (heart3d.sel) {
      const d = leadDir(heart3d.sel), A = proj(vadd(m.centre, vsc(d, -1.9))), B = proj(vadd(m.centre, vsc(d, 1.9)));
      ctx.save(); ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.9; ctx.lineWidth = 1.4; ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(A.X, A.Y); ctx.lineTo(B.X, B.Y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(B.X, B.Y, 6, 0, 6.2832); ctx.fill();
      ctx.font = "700 14px 'PT Sans', sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(heart3d.sel, B.X, B.Y - 10); ctx.restore();
    }
    ctx.font = "13px 'PT Sans', sans-serif"; ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const apex = proj(vadd(m.base, vsc(m.axis, 1.95)), true);
    ctx.fillText(HEART_TEXT.apex, apex.X, apex.Y + 4);
  }
  function aimHeart() {
    heart3d.target = heart3d.sel ? leadView(heart3d.sel) : null;
    if (heart3d.still && heart3d.target) { heart3d.yaw = heart3d.target.yaw; heart3d.pitch = heart3d.target.pitch; }
  }
  function setHeartNote() {
    const note = $("#heart-note"); if (!note) return;
    if (heart3d.fromRecording && heart3d.sel)
      note.textContent = heart3d.sel === "aVR" ? HEART_TEXT.fromRecordingAVR : HEART_TEXT.fromRecording(heart3d.sel, HEART_TEXT.wall[LEAD_WALL[heart3d.sel]]);
    else if (heart3d.lv) note.textContent = HEART_TEXT.lv;
    else if (!heart3d.sel) note.textContent = HEART_TEXT.idle;
    else if (heart3d.sel === "aVR") note.textContent = HEART_TEXT.aVR;
    else {
      const [plane, deg] = LEAD_AXES[heart3d.sel];
      const axis = `${deg}° ${plane === "f" ? (LANG === "ru" ? "во фронтальной плоскости" : "in the frontal plane") : (LANG === "ru" ? "в горизонтальной плоскости" : "in the horizontal plane")}`;
      note.textContent = HEART_TEXT.sees(heart3d.sel, HEART_TEXT.wall[LEAD_WALL[heart3d.sel]], axis);
    }
  }
  // A recording with a known lead turns the model to that lead's side; choosing anything by hand
  // on the model takes over again.
  function pressLead(lead) {
    document.querySelectorAll("[data-lead]").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.lead === lead)));
    const lv = $("#b-lv"); if (lv) lv.setAttribute("aria-pressed", "false");
  }
  function linkHeart(lead) {
    if (!heart3d.model) return;
    heart3d.sel = lead || null; heart3d.lv = false; heart3d.fromRecording = !!lead;
    pressLead(heart3d.sel); aimHeart(); setHeartNote(); drawHeart(performance.now());
  }
  function leadField(lead) {
    const dd = $("#f-lead-out");
    if (!lead) { dd.textContent = T.leadNone; return; }
    const wall = LEAD_WALL[lead];
    dd.innerHTML = `<a href="#anatomy">${lead}${wall ? ", " + HEART_TEXT.wallShort[wall] : ""}</a>`;
  }
  function initHeart() {
    const cv = $("#c-heart"); if (!cv) return;
    heart3d.model = heartModel();
    heart3d.still = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    const stage = $("#heart-stage");
    let px = 0, py = 0, visible = true, raf = 0, last = performance.now();
    stage.addEventListener("pointerdown", (e) => { heart3d.drag = true; px = e.clientX; py = e.clientY; stage.setPointerCapture(e.pointerId); });
    stage.addEventListener("pointermove", (e) => {
      if (!heart3d.drag) return;
      heart3d.yaw += (e.clientX - px) * 0.01; heart3d.pitch = Math.max(-1.2, Math.min(1.2, heart3d.pitch + (e.clientY - py) * 0.01));
      px = e.clientX; py = e.clientY; if (heart3d.still) drawHeart(0);
    });
    const end = () => (heart3d.drag = false);
    stage.addEventListener("pointerup", end); stage.addEventListener("pointercancel", end);
    document.querySelectorAll("[data-lead]").forEach((b) => b.addEventListener("click", () => {
      const on = b.getAttribute("aria-pressed") === "true";
      document.querySelectorAll("[data-lead]").forEach((x) => x.setAttribute("aria-pressed", "false"));
      heart3d.sel = on ? null : b.dataset.lead; heart3d.lv = false; heart3d.fromRecording = false; $("#b-lv").setAttribute("aria-pressed", "false");
      if (!on) b.setAttribute("aria-pressed", "true");
      aimHeart(); setHeartNote(); drawHeart(performance.now());
    }));
    $("#b-lv").addEventListener("click", (e) => {
      heart3d.lv = e.currentTarget.getAttribute("aria-pressed") !== "true"; heart3d.fromRecording = false;
      e.currentTarget.setAttribute("aria-pressed", String(heart3d.lv));
      if (heart3d.lv) { heart3d.sel = null; document.querySelectorAll("[data-lead]").forEach((x) => x.setAttribute("aria-pressed", "false")); }
      aimHeart(); setHeartNote(); drawHeart(performance.now());
    });
    setHeartNote(); drawHeart(0);
    if (heart3d.still) return;
    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      if (heart3d.drag) heart3d.target = null;
      else if (heart3d.target) {                                     // ease toward the lead's view, shortest way round
        const dy = ((heart3d.target.yaw - heart3d.yaw + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
        heart3d.yaw += dy * Math.min(1, dt * 4); heart3d.pitch += (heart3d.target.pitch - heart3d.pitch) * Math.min(1, dt * 4);
      } else if (!heart3d.sel) heart3d.yaw += dt * 0.18;
      last = now; drawHeart(now);
      if (visible && !document.hidden) raf = requestAnimationFrame(tick);
    };
    const start = () => { cancelAnimationFrame(raf); last = performance.now(); raf = requestAnimationFrame(tick); };
    if ("IntersectionObserver" in window)
      new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) start(); }).observe(stage);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && visible) start(); });
    start();
  }

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
  function redrawAll() { readColors(); if (lab.sig) drawLab(); drawHeart(performance.now()); }
  // Printing uses the light colour set: switch, redraw the charts, and switch back afterwards.
  window.addEventListener("beforeprint", () => { document.documentElement.classList.add("print-light"); redrawAll(); });
  window.addEventListener("afterprint", () => { document.documentElement.classList.remove("print-light"); redrawAll(); });
  let rz = null;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(redrawAll, 120); });
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // installable and usable offline after the first visit (sw.js); http on localhost is allowed for testing
  function registerOffline() {
    if (!("serviceWorker" in navigator)) return;
    if (location.protocol !== "https:" && location.hostname !== "localhost") return;
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
  function boot() {
    registerOffline();
    initLab();
    coverDots();
    initHeart();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
