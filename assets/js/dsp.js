/*  CardioOncoPredict — browser DSP library (port of cardioonco/*.py)
 *
 *  Everything the web page reports is computed here, live, from the signal:
 *    synth()          parametric PQRST generator with microvolt T-wave alternans
 *    bandpass()       zero-phase Butterworth (biquad, forward + backward)
 *    detectRPeaks()   Pan–Tompkins: 5–15 Hz band-pass → derivative → square → 150 ms integration
 *    analyzeTWA()     whole-recording TWA: beat clean-up and alignment → 128-beat Spectral Method
 *                     windows (V_alt, K-score) + Modified Moving Average → three outcomes
 *
 *  Units: seconds, millivolts (mV); TWA outputs in microvolts (µV).
 *  Research / education prototype — not a medical device.
 */
(function (root) {
  "use strict";

  // ---------- seeded random numbers (mulberry32 + Box–Muller) ----------
  function rng(seed) {
    let a = seed >>> 0;
    const uni = () => {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = uni();
      v = uni();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uni, normal };
  }

  // ---------- synthetic ECG: sum of Gaussians per beat ----------
  const WAVES = [ // amplitude mV, centre s (from beat onset), width s
    [0.15, 0.20, 0.030], [-0.10, 0.30, 0.010], [1.20, 0.32, 0.015], [-0.25, 0.35, 0.015]];

  function synth(o) {
    const c = Object.assign({ fs: 500, nBeats: 128, hr: 75, hrvStd: 0.01, altUv: 0, tAmp: 0.35,
      tWidth: 0.05, noiseUv: 10, wanderMv: 0.10, mainsUv: 0, mainsHz: 50, seed: 7 }, o || {});
    const R = rng(c.seed), rrMean = 60 / c.hr;
    const rr = [], onsets = [];
    let acc = 0;
    for (let k = 0; k < c.nBeats; k++) {
      const v = Math.min(2, Math.max(0.3, rrMean + c.hrvStd * R.normal()));
      rr.push(v); onsets.push(acc); acc += v;
    }
    const n = Math.floor((acc + 0.5) * c.fs), x = new Float64Array(n), rIdx = [];
    const alt = c.altUv / 1000;
    for (let k = 0; k < c.nBeats; k++) {
      const i0 = Math.floor(onsets[k] * c.fs), i1 = Math.min(n, i0 + Math.floor(1.2 * c.fs));
      const tMu = 0.32 + 0.23 * Math.sqrt(rr[k] / 0.8);
      const aT = c.tAmp + (k % 2 === 0 ? alt : -alt);
      for (let i = i0; i < i1; i++) {
        const t = i / c.fs - onsets[k];
        let s = 0;
        for (const [a, mu, w] of WAVES) s += a * Math.exp(-(((t - mu) / w) ** 2));
        s += aT * Math.exp(-(((t - tMu) / c.tWidth) ** 2));
        x[i] += s;
      }
      rIdx.push(i0 + Math.round(0.32 * c.fs));
    }
    for (let i = 0; i < n; i++) {
      const t = i / c.fs;
      x[i] += c.wanderMv * Math.sin(2 * Math.PI * 0.25 * t)
            + (c.mainsUv / 1000) * Math.sin(2 * Math.PI * c.mainsHz * t)
            + (c.noiseUv / 1000) * R.normal();
    }
    return { x, fs: c.fs, rTrue: rIdx };
  }

  // ---------- filters (RBJ biquads, Butterworth Q = 1/√2), zero-phase ----------
  function biquad(type, f0, fs) {
    const w0 = 2 * Math.PI * f0 / fs, cs = Math.cos(w0), al = Math.sin(w0) / (2 * Math.SQRT1_2);
    let b;
    if (type === "lp") b = [(1 - cs) / 2, 1 - cs, (1 - cs) / 2];
    else b = [(1 + cs) / 2, -(1 + cs), (1 + cs) / 2];
    const a0 = 1 + al;
    return { b0: b[0] / a0, b1: b[1] / a0, b2: b[2] / a0, a1: -2 * cs / a0, a2: (1 - al) / a0 };
  }
  function runBiquad(q, x) {
    const y = new Float64Array(x.length);
    // start in steady state for a constant input equal to x[0]
    let x1 = x[0], x2 = x[0], y1 = x[0] * (q.b0 + q.b1 + q.b2) / (1 + q.a1 + q.a2), y2 = y1;
    for (let i = 0; i < x.length; i++) {
      const v = q.b0 * x[i] + q.b1 * x1 + q.b2 * x2 - q.a1 * y1 - q.a2 * y2;
      x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
    }
    return y;
  }
  function filtfilt(sections, x) {
    const pad = Math.min(x.length - 1, 3 * 200);
    const n = x.length, ext = new Float64Array(n + 2 * pad);
    for (let i = 0; i < pad; i++) ext[i] = 2 * x[0] - x[pad - i];            // odd reflection
    ext.set(x, pad);
    for (let i = 0; i < pad; i++) ext[pad + n + i] = 2 * x[n - 1] - x[n - 2 - i];
    let y = ext;
    for (const s of sections) y = runBiquad(s, y);
    y.reverse();
    for (const s of sections) y = runBiquad(s, y);
    y.reverse();
    return y.slice(pad, pad + n);
  }
  function bandpass(x, fs, lo = 0.5, hi = 40) {
    hi = Math.min(hi, 0.45 * fs);
    return filtfilt([biquad("hp", lo, fs), biquad("lp", hi, fs)], x);
  }

  // ---------- helpers ----------
  function percentile(arr, p) {
    const s = Float64Array.from(arr).sort();
    const idx = (p / 100) * (s.length - 1), lo = Math.floor(idx), hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }
  function findPeaks(y, height, distance) {
    const cand = [];
    for (let i = 1; i < y.length - 1; i++)
      if (y[i] >= height && y[i] > y[i - 1] && y[i] >= y[i + 1]) cand.push(i);
    cand.sort((a, b) => y[b] - y[a]);
    const taken = new Uint8Array(y.length), out = [];
    for (const i of cand) {
      if (taken[i]) continue;
      out.push(i);
      for (let j = Math.max(0, i - distance + 1); j < Math.min(y.length, i + distance); j++) taken[j] = 1;
    }
    return out.sort((a, b) => a - b);
  }
  const median = (a) => percentile(a, 50);

  // ---------- small vector helpers ----------
  const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a));
  function colMedian(rows) {                        // per-sample median over beats
    const L = rows[0].length, out = new Float64Array(L), col = new Float64Array(rows.length);
    for (let k = 0; k < L; k++) { for (let i = 0; i < rows.length; i++) col[i] = rows[i][k]; out[k] = median(col); }
    return out;
  }
  const centred = (a) => { const m = mean(a); return Float64Array.from(a, (v) => v - m); };

  // ---------- Pan–Tompkins R-peak detection ----------
  function detectRPeaks(x, fs) {
    const qrs = bandpass(x, fs, 5, 15);
    const n = qrs.length, e = new Float64Array(n);
    for (let i = 1; i < n - 1; i++) { const d = qrs[i + 1] - qrs[i - 1]; e[i] = d * d; }
    const w = Math.max(1, Math.round(0.15 * fs)), mwi = new Float64Array(n);
    let run = 0;
    for (let i = 0; i < n + (w >> 1); i++) {        // centred moving average
      if (i < n) run += e[i];
      if (i - w >= 0) run -= e[i - w];
      const c = i - (w >> 1);
      if (c >= 0 && c < n) mwi[c] = run / w;
    }
    const peaks = findPeaks(mwi, 0.3 * percentile(mwi, 99), Math.round(0.25 * fs));
    const xf = bandpass(x, fs, 0.5, 40), half = Math.round(0.06 * fs), r = [];
    for (const p of peaks) {
      const a = Math.max(0, p - 2 * half), b = Math.min(n, p + half);
      let best = a;
      for (let i = a; i < b; i++) if (Math.abs(xf[i]) > Math.abs(xf[best])) best = i;
      if (!r.length || best !== r[r.length - 1]) r.push(best);
    }
    return { r: removeExtraBeats(r, xf, fs), xf };
  }

  // median of the surrounding ±context R-R intervals, for each interval
  function localRR(rr, context = 4) {
    return rr.map((_, i) => median(rr.slice(Math.max(0, i - context), i + context + 1)));
  }
  const diffs = (r) => r.slice(1).map((v, i) => v - r[i]);

  // correlation of each detection's ±50 ms neighbourhood with the median QRS
  function qrsSimilarity(xf, r, fs, halfS = 0.05) {
    const h = Math.trunc(halfS * fs), sim = new Float64Array(r.length).fill(1);
    const ok = r.map((_, i) => i).filter((i) => r[i] - h >= 0 && r[i] + h <= xf.length);
    if (ok.length < 3) return sim;
    const seg = ok.map((i) => centred(xf.subarray(r[i] - h, r[i] + h)));
    const t = centred(colMedian(seg)), tn = norm(t);
    ok.forEach((i, j) => { sim[i] = dot(seg[j], t) / (norm(seg[j]) * tn + 1e-12); });
    return sim;
  }

  // A tall, sharp T wave can pass the QRS threshold: a short interval whose sum with the
  // next is one normal R-R (Lipponen & Tarvainen 2019). Every such extra "beat" would flip
  // the ABAB parity of all later beats. A candidate that still looks like a QRS is kept.
  function removeExtraBeats(r, xf, fs, tol = 0.20, context = 15, minSim = 0.90) {
    for (let pass = 0; pass < 2; pass++) {
      if (r.length < 4) return r;
      const ref = localRR(diffs(r), context);
      const sim = xf ? qrsSimilarity(xf, r, fs) : new Float64Array(r.length);
      const kept = [r[0]];
      for (let i = 1; i < r.length - 1; i++) {
        const cycle = ref[i - 1], prev = kept[kept.length - 1];
        if (r[i] - prev < (1 - tol) * cycle && Math.abs(r[i + 1] - prev - cycle) < tol * cycle && sim[i] < minSim) continue;
        kept.push(r[i]);
      }
      kept.push(r[r.length - 1]);
      r = kept;
    }
    return r;
  }

  // A gap of about two R-R intervals is a missed beat; a flagged placeholder keeps the parity.
  function fillMissedBeats(r, tol = 0.20) {
    if (r.length < 4) return { r: r.slice(), filled: r.map(() => false) };
    const rr = diffs(r), ref = localRR(rr), out = [r[0]], filled = [false];
    rr.forEach((gap, i) => {
      if (Math.abs(gap - 2 * ref[i]) < 2 * tol * ref[i]) { out.push(r[i] + Math.round(gap / 2)); filled.push(true); }
      out.push(r[i + 1]); filled.push(false);
    });
    return { r: out, filled };
  }

  // Superimpose the beats to a sample: cross-correlate each QRS with the median QRS (±20 ms).
  function alignBeats(xf, r, fs, halfS = 0.05, maxShiftS = 0.02, passes = 2) {
    const h = Math.trunc(halfS * fs), m = Math.trunc(maxShiftS * fs);
    r = r.slice();
    const inRange = (i) => r[i] - h - m >= 0 && r[i] + h + m < xf.length;
    let ok = r.map((_, i) => i).filter(inRange);
    if (ok.length < 3 || m === 0) return r;
    for (let pass = 0; pass < passes; pass++) {
      const t = centred(colMedian(ok.map((i) => xf.subarray(r[i] - h, r[i] + h))));
      for (const i of ok) {
        let best = -Infinity, shift = 0;
        for (let s = -m; s <= m; s++) {
          let c = 0;
          for (let k = 0; k < t.length; k++) c += xf[r[i] + s - h + k] * t[k];
          if (c > best) { best = c; shift = s; }
        }
        r[i] += shift;
      }
      ok = ok.filter(inRange);
    }
    return r;
  }

  // premature / post-ectopic / misshapen beats: R-R >20% off the local median, or shape r < 0.9
  function flagEctopic(r, full, rrTol = 0.20, minCorr = 0.90, context = 4) {
    const flags = r.map(() => false), rr = diffs(r);
    if (rr.length) localRR(rr, context).forEach((ref, i) => { if (Math.abs(rr[i] - ref) > rrTol * ref) flags[i + 1] = true; });
    const t = centred(colMedian(full)), tn = norm(t);
    full.forEach((row, i) => {
      const v = centred(row), den = norm(v) * tn;
      if (den === 0 || dot(v, t) / den < minCorr) flags[i] = true;
    });
    return flags;
  }

  // replace flagged beats by the median of same-parity beats, keeping the ABAB phase
  function replaceFlagged(beats, flags) {
    const out = beats.slice();
    for (const parity of [0, 1]) {
      const idx = []; for (let i = parity; i < out.length; i += 2) idx.push(i);
      const good = idx.filter((i) => !flags[i]), bad = idx.filter((i) => flags[i]);
      if (bad.length && good.length) { const m = colMedian(good.map((i) => beats[i])); for (const i of bad) out[i] = m; }
    }
    return out;
  }

  function windowStarts(n, size, hop) {
    if (n <= size) return [0];
    const s = []; for (let i = 0; i <= n - size; i += hop) s.push(i);
    if (s[s.length - 1] !== n - size) s.push(n - size);
    return s;
  }

  // ---------- Spectral Method ----------
  // allBins = false computes only the noise band and 0.5 cycles/beat (the per-window scan)
  function spectral(beats, band = [0.44, 0.49], allBins = true) {
    const N = beats.length, L = beats[0].length, nF = Math.floor(N / 2) + 1;
    const freqs = Array.from({ length: nF }, (_, f) => f / N);
    const iAlt = nF - 1, inBand = freqs.map((f) => f >= band[0] && f <= band[1]);
    const P = new Float64Array(nF), Palt = new Float64Array(L), Pband = new Float64Array(L);
    const mu0 = new Float64Array(L);
    for (const row of beats) for (let k = 0; k < L; k++) mu0[k] += row[k] / N;
    for (let f = 0; f < nF; f++) {
      if (!allBins && !inBand[f] && f !== iAlt) continue;
      const w = 2 * Math.PI * f / N, c = new Float64Array(N), s = new Float64Array(N);
      for (let n = 0; n < N; n++) { c[n] = Math.cos(w * n); s[n] = Math.sin(w * n); }
      for (let k = 0; k < L; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < N; n++) { const v = (beats[n][k] - mu0[k]) * 1000; re += v * c[n]; im -= v * s[n]; }  // µV
        const p = (re * re + im * im) / (N * N);
        P[f] += p / L;
        if (f === iAlt) Palt[k] = p; else if (inBand[f]) Pband[k] += p;
      }
    }
    const bandVals = Array.from(P).filter((_, i) => inBand[i]);
    const mu = mean(bandVals);
    const sd = Math.sqrt(bandVals.reduce((s, v) => s + (v - mu) ** 2, 0) / (bandVals.length - 1)) + 1e-12;
    let peak = 0;
    for (let k = 0; k < L; k++) peak = Math.max(peak, Palt[k] - Pband[k] / bandVals.length);
    return { freqs, P: Array.from(P), vAlt: Math.sqrt(Math.max(P[iAlt] - mu, 0)),
             k: (P[iAlt] - mu) / sd, noise: Math.sqrt(mu), vPeak: Math.sqrt(Math.max(peak, 0)) };
  }

  function mma(beats, step = 1 / 8, limit = 32) {
    const L = beats[0].length;
    const A = Array.from(beats[0], (v) => v * 1000), B = Array.from(beats[1], (v) => v * 1000);
    for (let i = 2; i < beats.length; i++) {
      const T = i % 2 === 0 ? A : B;
      for (let k = 0; k < L; k++) {
        const d = Math.max(-limit, Math.min(limit, (beats[i][k] * 1000 - T[k]) * step));
        T[k] += d;
      }
    }
    let m = 0;
    for (let k = 0; k < L; k++) m = Math.max(m, Math.abs(A[k] - B[k]));
    return m;   // full even-minus-odd difference: the scale clinical MMA cutpoints use
  }

  // Below 64 beats the 0.44–0.49 noise band holds too few spectral bins for a usable estimate.
  const MIN_BEATS = 64, STANDARD_BEATS = 128, WINDOW_HOP = 16, MAX_ECTOPIC = 0.10;
  const V_ALT_MIN = 1.9, K_MIN = 3, NOISE_MAX = 1.8, HR_ONSET_MAX = 110, HR_NEGATIVE_MIN = 105;
  function analysisError(code, message, extra) {
    return Object.assign(new Error(message), { code }, extra || {});
  }

  // Whole-recording pipeline, the same as cardioonco/twa.py analyze():
  // R peaks -> missed/extra beats -> alignment -> PR baseline -> ectopy control ->
  // 128-beat windows every 16 beats -> positive / negative / indeterminate.
  function analyzeTWA(x, fs, opts) {
    const o = Object.assign({ nBeats: STANDARD_BEATS, hop: WINDOW_HOP, rPeaks: null }, opts || {});
    let r0, xf;
    if (o.rPeaks) { r0 = Array.from(o.rPeaks); xf = bandpass(x, fs, 0.5, 40); }
    else ({ r: r0, xf } = detectRPeaks(x, fs));
    if (r0.length < 3) throw analysisError("no_beats", "too few R peaks");
    const rrMed = median(diffs(r0)) / fs, hr = 60 / rrMed, scale = Math.sqrt(rrMed / 0.8);
    const fill = fillMissedBeats(r0);
    const rAll = alignBeats(xf, fill.r, fs);
    const aSt = Math.trunc(0.10 * scale * fs), bSt = Math.trunc(0.42 * scale * fs);
    const aFull = Math.trunc(-0.10 * fs), bFull = bSt;                // never into the next beat
    const aPr = Math.trunc(-0.08 * fs), bPr = Math.trunc(-0.04 * fs);
    const lo = Math.min(aSt, aFull), hi = Math.max(bSt, bFull);
    const keepIdx = rAll.map((_, i) => i).filter((i) => rAll[i] + lo >= 0 && rAll[i] + hi <= xf.length);
    const r = keepIdx.map((i) => rAll[i]), filled = keepIdx.map((i) => fill.filled[i]);
    const usable = r.length - (r.length % 2);
    if (usable < MIN_BEATS)
      throw analysisError("too_short", `recording too short: ${usable} usable beats, at least ${MIN_BEATS} needed`,
        { nBeats: usable, minBeats: MIN_BEATS });

    const pr = r.map((ri) => median(xf.subarray(ri + aPr, ri + bPr)));
    const st = r.map((ri, i) => Float64Array.from(xf.subarray(ri + aSt, ri + bSt), (v) => v - pr[i]));
    const full = r.map((ri) => xf.subarray(ri + aFull, ri + bFull));
    const flags = flagEctopic(r, full).map((f, i) => f || filled[i]);

    const size = Math.min(o.nBeats, usable) - (Math.min(o.nBeats, usable) % 2);
    const windows = windowStarts(r.length, size, o.hop).map((s0) => {
      const wf = flags.slice(s0, s0 + size), beats = replaceFlagged(st.slice(s0, s0 + size), wf);
      const s = spectral(beats, undefined, false);
      return { start: s0, hr: 60 / (median(diffs(r.slice(s0, s0 + size))) / fs), vAlt: s.vAlt, k: s.k,
               noise: s.noise, vPeak: s.vPeak, ectopic: wf.filter(Boolean).length / size, mma: mma(beats) };
    });
    if (!windows.every((w) => isFinite(w.k))) throw analysisError("no_noise_estimate", "noise band estimate unavailable");

    const valid = windows.filter((w) => w.ectopic <= MAX_ECTOPIC);
    const significant = valid.filter((w) => w.k >= K_MIN && w.vAlt >= V_ALT_MIN);
    const estimate = Math.max(0, ...valid.filter((w) => w.k >= K_MIN).map((w) => w.vAlt));
    const clean = valid.filter((w) => w.noise <= NOISE_MAX);
    const hrMaxClean = clean.length ? Math.max(...clean.map((w) => w.hr)) : NaN;
    let outcome, reason;
    if (!valid.length) [outcome, reason] = ["indeterminate", "ectopy"];
    else if (significant.some((w) => w.hr <= HR_ONSET_MAX && w.noise <= NOISE_MAX)) [outcome, reason] = ["positive", "criterion_met"];
    else if (significant.some((w) => w.hr > HR_ONSET_MAX)) [outcome, reason] = ["indeterminate", "hr_too_high"];
    else if (significant.length) [outcome, reason] = ["indeterminate", "noise"];
    else if (!clean.length) [outcome, reason] = ["indeterminate", "noise"];
    else if (hrMaxClean >= HR_NEGATIVE_MIN) [outcome, reason] = ["negative", "criterion_not_met"];
    else [outcome, reason] = ["indeterminate", "hr_too_low"];

    const pool = significant.length ? significant : valid.length ? valid : windows;
    const best = pool.reduce((a, b) => (b.vAlt > a.vAlt ? b : a));
    const bi = best.start, wflags = flags.slice(bi, bi + size);
    const spec = spectral(replaceFlagged(st.slice(bi, bi + size), wflags));   // full spectrum for the plot
    // display beats -0.25..+0.55 s of the reported window (edge samples clamped), PR baseline removed
    const d0 = Math.trunc(-0.25 * fs), d1 = Math.trunc(0.55 * fs);
    const shown = r.slice(bi, bi + size).map((ri, j) => Float64Array.from({ length: d1 - d0 },
      (_, k) => xf[Math.min(xf.length - 1, Math.max(0, ri + d0 + k))] - pr[bi + j]));
    return { r, xf, hr, nBeats: r.length, nWindows: windows.length, windowStart: bi, windowHr: best.hr,
             windowBeats: size, ectopic: best.ectopic, vAlt: best.vAlt, vPeak: best.vPeak, k: best.k,
             noise: best.noise, mma: Math.max(...(valid.length ? valid : [best]).map((w) => w.mma)),
             estimate, hrMaxClean, outcome, reason, positive: outcome === "positive",
             freqs: spec.freqs, P: spec.P, fullBeats: replaceFlagged(shown, wflags),
             window: [aSt / fs, bSt / fs], windows };
  }

  root.CardioDSP = { version: "1.1.0", MIN_BEATS, STANDARD_BEATS, V_ALT_MIN, K_MIN, NOISE_MAX,
    HR_ONSET_MAX, HR_NEGATIVE_MIN, MAX_ECTOPIC, rng, synth, bandpass, detectRPeaks, removeExtraBeats,
    fillMissedBeats, alignBeats, flagEctopic, spectral, mma, analyzeTWA, percentile };
  if (typeof module !== "undefined") module.exports = root.CardioDSP;
})(typeof window !== "undefined" ? window : globalThis);
