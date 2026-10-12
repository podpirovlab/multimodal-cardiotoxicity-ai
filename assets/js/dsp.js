/*  CardioOncoPredict — browser DSP library (port of cardioonco/*.py)
 *
 *  Everything the web page reports is computed here, live, from the signal:
 *    synth()          parametric PQRST generator with microvolt T-wave alternans
 *    bandpass()       zero-phase Butterworth (biquad, forward + backward)
 *    detectRPeaks()   Elgendi (2013): 8–20 Hz band-pass → square → QRS-length vs beat-length averages
 *    resamplePoly()   interpolation of recordings sampled below 400 Hz, as scipy.signal.resample_poly
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

  // ---------- R-peak detection ----------
  // Elgendi (2013) two-moving-average QRS detection, as in cardioonco/preprocess.py: 8-20 Hz
  // band-pass, squaring, and a block of interest wherever the QRS-length (97 ms) average of the
  // squared signal exceeds the beat-length (611 ms) average plus 0.08 x its mean.  Each beat is
  // judged against the energy around it, so small beats next to large ectopic ones are found.
  function movingAverage(y, n) {                     // centred, like numpy.convolve(..., "same")
    const out = new Float64Array(y.length), off = Math.floor((n - 1) / 2);
    let run = 0;
    for (let i = 0; i < y.length + n; i++) {
      if (i < y.length) run += y[i];
      if (i - n >= 0) run -= y[i - n];
      const c = i - off;                            // the window y[i-n+1..i] belongs to c
      if (c >= 0 && c < y.length) out[c] = run / n;
    }
    return out;
  }
  function qrsBlocks(x, fs) {
    const f = bandpass(x, fs, 8, 20), n = f.length, y = new Float64Array(n);
    let m = 0;
    for (let i = 0; i < n; i++) { y[i] = f[i] * f[i]; m += y[i]; }
    m /= n;
    const n1 = Math.max(1, Math.round(0.097 * fs)), n2 = Math.max(1, Math.round(0.611 * fs));
    const a = movingAverage(y, n1), b = movingAverage(y, n2), peaks = [];
    let start = -1;
    for (let i = 0; i <= n; i++) {
      const on = i < n && a[i] > b[i] + 0.08 * m;
      if (on && start < 0) start = i;
      if (!on && start >= 0) {
        if (i - start >= n1) {
          let best = start;
          for (let j = start; j < i; j++) if (Math.abs(f[j]) > Math.abs(f[best])) best = j;
          peaks.push(best);
        }
        start = -1;
      }
    }
    return peaks;
  }

  function detectRPeaks(x, fs) {
    const blocks = qrsBlocks(x, fs);
    const xf = bandpass(x, fs, 0.5, 40), n = xf.length, half = Math.trunc(0.06 * fs);
    const gap = Math.trunc(0.25 * fs), r = [];
    for (const p of blocks) {                       // refine to the largest |ECG| near the block peak
      const a = Math.max(0, p - 2 * half), b = Math.min(n, p + half);
      let q = a;
      for (let i = a; i < b; i++) if (Math.abs(xf[i]) > Math.abs(xf[q])) q = i;
      if (r.length && q - r[r.length - 1] < gap) {  // one beat cannot follow another within 250 ms
        if (Math.abs(xf[q]) > Math.abs(xf[r[r.length - 1]])) r[r.length - 1] = q;
      } else if (!r.length || q !== r[r.length - 1]) r.push(q);
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
  // ---------- interpolation for coarse sampling (as scipy.signal.resample_poly(x, up, 1)) ----------
  // Below MIN_ANALYSIS_FS one sample is too coarse to superimpose beats to a microvolt, so the
  // recording is interpolated by an integer factor first, exactly as cardioonco/twa.py does.
  const MIN_ANALYSIS_FS = 400;
  const analysisRate = (fs) => (fs >= MIN_ANALYSIS_FS ? 1 : Math.ceil(500 / fs));
  function besselI0(v) {
    let sum = 1, term = 1;
    for (let k = 1; k < 60; k++) { term *= (v / (2 * k)) ** 2; sum += term; if (term < 1e-17 * sum) break; }
    return sum;
  }
  function resamplePoly(x, up) {
    // scipy's default filter: firwin(2*10*up + 1, 1/up, window=("kaiser", 5.0)) * up, zero padding
    const half = 10 * up, N = 2 * half + 1, fc = 1 / up, h = new Float64Array(N);
    let sum = 0;
    for (let n = 0; n < N; n++) {
      const m = n - half, arg = fc * m;
      const sinc = arg === 0 ? 1 : Math.sin(Math.PI * arg) / (Math.PI * arg);
      const r = 2 * n / (N - 1) - 1;
      h[n] = fc * sinc * besselI0(5 * Math.sqrt(Math.max(0, 1 - r * r))) / besselI0(5);
      sum += h[n];
    }
    for (let n = 0; n < N; n++) h[n] *= up / sum;
    const nOut = x.length * up, y = new Float64Array(nOut);
    for (let m = 0; m < nOut; m++) {
      // y[m] = sum_n h[n] * xu[m + half - n], where xu holds x at multiples of `up` and zeros elsewhere
      const j0 = m + half;
      let acc = 0;
      for (let n = ((j0 % up) + up) % up; n < N; n += up) {
        const i = (j0 - n) / up;
        if (i >= 0 && i < x.length) acc += h[n] * x[i];
      }
      y[m] = acc;
    }
    return y;
  }

  function analyzeTWA(x, fs, opts) {
    const o = Object.assign({ nBeats: STANDARD_BEATS, hop: WINDOW_HOP, rPeaks: null }, opts || {});
    const up = analysisRate(fs);
    if (up > 1) {
      // analyse at fs*up, then hand back R peaks, filtered signal and beats at the input rate
      const res = analyzeTWA(resamplePoly(x, up), fs * up,
        Object.assign({}, o, { rPeaks: o.rPeaks ? Array.from(o.rPeaks, (v) => v * up) : null }));
      const every = (a) => Float64Array.from({ length: Math.ceil(a.length / up) }, (_, k) => a[k * up]);
      return Object.assign(res, { r: res.r.map((v) => Math.round(v / up)), xf: every(res.xf),
                                  fullBeats: res.fullBeats.map(every), upsampled: up });
    }
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
      const rates = diffs(r.slice(s0, s0 + size)).map((d) => 60 * fs / d);
      return { start: s0, hr: 60 / (median(diffs(r.slice(s0, s0 + size))) / fs), hrP90: percentile(rates, 90), vAlt: s.vAlt, k: s.k,
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

    // alternans against heart rate (Holter, daily life): onset rate and how sustained it is above it
    const sigClean = significant.filter((w) => w.noise <= NOISE_MAX);
    const onsetHr = sigClean.length ? Math.min(...sigClean.map((w) => w.hrP90)) : NaN;   // the rate a window reached
    const above = valid.filter((w) => w.noise <= NOISE_MAX && w.hrP90 >= onsetHr);
    const sustainedShare = above.length ? above.filter((w) => significant.includes(w)).length / above.length : NaN;
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
             window: [aSt / fs, bSt / fs], windows, onsetHr, sustainedShare };
  }

  // ---------- several leads: the two-adjacent-precordial-leads rule (port of cardioonco.twa.analyze_leads) ----------
  // Microvolt TWA counts when it "is present in leads X, Y, Z, VM or two adjacent precordial leads"
  // (FDA 510(k) K012206, 2001); a standard 12-lead ECG has no Frank leads, so the precordial rule is used.
  const PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"];
  const REASON_ORDER = ["noise", "ectopy", "hr_too_high", "hr_too_low"];
  function analyzeLeads(leads, names, fs, rPeaks) {
    const r = rPeaks || consensusRPeaks(leads, fs);
    const per = {}, done = {};
    leads.forEach((x, i) => {
      try {
        const a = analyzeTWA(x, fs, { rPeaks: r });
        done[names[i]] = a;
        per[names[i]] = { outcome: a.outcome, reason: a.reason, vAlt: a.vAlt, k: a.k, noise: a.noise,
                          windowHr: a.windowHr, estimate: a.estimate };
      } catch (e) { per[names[i]] = { error: e.message, code: e.code }; }
    });
    const adj = PRECORDIAL.slice(0, -1).map((a, i) => [a, PRECORDIAL[i + 1]]).filter(([a, b]) => done[a] && done[b]);
    const pairs = adj.filter(([a, b]) => done[a].positive && done[b].positive);
    const pairEstimate = Math.max(0, ...adj.map(([a, b]) => Math.min(done[a].estimate, done[b].estimate)));
    const prec = PRECORDIAL.filter((n) => done[n]).map((n) => done[n]);
    let outcome, reason;
    if (pairs.length) [outcome, reason] = ["positive", "adjacent_precordial"];
    else if (names.filter((n) => PRECORDIAL.includes(n)).length < 2) [outcome, reason] = ["indeterminate", "no_precordial"];
    else if (prec.length < 2) [outcome, reason] = ["indeterminate", "not_analysable"];
    else if (Object.values(done).some((a) => a.positive)) [outcome, reason] = ["indeterminate", "isolated"];
    else if (prec.every((a) => a.outcome === "negative")) [outcome, reason] = ["negative", "criterion_not_met"];
    else {
      const rs = prec.filter((a) => a.outcome !== "negative").map((a) => a.reason);
      const count = (q) => rs.filter((v) => v === q).length;
      outcome = "indeterminate";
      reason = REASON_ORDER.reduce((best, q) => (count(q) > count(best) ? q : best), REASON_ORDER[0]);
    }
    const rr = median(diffs(r));
    return { outcome, reason, positivePairs: pairs, pairEstimate, hr: 60 * fs / rr, leads: per };
  }

  // ---------- QT and QTc (port of cardioonco/qt.py; constants chosen on PTB-XL fold 9) ----------
  const QT = { offsetMs: 35.2, maxIqrMs: 40, minTAmp: 0.05, minLeads: 6, segmentS: 10 };
  function qtMedianBeats(leads, fs, r) {
    const rr = diffs(r), med = median(rr);
    const a = Math.round(0.30 * fs), b = Math.round(Math.min(0.75, 0.95 * med / fs) * fs);
    const use = [];
    for (let i = 1; i < r.length - 1; i++)
      if (Math.abs(rr[i - 1] - med) < 0.2 * med && Math.abs(rr[i] - med) < 0.2 * med && r[i] - a >= 0 && r[i] + b <= leads[0].length) use.push(r[i]);
    if (use.length < 3) throw analysisError("qt_beats", `too few regular beats for QT: ${use.length}`);
    const L = a + b, t = Float64Array.from({ length: L }, (_, k) => (k - a) / fs);
    const pr = [];
    for (let k = 0; k < L; k++) if (t[k] >= -0.10 && t[k] <= -0.04) pr.push(k);
    const beats = leads.map((x) => {
      const m = new Float64Array(L), col = new Float64Array(use.length);
      for (let k = 0; k < L; k++) { for (let j = 0; j < use.length; j++) col[j] = x[use[j] - a + k]; m[k] = median(col); }
      const base = median(pr.map((k) => m[k]));
      return m.map((v) => v - base);
    });
    return { beats, t, n: use.length, rr: med / fs };
  }
  const grad = (y) => Float64Array.from(y, (_, i) => i === 0 ? y[1] - y[0] : i === y.length - 1 ? y[i] - y[i - 1] : (y[i + 1] - y[i - 1]) / 2);
  const firstAtLeast = (t, v) => { let i = 0; while (i < t.length && t[i] < v) i++; return i; };
  function qtQrsOnset(beats, t, fs) {
    const v = new Float64Array(t.length);
    for (const b of beats) { const d = grad(b); for (let k = 0; k < t.length; k++) v[k] += Math.abs(d[k]) * fs; }
    let mx = 0;
    for (let k = 0; k < t.length; k++) if (t[k] >= -0.06 && t[k] <= 0.06) mx = Math.max(mx, v[k]);
    let i = 0, best = Infinity;
    for (let k = 0; k < t.length; k++) if (Math.abs(t[k]) < best) { best = Math.abs(t[k]); i = k; }
    const lo = firstAtLeast(t, -0.15);
    while (i > lo && v[i] >= 0.10 * mx) i--;
    return t[i];
  }
  function qtTEnd(beat, t, fs, rr) {
    const tMax = Math.min(rr - 0.12, t[t.length - 1]);
    const lo = firstAtLeast(t, 0.10 * Math.sqrt(rr / 0.8) + 0.05), hi = firstAtLeast(t, Math.min(0.65 * rr + 0.05, tMax));
    if (hi - lo < Math.trunc(0.08 * fs)) return NaN;
    let peak = lo;
    for (let k = lo; k < hi; k++) if (Math.abs(beat[k]) > Math.abs(beat[peak])) peak = k;
    const amp = beat[peak];
    if (Math.abs(amp) < QT.minTAmp) return NaN;
    const s = beat.map((v) => Math.sign(amp) * v), d = grad(s).map((v) => v * fs);
    const stopLim = firstAtLeast(t, tMax);
    let stop = -1;
    for (let k = peak; k < stopLim; k++) if (s[k] < 0.15 * Math.abs(amp)) { stop = k; break; }
    if (stop < 0 || stop <= peak + 2) return NaN;
    let j = peak;
    for (let k = peak; k <= stop; k++) if (d[k] < d[j]) j = k;
    if (d[j] >= 0) return NaN;
    const te = t[j] - s[j] / d[j];
    return te > t[peak] && te <= tMax ? te : NaN;
  }
  // R peaks shared by all leads: the detections of the lead the others agree with most
  // (port of cardioonco.preprocess.consensus_r_peaks)
  function consensusRPeaks(leads, fs, tolS = 0.05) {
    const cands = leads.map((x) => detectRPeaks(x, fs).r);
    if (cands.length === 1) return cands[0];
    const tol = tolS * fs;
    const share = (a, b) => {
      if (!a.length || b.length < 2) return 0;
      let hit = 0;
      for (const v of a) {
        let lo = 0, hi = b.length;                  // first index with b[i] >= v
        while (lo < hi) { const m = (lo + hi) >> 1; if (b[m] < v) lo = m + 1; else hi = m; }
        const i = Math.min(b.length - 1, Math.max(1, lo));
        if (Math.min(Math.abs(b[i] - v), Math.abs(b[i - 1] - v)) <= tol) hit++;
      }
      return hit / a.length;
    };
    let best = 0, bestScore = -Infinity;
    cands.forEach((a, i) => {
      const s = median(cands.filter((_, j) => j !== i).map((b) => Math.min(share(a, b), share(b, a))));
      if (s > bestScore) { bestScore = s; best = i; }
    });
    return cands[best];
  }
  // leads: array of Float64Array in mV (one or more); rPeaks: sample indices of R (found if omitted)
  function measureQT(leads, fs, rPeaks) {
    const up = analysisRate(fs);          // below 400 Hz: interpolate first, as for TWA
    if (up > 1) {
      leads = leads.map((x) => resamplePoly(x, up)); fs *= up;
      if (rPeaks) rPeaks = Array.from(rPeaks, (v) => v * up);
    }
    const r = rPeaks || consensusRPeaks(leads, fs);
    if (r.length < 5) throw analysisError("qt_beats", "too few beats for QT");
    const { beats, t, n, rr } = qtMedianBeats(leads.map((x) => bandpass(x, fs, 0.5, 40)), fs, r);
    const on = qtQrsOnset(beats, t, fs);
    const ends = beats.map((b) => qtTEnd(b, t, fs, rr));
    const good = ends.filter(Number.isFinite).map((e) => e * 1000);
    if (!good.length) throw analysisError("qt_no_t", "no lead with a measurable T wave");
    const raw = median(good) - on * 1000, qt = raw + QT.offsetMs;
    const iqr = good.length >= 2 ? percentile(good, 75) - percentile(good, 25) : NaN;
    const reliable = good.length >= QT.minLeads && iqr <= QT.maxIqrMs;
    return { qtMs: qt, qtcfMs: qt / Math.cbrt(rr), qtcbMs: qt / Math.sqrt(rr), qtTangentMs: raw, rrMs: rr * 1000,
             hr: 60 / rr, reliable, nLeadsUsed: good.length, iqrMs: iqr, nBeats: n,
             segmentStart: 0, nSegments: 1, nSegmentsReliable: +reliable };
  }
  // consecutive 10-s segments (at most maxSegments, evenly spread); the whole recording if shorter than 20 s
  function qtSegmentStarts(nSamples, fs, segS = QT.segmentS, maxSegments = 60) {
    const seg = Math.round(segS * fs), k = Math.floor(nSamples / seg);
    if (k < 2) return [0];
    const m = Math.min(k, maxSegments), idx = new Set();
    for (let i = 0; i < m; i++) idx.add(Math.round(m === 1 ? 0 : i * (k - 1) / (m - 1)));
    return [...idx].sort((a, b) => a - b).map((i) => i * seg);
  }
  // QT of a recording of any length: the segment with the median QTcF among the reliable ones
  // (port of cardioonco.qt.measure_qt_segments). segments[k] holds the leads of the segment that
  // starts at sample starts[k], so a caller can read just those stretches of a long file.
  function measureQTOnSegments(segments, starts, fs) {
    const found = [];
    segments.forEach((leads, k) => {
      try { found.push([starts[k], measureQT(leads, fs)]); } catch (e) { /* next segment */ }
    });
    if (!found.length) throw analysisError("qt_none", "QT could not be measured in any segment");
    const good = found.filter((f) => f[1].reliable);
    const pool = (good.length ? good : found).slice().sort((a, b) => a[1].qtcfMs - b[1].qtcfMs);
    const [s, res] = pool[Math.floor((pool.length - 1) / 2)];
    return Object.assign(res, { segmentStart: s / fs, nSegments: found.length, nSegmentsReliable: good.length });
  }
  function measureQTSegments(leads, fs, segS = QT.segmentS, maxSegments = 60) {
    const starts = qtSegmentStarts(leads[0].length, fs, segS, maxSegments);
    const seg = starts.length === 1 ? leads[0].length : Math.round(segS * fs);
    return measureQTOnSegments(starts.map((s) => leads.map((x) => x.subarray(s, s + seg))), starts, fs);
  }

  root.CardioDSP = { version: "1.9.0", MIN_BEATS, STANDARD_BEATS, V_ALT_MIN, K_MIN, NOISE_MAX,
    HR_ONSET_MAX, HR_NEGATIVE_MIN, MAX_ECTOPIC, rng, synth, bandpass, detectRPeaks, removeExtraBeats,
    fillMissedBeats, alignBeats, flagEctopic, spectral, mma, analyzeTWA, percentile, resamplePoly, MIN_ANALYSIS_FS, PRECORDIAL, analyzeLeads, QT, consensusRPeaks, measureQT, qtSegmentStarts, measureQTOnSegments, measureQTSegments };
  if (typeof module !== "undefined") module.exports = root.CardioDSP;
})(typeof window !== "undefined" ? window : globalThis);
