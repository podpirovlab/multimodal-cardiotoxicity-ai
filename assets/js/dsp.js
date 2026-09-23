/*  CardioOncoPredict — browser DSP library (port of cardioonco/*.py)
 *
 *  Everything the web page reports is computed here, live, from the signal:
 *    synth()          parametric PQRST generator with microvolt T-wave alternans
 *    bandpass()       zero-phase Butterworth (biquad, forward + backward)
 *    detectRPeaks()   Pan–Tompkins: 5–15 Hz band-pass → derivative → square → 150 ms integration
 *    analyzeTWA()     beat alignment → Spectral Method (V_alt, K-score) + Modified Moving Average
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
    return { r, xf };
  }

  // ---------- TWA ----------
  function beatMatrix(xf, r, fs, startS, endS) {
    const a = Math.round(startS * fs), b = Math.round(endS * fs), rows = [];
    for (const ri of r) {
      if (ri + a < 0 || ri + b > xf.length) continue;
      const row = Array.from(xf.subarray(ri + a, ri + b));
      const m = median(row);
      rows.push(row.map((v) => v - m));
    }
    return rows;
  }

  function spectral(beats, band = [0.44, 0.49]) {
    const N = beats.length, L = beats[0].length;
    const nF = Math.floor(N / 2) + 1, P = new Float64Array(nF), Pk = [];
    const mean = new Float64Array(L);
    for (const row of beats) for (let k = 0; k < L; k++) mean[k] += row[k] / N;
    for (let k = 0; k < L; k++) Pk.push(new Float64Array(nF));
    for (let f = 0; f < nF; f++) {
      const w = 2 * Math.PI * f / N;
      for (let k = 0; k < L; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < N; n++) {
          const s = (beats[n][k] - mean[k]) * 1000;           // µV
          re += s * Math.cos(w * n); im -= s * Math.sin(w * n);
        }
        const p = (re * re + im * im) / (N * N);
        Pk[k][f] = p; P[f] += p / L;
      }
    }
    const freqs = Array.from({ length: nF }, (_, f) => f / N);
    const iAlt = nF - 1, inBand = freqs.map((f) => f >= band[0] && f <= band[1]);
    const bandVals = Array.from(P).filter((_, i) => inBand[i]);
    const mu = bandVals.reduce((s, v) => s + v, 0) / bandVals.length;
    const sd = Math.sqrt(bandVals.reduce((s, v) => s + (v - mu) ** 2, 0) / (bandVals.length - 1)) + 1e-12;
    let peak = 0;
    for (let k = 0; k < L; k++) {
      let nb = 0, cnt = 0;
      for (let f = 0; f < nF; f++) if (inBand[f]) { nb += Pk[k][f]; cnt++; }
      peak = Math.max(peak, Pk[k][iAlt] - nb / cnt);
    }
    return { freqs, P: Array.from(P), vAlt: Math.sqrt(Math.max(P[iAlt] - mu, 0)),
             k: (P[iAlt] - mu) / sd, noise: Math.sqrt(mu), vPeak: Math.sqrt(Math.max(peak, 0)) };
  }

  function mma(beats, step = 1 / 8, limit = 32) {
    const L = beats[0].length;
    const A = beats[0].map((v) => v * 1000), B = beats[1].map((v) => v * 1000);
    for (let i = 2; i < beats.length; i++) {
      const T = i % 2 === 0 ? A : B;
      for (let k = 0; k < L; k++) {
        const d = Math.max(-limit, Math.min(limit, (beats[i][k] * 1000 - T[k]) * step));
        T[k] += d;
      }
    }
    let m = 0;
    for (let k = 0; k < L; k++) m = Math.max(m, Math.abs(A[k] - B[k]));
    return m / 2;
  }

  function analyzeTWA(x, fs, nBeats = 128) {
    const { r, xf } = detectRPeaks(x, fs);
    if (r.length < 3) throw new Error("too few R peaks");
    const rrMed = median(r.slice(1).map((v, i) => v - r[i])) / fs;
    const hr = 60 / rrMed, scale = Math.sqrt(rrMed / 0.8);
    let beats = beatMatrix(xf, r, fs, 0.10 * scale, 0.42 * scale).slice(0, nBeats);
    if (beats.length % 2) beats = beats.slice(0, -1);
    if (beats.length < 16) throw new Error(`need at least 16 beats, got ${beats.length}`);
    const s = spectral(beats);
    const full = beatMatrix(xf, r, fs, -0.25, 0.55).slice(0, beats.length);
    return { r, xf, hr, nBeats: beats.length, vAlt: s.vAlt, vPeak: s.vPeak, k: s.k, noise: s.noise,
             mma: mma(beats), freqs: s.freqs, P: s.P, positive: s.vAlt >= 1.9 && s.k >= 3,
             fullBeats: full, window: [0.10 * scale, 0.42 * scale] };
  }

  root.CardioDSP = { rng, synth, bandpass, detectRPeaks, beatMatrix, spectral, mma, analyzeTWA, percentile };
  if (typeof module !== "undefined") module.exports = root.CardioDSP;
})(typeof window !== "undefined" ? window : globalThis);
