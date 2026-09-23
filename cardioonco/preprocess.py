"""ECG pre-processing: filtering and R-peak detection.

Filtering
---------
* Band-pass 0.5-40 Hz, 4th-order Butterworth, applied forward and backward
  (``filtfilt``) so the phase response cancels: zero phase distortion, which
  matters because TWA is measured on the *shape* of the ST-T segment.
  The 0.5 Hz high-pass removes respiratory baseline wander (~0.1-0.4 Hz);
  the 40 Hz low-pass removes EMG noise while keeping the QRS energy (5-25 Hz).
* Optional IIR notch at 50/60 Hz for power-line interference.

R-peak detection (Pan & Tompkins, 1985, simplified)
---------------------------------------------------
1. band-pass 5-15 Hz  -> emphasises QRS slopes
2. derivative          -> y[n] = x[n+1] - x[n-1]
3. squaring            -> makes everything positive, amplifies large slopes
4. moving-window integration over 150 ms
5. peak picking with a 250 ms refractory period and an adaptive threshold,
   then refinement to the true maximum of the filtered ECG within +-60 ms.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def notch(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    if f0 >= fs / 2:
        return x
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, x, axis=-1)


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Return sample indices of R peaks in a single-lead ECG (mV)."""
    qrs = bandpass(x, fs, 5.0, 15.0, order=2)
    d = np.zeros_like(qrs)
    d[1:-1] = qrs[2:] - qrs[:-2]
    e = d ** 2
    w = max(1, int(0.150 * fs))
    mwi = np.convolve(e, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(mwi, 99)
    peaks, _ = find_peaks(mwi, height=thr, distance=int(0.25 * fs))
    # refine to the maximum |ECG| near each energy peak
    xf = bandpass(x, fs, 0.5, 40.0)
    r = []
    half = int(0.06 * fs)
    for p in peaks:
        a, b = max(0, p - 2 * half), min(len(xf), p + half)
        r.append(a + int(np.argmax(np.abs(xf[a:b]))))
    return np.unique(np.asarray(r, dtype=int))


def heart_rate_bpm(r_peaks: np.ndarray, fs: float) -> float:
    if len(r_peaks) < 2:
        return float("nan")
    return 60.0 / float(np.median(np.diff(r_peaks)) / fs)
