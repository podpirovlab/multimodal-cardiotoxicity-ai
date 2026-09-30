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
6. removal of detections that split one normal R-R interval in two (tall T waves).
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
    return remove_extra_beats(np.unique(np.asarray(r, dtype=int)), xf, fs)


def consensus_r_peaks(leads_mv: np.ndarray, fs: float, tol_s: float = 0.05) -> np.ndarray:
    """R peaks for a multi-lead recording (leads_mv: n_leads x n_samples).

    Detects on every lead and returns the detections of the lead the others agree with
    most (a beat matches when another lead has one within 50 ms, counted both ways so
    that extra and missed beats both cost).  In a lead whose T wave is taller than its
    QRS the fiducial point can land on the T wave; the leads are simultaneous, so one
    shared set of R peaks is both safer and what 12-lead TWA analysis uses.
    """
    cands = [detect_r_peaks(x, fs) for x in np.atleast_2d(leads_mv)]
    if len(cands) == 1:
        return cands[0]
    tol = tol_s * fs

    def share(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) < 2:
            return 0.0
        i = np.searchsorted(b, a).clip(1, len(b) - 1)
        return float((np.minimum(np.abs(b[i] - a), np.abs(b[i - 1] - a)) <= tol).mean())

    score = [np.median([min(share(a, b), share(b, a)) for j, b in enumerate(cands) if j != i])
             for i, a in enumerate(cands)]
    return cands[int(np.argmax(score))]


def _local_rr(rr: np.ndarray, context: int = 4) -> np.ndarray:
    """Median of the surrounding +-context R-R intervals, for each interval."""
    return np.array([np.median(rr[max(0, i - context):i + context + 1]) for i in range(len(rr))])


def qrs_similarity(xf: np.ndarray, r_peaks: np.ndarray, fs: float, half_s: float = 0.05) -> np.ndarray:
    """Correlation of each detection's +-50 ms neighbourhood with the median QRS."""
    h = int(half_s * fs)
    sim = np.ones(len(r_peaks))
    ok = np.where((r_peaks - h >= 0) & (r_peaks + h <= len(xf)))[0]
    if len(ok) < 3:
        return sim
    seg = np.vstack([xf[r_peaks[i] - h:r_peaks[i] + h] for i in ok])
    seg = seg - seg.mean(axis=1, keepdims=True)
    t = np.median(seg, axis=0)
    t = t - t.mean()
    sim[ok] = seg @ t / (np.linalg.norm(seg, axis=1) * np.linalg.norm(t) + 1e-12)
    return sim


def remove_extra_beats(r_peaks: np.ndarray, xf: np.ndarray | None = None, fs: float | None = None,
                       tol: float = 0.20, context: int = 15, min_similarity: float = 0.90) -> np.ndarray:
    """Drop detections that split one normal cardiac cycle in two.

    A tall, sharp T wave (or a noise spike) can pass the QRS threshold.  It shows up
    as a short interval whose sum with the next one is a normal R-R interval -- the
    "extra beat" pattern of Lipponen & Tarvainen (J Med Eng Technol 2019;43:173).
    A true premature beat is followed by a pause, so the two intervals add up to well
    over one R-R and it is kept.  For TWA this matters: every extra "beat" flips the
    ABAB parity of all the beats after it.  The reference R-R is the median of the
    surrounding +-15 intervals, wide enough that a run of extra detections cannot
    drag it down; a second pass catches what the first one uncovered.  Given the
    filtered signal, a candidate that still looks like the median QRS (correlation
    >= 0.9) is kept whatever its timing: a real early beat must not be deleted.
    """
    r = np.asarray(r_peaks, dtype=int)
    for _ in range(2):
        if len(r) < 4:
            return r
        ref = _local_rr(np.diff(r).astype(float), context)
        sim = qrs_similarity(xf, r, fs) if xf is not None else np.zeros(len(r))
        kept = [r[0]]
        for i in range(1, len(r) - 1):
            cycle = ref[i - 1]
            if (r[i] - kept[-1] < (1 - tol) * cycle and abs(r[i + 1] - kept[-1] - cycle) < tol * cycle
                    and sim[i] < min_similarity):
                continue
            kept.append(r[i])
        kept.append(r[-1])
        r = np.asarray(kept, dtype=int)
    return r


def heart_rate_bpm(r_peaks: np.ndarray, fs: float) -> float:
    if len(r_peaks) < 2:
        return float("nan")
    return 60.0 / float(np.median(np.diff(r_peaks)) / fs)
