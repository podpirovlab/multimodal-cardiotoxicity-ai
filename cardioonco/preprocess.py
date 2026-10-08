"""ECG pre-processing: filtering and R-peak detection.

Filtering
---------
* Band-pass 0.5-40 Hz, 4th-order Butterworth, applied forward and backward
  (``filtfilt``) so the phase response cancels: zero phase distortion, which
  matters because TWA is measured on the *shape* of the ST-T segment.
  The 0.5 Hz high-pass removes respiratory baseline wander (~0.1-0.4 Hz);
  the 40 Hz low-pass removes EMG noise while keeping the QRS energy (5-25 Hz).
* Optional IIR notch at 50/60 Hz for power-line interference.

R-peak detection (Elgendi, 2013)
--------------------------------
1. band-pass 8-20 Hz (3rd-order Butterworth, zero phase) and squaring;
2. two moving averages of the squared signal: one as long as a QRS complex (97 ms),
   one as long as a heartbeat (611 ms);
3. a QRS is a block of at least 97 ms where the short average exceeds the long one plus
   0.08 times the mean of the squared signal.  The comparison is local: each beat is
   judged against the energy around it, so small beats next to large ectopic beats are
   still found;
4. the R peak is the largest |ECG| in the block, refined on the 0.5-40 Hz signal from
   120 ms before to 60 ms after it; detections closer than 250 ms are one beat;
5. removal of detections that split one normal R-R interval in two (tall T waves).

Constants are those of the paper (W1 = 97 ms, W2 = 611 ms, beta = 0.08); none was tuned
here.  Up to version 1.1 the detector was a Pan-Tompkins variant with one fixed threshold
for the whole recording, which missed 9.3% of the beats of the MIT-BIH Arrhythmia
Database; the adaptive Pan-Tompkins thresholds of the 1985 paper were also tried and lost
every small beat of a bigeminy whose ectopic beats are three times larger
(scripts/validate_rpeaks.py, README section 4.3).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


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


def qrs_blocks(x: np.ndarray, fs: float, w1_s: float = 0.097, w2_s: float = 0.611,
               beta: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Elgendi (2013) two-moving-average QRS detection.

    Returns the index of the largest |8-20 Hz signal| in every block of interest, and the
    8-20 Hz signal itself.  A block is a run of at least ``w1_s`` seconds where the
    QRS-length moving average of the squared signal exceeds the beat-length moving average
    plus ``beta`` times the mean of the squared signal.
    """
    f = bandpass(x, fs, 8.0, 20.0, order=3)
    y = f ** 2
    n1, n2 = max(1, int(round(w1_s * fs))), max(1, int(round(w2_s * fs)))
    ma_qrs = np.convolve(y, np.ones(n1) / n1, mode="same")
    ma_beat = np.convolve(y, np.ones(n2) / n2, mode="same")
    on = np.concatenate([[False], ma_qrs > ma_beat + beta * y.mean(), [False]])
    edges = np.flatnonzero(np.diff(on.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2]
    peaks = [s0 + int(np.argmax(np.abs(f[s0:e0]))) for s0, e0 in zip(starts, ends) if e0 - s0 >= n1]
    return np.asarray(peaks, dtype=int), f


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Return sample indices of R peaks in a single-lead ECG (mV)."""
    blocks, _ = qrs_blocks(x, fs)
    xf = bandpass(x, fs, 0.5, 40.0)
    half = int(0.06 * fs)
    r: list[int] = []
    for p in blocks:                      # refine to the largest |ECG| near the block peak
        a, b = max(0, p - 2 * half), min(len(xf), p + half)
        q = a + int(np.argmax(np.abs(xf[a:b])))
        if r and q - r[-1] < int(0.25 * fs):   # one beat cannot follow another within 250 ms
            if abs(xf[q]) > abs(xf[r[-1]]):
                r[-1] = q
        else:
            r.append(q)
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
