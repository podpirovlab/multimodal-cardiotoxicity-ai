"""QT interval and heart-rate-corrected QTc from a 12-lead (or single-lead) ECG.

Method
------
1. Filter 0.5-40 Hz (zero phase), find R peaks shared by all leads, and build one median beat
   per lead from beats whose R-R interval lies within 20% of the median (ectopic beats and
   their neighbours are left out).  Each median beat is measured from its own PR segment
   (R-100 to R-40 ms), as for TWA.
2. QRS onset, one for all leads: the spatial velocity v(t) = sum over leads of |dx/dt| rises
   steeply at the start of the QRS.  Going back from R, the onset is the first sample where v
   falls below 10% of its peak in the QRS.
3. T-wave end in each lead by the tangent method (Lepeschkin & Surawicz, 1952): the tangent at
   the steepest point of the descending limb of the T wave, extended to the isoelectric line.
   Leads whose T wave is smaller than 0.05 mV are not used.
4. The global T end combines the leads (``T_END_RULE``, chosen on PTB-XL fold 9 against the
   GE 12SL measurements of PTB-XL+, see scripts/validate_qt.py); QT = T end - QRS onset.
5. Rate correction with the median R-R interval (seconds): Fridericia QTcF = QT / RR^(1/3),
   the formula the 2022 ESC cardio-oncology guidelines use, and Bazett QTcB = QT / RR^(1/2).
6. A recording longer than 20 s is measured in 10-s segments, the length of the validation ECGs
   (measure_qt_segments).  Below 400 Hz the leads are first interpolated to at least 500 Hz, as
   for TWA; on fold 9 the 100 Hz copies of the same ECGs then agreed with 12SL as well as the
   500 Hz originals (scripts/validate_qt.py --hz100).

Research software, not a medical device: an automatic QT can be wrong in wide-QRS rhythms,
flat or biphasic T waves, U waves and atrial fibrillation, where cardiologists measure by hand.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

T_END_RULE = "median"                 # how per-lead T ends become one global T end (chosen on fold 9)
MIN_T_AMPLITUDE_MV = 0.05
# The tangent method places the T end earlier than the GE 12SL algorithm.  QT is reported on the
# 12SL scale by adding this offset, the median difference on PTB-XL fold 9 (scripts/validate_qt.py);
# fold 10 then tests it.  qt_tangent_ms keeps the uncorrected value.
QT_OFFSET_MS = 35.2
# A measurement is flagged unreliable when fewer than MIN_LEADS leads give a T end, or when their
# T ends spread over more than MAX_T_END_IQR_MS (interquartile range), as in flutter, fibrillation,
# flat or biphasic T waves; the threshold was chosen on fold 9.  The comparison with 12SL is for
# 12-lead ECGs, so a QT from one or a few leads is always flagged: it has not been validated.
MIN_LEADS = 6
MAX_T_END_IQR_MS = 40.0
# PTB-XL ECGs last 10 s, the length the method was validated on; a longer recording is measured
# in consecutive 10-s segments (see measure_qt_segments).
SEGMENT_S = 10.0


@dataclass
class QTResult:
    qt_ms: float                      # on the 12SL scale (QT_OFFSET_MS added)
    qtcf_ms: float                    # Fridericia
    qtcb_ms: float                    # Bazett
    reliable: bool                    # False: leads disagree or too few leads; check by hand
    qt_tangent_ms: float              # the uncorrected tangent-method QT
    rr_ms: float
    heart_rate_bpm: float
    qrs_onset_ms: float               # relative to the R peak (negative)
    t_end_ms: float                   # relative to the R peak
    n_beats: int
    n_leads_used: int
    t_end_iqr_ms: float               # spread of the per-lead T ends
    t_end_per_lead_ms: list           # NaN where the lead was not usable
    segment_start_s: float = 0.0      # where the reported 10-s segment starts (measure_qt_segments)
    n_segments: int = 1               # segments measured
    n_segments_reliable: int = 0      # of which reliable

    def as_dict(self) -> dict:
        return asdict(self)


def median_beats(leads_mv: np.ndarray, fs: float, r_peaks: np.ndarray, pre_s: float = 0.30, post_s: float = 0.75):
    """(n_leads, L) median beats around R, the time axis in seconds, and the beats used."""
    from .preprocess import bandpass
    x = bandpass(np.atleast_2d(leads_mv).astype(float), fs, 0.5, 40.0)
    rr = np.diff(r_peaks)
    med = np.median(rr)
    ok = np.zeros(len(r_peaks), bool)
    for i in range(1, len(r_peaks) - 1):          # regular before and after: no ectopic beat nearby
        ok[i] = abs(rr[i - 1] - med) < 0.2 * med and abs(rr[i] - med) < 0.2 * med
    a, b = int(round(pre_s * fs)), int(round(min(post_s, 0.95 * med / fs) * fs))
    use = [r for r, k in zip(r_peaks, ok) if k and r - a >= 0 and r + b <= x.shape[1]]
    if len(use) < 3:
        raise ValueError(f"too few regular beats for QT: {len(use)}")
    stack = np.stack([x[:, r - a:r + b] for r in use])          # (beats, leads, L)
    beats = np.median(stack, axis=0)
    t = (np.arange(-a, b)) / fs
    pr = (t >= -0.10) & (t <= -0.04)
    beats = beats - np.median(beats[:, pr], axis=1, keepdims=True)
    return beats, t, len(use), med / fs


def qrs_onset(beats: np.ndarray, t: np.ndarray, fs: float) -> float:
    v = np.abs(np.gradient(beats, axis=1)).sum(axis=0) * fs           # mV/s, summed over leads
    qrs = (t >= -0.06) & (t <= 0.06)
    thr = 0.10 * v[qrs].max()
    i = int(np.argmin(np.abs(t)))                                      # the R peak
    lo = int(np.searchsorted(t, -0.15))
    while i > lo and v[i] >= thr:
        i -= 1
    return float(t[i])


def tangent(beat: np.ndarray, t: np.ndarray, fs: float, rr_s: float) -> dict | None:
    """The tangent-method construction on one median beat, or None when T is not measurable.

    The tangent is taken at the steepest point of the T wave's own descending limb, between its
    peak and the point where it has fallen to 15% of the peak, so that it cannot catch the P wave
    of the next beat; the T end must lie at least 120 ms before the next R peak.  Returns the T
    peak, the point j of the tangent, its slope (mV/s, on the T wave made positive), the sign of
    the T wave and the T end, all times in seconds from R."""
    t_max = min(rr_s - 0.12, t[-1])
    lo = int(np.searchsorted(t, 0.10 * np.sqrt(rr_s / 0.8) + 0.05))
    hi = int(np.searchsorted(t, min(0.65 * rr_s + 0.05, t_max)))
    if hi - lo < int(0.08 * fs):
        return None
    seg = beat[lo:hi]
    k = int(np.argmax(np.abs(seg)))
    peak, amp = lo + k, seg[k]
    if abs(amp) < MIN_T_AMPLITUDE_MV:
        return None
    sign = float(np.sign(amp))
    s = sign * beat                               # make the T wave positive
    d = np.gradient(s) * fs
    stop = int(np.searchsorted(t, t_max))
    below = np.flatnonzero(s[peak:stop] < 0.15 * abs(amp))
    if len(below) == 0:
        return None
    stop = peak + int(below[0])
    if stop <= peak + 2:
        return None
    j = peak + int(np.argmin(d[peak:stop + 1]))   # steepest point of the descending limb
    slope = d[j]
    if slope >= 0:
        return None
    te = t[j] - s[j] / slope                      # where the tangent meets the baseline (0)
    if not t[peak] < te <= t_max:
        return None
    return {"t_peak": float(t[peak]), "t_j": float(t[j]), "v_j": float(beat[j]), "slope": float(sign * slope),
            "sign": sign, "t_end": float(te)}


def t_end_tangent(beat: np.ndarray, t: np.ndarray, fs: float, rr_s: float) -> float:
    """T-wave end of one median beat by the tangent method (see tangent), or NaN."""
    c = tangent(beat, t, fs, rr_s)
    return float("nan") if c is None else c["t_end"]


def combine(t_ends: np.ndarray, rule: str = T_END_RULE) -> float:
    v = np.sort(t_ends[np.isfinite(t_ends)])
    if len(v) == 0:
        return float("nan")
    if rule == "median":
        return float(np.median(v))
    if rule == "p75":
        return float(np.percentile(v, 75))
    if rule == "latest":
        return float(v[-1])
    if rule == "second_latest":
        return float(v[-2] if len(v) >= 2 else v[-1])
    raise ValueError(rule)


def measure_qt(leads_mv: np.ndarray, fs: float, r_peaks: np.ndarray | None = None, rule: str = T_END_RULE) -> QTResult:
    """QT, QTcF and QTcB of an ECG given as (n_leads, n_samples) or (n_samples,) in mV."""
    from scipy.signal import resample_poly

    from .preprocess import consensus_r_peaks
    from .twa import analysis_rate
    leads = np.atleast_2d(np.asarray(leads_mv, dtype=float))
    up = analysis_rate(fs)               # below 400 Hz: interpolate first, as for TWA
    if up > 1:
        leads, fs = resample_poly(leads, up, 1, axis=1), fs * up
        r_peaks = None if r_peaks is None else np.asarray(r_peaks) * up
    if r_peaks is None:
        r_peaks = consensus_r_peaks(leads, fs)
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 5:
        raise ValueError("too few beats for QT")
    beats, t, n, rr_s = median_beats(leads, fs, r_peaks)
    on = qrs_onset(beats, t, fs)
    ends = np.array([t_end_tangent(b, t, fs, rr_s) for b in beats])
    te = combine(ends, rule)
    if not np.isfinite(te):
        raise ValueError("no lead with a measurable T wave")
    raw = (te - on) * 1000
    qt = raw + QT_OFFSET_MS
    good = ends[np.isfinite(ends)] * 1000
    iqr = float(np.subtract(*np.percentile(good, [75, 25]))) if len(good) >= 2 else float("nan")
    n_used = int(len(good))
    reliable = bool(n_used >= MIN_LEADS and iqr <= MAX_T_END_IQR_MS)
    return QTResult(qt_ms=qt, qtcf_ms=qt / rr_s ** (1 / 3), qtcb_ms=qt / rr_s ** 0.5, reliable=reliable,
                    qt_tangent_ms=raw, rr_ms=rr_s * 1000, heart_rate_bpm=60 / rr_s, qrs_onset_ms=on * 1000,
                    t_end_ms=te * 1000, n_beats=n, n_leads_used=n_used, t_end_iqr_ms=iqr,
                    t_end_per_lead_ms=[float(e * 1000) for e in ends], n_segments_reliable=int(reliable))


def segment_starts(n_samples: int, fs: float, seg_s: float = SEGMENT_S, max_segments: int = 60) -> list:
    """Start samples of the consecutive seg_s segments (at most max_segments, evenly spread);
    [0] with the whole recording as one segment when it is shorter than two segments."""
    seg = int(round(seg_s * fs))
    k = n_samples // seg
    if k < 2:
        return [0]
    idx = np.unique(np.floor(np.linspace(0, k - 1, min(k, max_segments)) + 0.5).astype(int))   # halves up, as in JS
    return [int(i * seg) for i in idx]


def measure_qt_segments(leads_mv: np.ndarray, fs: float, seg_s: float = SEGMENT_S, max_segments: int = 60) -> QTResult:
    """QT of a recording of any length, measured on 10-s segments as on a resting 12-lead ECG.

    Each segment gets its own R peaks and median beats.  The reported segment is the one with the
    median QTcF among the reliable segments (among all measured ones if none is reliable), so QT,
    QTcF, QTcB and the heart rate all come from one real stretch of the recording."""
    leads = np.atleast_2d(np.asarray(leads_mv, dtype=float))
    starts = segment_starts(leads.shape[1], fs, seg_s, max_segments)
    seg = leads.shape[1] if len(starts) == 1 else int(round(seg_s * fs))
    found = []
    for s in starts:
        try:
            found.append((s, measure_qt(leads[:, s:s + seg], fs)))
        except ValueError:
            continue
    if not found:
        raise ValueError("QT could not be measured in any segment")
    good = [f for f in found if f[1].reliable]
    pool = sorted(good or found, key=lambda f: f[1].qtcf_ms)
    s, res = pool[(len(pool) - 1) // 2]
    res.segment_start_s, res.n_segments, res.n_segments_reliable = s / fs, len(found), len(good)
    return res
