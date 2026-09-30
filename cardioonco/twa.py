"""T-wave alternans (TWA) quantification.

TWA is an ABAB... beat-to-beat oscillation of the ST-T segment amplitude,
typically 1-100 microvolts -- far below what the eye can see on paper ECG.

Two standard estimators are implemented:

1. Spectral Method (Smith et al. 1988; Rosenbaum et al. 1994)
   ---------------------------------------------------------
   Align N consecutive beats (N = 128) on their R peaks.  For every sample
   offset k inside the ST-T window take the *beat series*  s_k[n], n = 0..N-1,
   i.e. the ECG value at the same place of every beat.  Remove its mean and
   compute the periodogram

        P_k(f) = | (1/N) * sum_n s_k[n] * exp(-2*pi*i*f*n) |^2 ,   f in cycles/beat.

   Average over k (the "aggregate spectrum").  Alternation with period 2 beats
   concentrates power at f = 0.5 cycles/beat, because exp(-i*pi*n) = (-1)^n.
   With a reference noise band B = [0.44, 0.49] cycles/beat:

        V_alt = sqrt( P(0.5) - mean_B(P) )        (alternans voltage, uV)
        K     = ( P(0.5) - mean_B(P) ) / std_B(P) (signal-to-noise "K-score")

   Conventional positivity criterion: V_alt >= 1.9 uV and K >= 3.
   With this normalisation a pure series a*(-1)^n gives P(0.5) = a^2, so V_alt = a
   (half of the even-minus-odd difference).

2. Modified Moving Average (Nearing & Verrier 2002)
   ------------------------------------------------
   Keep two recursive templates, one for even beats (A) and one for odd beats (B):

        A_n = A_{n-1} + clip( (beat_n - A_{n-1}) / 8 , +-32 uV )

   TWA_MMA = max over the ST-T window of |A - B|.  Robust to single noisy beats.
   This is the full even-minus-odd difference, the scale the clinical MMA
   cutpoints (e.g. 47 uV) refer to -- about twice V_alt for pure alternans.

Whole-recording analysis
------------------------
A recording is scanned with 128-beat windows (hop 16 beats), the way Holter TWA
reports the maximum over the day rather than the first two minutes.  Premature or
misshapen beats would flip the ABAB phase, so they are replaced by the median of
same-parity beats in their window (Armoundas, Heart Rhythm 2012;9:449); a window
with more than 10% such beats is not used.

The result has three outcomes, following the Spectral Method's clinical rules
(Verrier et al., JACC 2011;58:1309 consensus; Bloomfield et al.):
  positive       significant alternans (V_alt >= 1.9 uV, K >= 3) in a window with
                 heart rate <= 110 bpm and noise <= 1.8 uV;
  negative       no significant alternans, with at least one clean window at
                 heart rate >= 105 bpm;
  indeterminate  everything else -- too noisy, too many ectopic beats, alternans
                 only above 110 bpm, or the heart rate never reached 105 bpm.
Those rules were written for exercise tests; at rest most negatives are
indeterminate by design, and the reason is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# Below 64 beats the 0.44-0.49 cycles/beat noise band holds too few spectral bins
# for a usable noise estimate (at 16-32 beats it is 0-1 bins and K becomes NaN).
MIN_BEATS = 64
STANDARD_BEATS = 128
WINDOW_HOP = 16
MAX_ECTOPIC_FRACTION = 0.10
V_ALT_MIN_UV, K_MIN = 1.9, 3.0
NOISE_MAX_UV = 1.8
HR_ONSET_MAX, HR_NEGATIVE_MIN = 110.0, 105.0


@dataclass
class TWAResult:
    n_beats: int              # beats analysed across the whole recording
    heart_rate_bpm: float     # median over the recording
    v_alt_uv: float           # Spectral-Method V_alt of the reported window (RMS over ST-T)
    v_alt_peak_uv: float      # alternans amplitude at the most alternating sample of ST-T
    k_score: float            # Spectral-Method signal-to-noise ratio of the reported window
    noise_uv: float           # sqrt of mean noise-band power of the reported window
    mma_uv: float             # Modified-Moving-Average TWA, max over valid windows
    positive: bool            # outcome == "positive"
    outcome: str              # "positive" | "negative" | "indeterminate"
    reason: str               # why, in a short machine-readable code
    estimate_uv: float        # max V_alt over windows with K >= 3 (0 if none): the ranking estimate
    window_start_beat: int
    window_hr_bpm: float
    ectopic_fraction: float   # share of replaced beats in the reported window
    spectrum_f: list          # cycles/beat, reported window
    spectrum_p_uv2: list      # aggregate power, uV^2, reported window
    windows: list = field(default_factory=list)   # per-window summary for trend plots

    def as_dict(self) -> dict:
        return asdict(self)


def beat_matrix(x: np.ndarray, r_peaks: np.ndarray, fs: float,
                start_s: float = 0.10, end_s: float = 0.42) -> np.ndarray:
    """Stack the ST-T windows [R+start_s, R+end_s] of consecutive beats -> (n_beats, L)."""
    a, b = int(start_s * fs), int(end_s * fs)
    rows = [x[r + a:r + b] for r in r_peaks if r + b <= len(x) and r + a >= 0]
    if not rows:
        return np.empty((0, b - a))
    m = np.vstack(rows)
    # remove each beat's own baseline (median of the window) -> kills residual wander
    return m - np.median(m, axis=1, keepdims=True)


def spectral_twa(beats_mv: np.ndarray, noise_band=(0.44, 0.49)):
    """Aggregate beat-series spectrum.

    Returns (f, P_uv2, V_alt_uv, K, noise_uv, V_alt_peak_uv).
    V_alt is the RMS alternans over the whole ST-T window (aggregate spectrum);
    V_alt_peak is the alternans amplitude at the single most alternating sample.
    """
    n = beats_mv.shape[0]
    s = (beats_mv - beats_mv.mean(axis=0, keepdims=True)) * 1000.0  # -> uV
    X = np.fft.rfft(s, axis=0) / n                 # (n_freq, L)
    Pk = np.abs(X) ** 2                             # per-sample spectra
    P = Pk.mean(axis=1)                             # aggregate spectrum
    f = np.fft.rfftfreq(n, d=1.0)
    i_alt = int(np.argmin(np.abs(f - 0.5)))
    band = (f >= noise_band[0]) & (f <= noise_band[1])
    mu, sd = float(P[band].mean()), float(P[band].std(ddof=1) + 1e-12)
    excess = max(P[i_alt] - mu, 0.0)
    peak_excess = max(float(Pk[i_alt].max() - Pk[band].mean()), 0.0)
    return (f, P, float(np.sqrt(excess)), float((P[i_alt] - mu) / sd),
            float(np.sqrt(mu)), float(np.sqrt(peak_excess)))


def mma_twa(beats_mv: np.ndarray, step: float = 1 / 8, limit_uv: float = 32.0) -> float:
    b = beats_mv * 1000.0
    A, B = b[0].copy(), b[1].copy()
    for i in range(2, len(b)):
        if i % 2 == 0:
            A += np.clip((b[i] - A) * step, -limit_uv, limit_uv)
        else:
            B += np.clip((b[i] - B) * step, -limit_uv, limit_uv)
    return float(np.max(np.abs(A - B)))


def flag_ectopic(r_peaks: np.ndarray, full_beats_mv: np.ndarray,
                 rr_tolerance: float = 0.20, min_correlation: float = 0.90,
                 rr_context: int = 4) -> np.ndarray:
    """Flag beats whose preceding R-R interval is >20% off the median of the surrounding
    intervals, or whose shape correlates < 0.9 with the median beat (premature,
    post-ectopic or artefactual).  The RR reference is local (+-4 intervals) so a heart
    rate that drifts during the recording is not mistaken for ectopy."""
    from .preprocess import _local_rr

    n = len(r_peaks)
    flags = np.zeros(n, dtype=bool)
    rr = np.diff(r_peaks).astype(float)
    if len(rr):
        ref = _local_rr(rr, rr_context)
        flags[1:] |= np.abs(rr - ref) > rr_tolerance * ref
    template = np.median(full_beats_mv, axis=0)
    t = template - template.mean()
    for i, row in enumerate(full_beats_mv):
        v = row - row.mean()
        denom = np.linalg.norm(v) * np.linalg.norm(t)
        if denom == 0 or float(v @ t) / denom < min_correlation:
            flags[i] = True
    return flags


def fill_missed_beats(r_peaks: np.ndarray, tol: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """A gap of about two R-R intervals is a beat the detector missed (small QRS, noise).
    A placeholder goes in the middle so that the beats after it keep their ABAB parity;
    it is returned flagged and later replaced like an ectopic beat."""
    from .preprocess import _local_rr

    r = np.asarray(r_peaks, dtype=int)
    if len(r) < 4:
        return r, np.zeros(len(r), dtype=bool)
    rr = np.diff(r).astype(float)
    ref = _local_rr(rr)
    out, filled = [r[0]], [False]
    for i, gap in enumerate(rr):
        if abs(gap - 2 * ref[i]) < 2 * tol * ref[i]:
            out.append(r[i] + int(round(gap / 2)))
            filled.append(True)
        out.append(r[i + 1])
        filled.append(False)
    return np.asarray(out, dtype=int), np.asarray(filled, dtype=bool)


def align_beats(xf: np.ndarray, r_peaks: np.ndarray, fs: float,
                half_s: float = 0.05, max_shift_s: float = 0.02, passes: int = 2) -> np.ndarray:
    """Move each fiducial point to where the beat's QRS best matches the median QRS
    (cross-correlation over +-20 ms).  The largest |ECG| sample can jump between R and
    S from beat to beat; microvolt TWA needs the beats superimposed to a sample."""
    h, m = int(half_s * fs), int(max_shift_s * fs)
    r = np.asarray(r_peaks, dtype=int).copy()
    ok = np.where((r - h - m >= 0) & (r + h + m < len(xf)))[0]
    if len(ok) < 3 or m == 0:
        return r
    for _ in range(passes):
        template = np.median(np.vstack([xf[r[i] - h:r[i] + h] for i in ok]), axis=0)
        template -= template.mean()
        for i in ok:
            c = np.correlate(xf[r[i] - h - m:r[i] + h + m], template, mode="valid")
            r[i] += int(np.argmax(c)) - m
        ok = ok[(r[ok] - h - m >= 0) & (r[ok] + h + m < len(xf))]
    return r


def _replace_flagged(beats: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Replace flagged beats by the median of same-parity beats, keeping the ABAB phase."""
    out = beats.copy()
    for parity in (0, 1):
        idx = np.arange(parity, len(out), 2)
        good = idx[~flags[idx]]
        bad = idx[flags[idx]]
        if len(bad) and len(good):
            out[bad] = np.median(out[good], axis=0)
    return out


def _window_starts(n: int, size: int, hop: int) -> list[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, hop))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


def analyze(x_mv: np.ndarray, fs: float, r_peaks: np.ndarray | None = None,
            n_beats: int = STANDARD_BEATS, hop: int = WINDOW_HOP) -> TWAResult:
    """Whole-recording pipeline on one ECG lead: filter -> R peaks -> ectopy control ->
    sliding 128-beat Spectral-Method windows -> three-outcome decision."""
    from .preprocess import bandpass, detect_r_peaks, heart_rate_bpm

    xf = bandpass(x_mv, fs, 0.5, 40.0)
    if r_peaks is None:
        r_peaks = detect_r_peaks(x_mv, fs)
    r_peaks = np.asarray(r_peaks, dtype=int)
    hr = heart_rate_bpm(r_peaks, fs)
    r_peaks, filled = fill_missed_beats(r_peaks)
    r_peaks = align_beats(xf, r_peaks, fs)
    rr = 60.0 / hr if np.isfinite(hr) else 0.8
    scale = np.sqrt(rr / 0.8)                       # ST-T window adapts to heart rate (QT ~ sqrt(RR))
    a_st, b_st = int(0.10 * scale * fs), int(0.42 * scale * fs)
    # morphology window for ectopy: P-QRS through the end of ST-T, never into the next beat
    a_full, b_full = int(-0.10 * fs), b_st
    lo, hi = min(a_st, a_full), max(b_st, b_full)
    inside = (r_peaks + lo >= 0) & (r_peaks + hi <= len(xf))
    keep, filled = r_peaks[inside], filled[inside]

    usable = len(keep) - (len(keep) % 2)
    if usable < MIN_BEATS:
        raise ValueError(f"recording too short for TWA: {usable} usable beats, at least {MIN_BEATS} needed")

    st = np.vstack([xf[r + a_st:r + b_st] for r in keep])
    # baseline of each beat: its isoelectric PR segment (R-80..R-40 ms).  Subtracting
    # the ST-T window's own median, as v0.4 did, also subtracted part of the alternans.
    a_pr, b_pr = int(-0.08 * fs), int(-0.04 * fs)
    st = st - np.array([np.median(xf[r + a_pr:r + b_pr]) for r in keep])[:, None]
    full = np.vstack([xf[r + a_full:r + b_full] for r in keep])
    flags = flag_ectopic(keep, full) | filled

    size = min(n_beats, usable)
    size -= size % 2                                 # even count so f = 0.5 is a bin
    windows = []
    for s0 in _window_starts(len(keep), size, hop):
        wflags = flags[s0:s0 + size]
        beats = _replace_flagged(st[s0:s0 + size], wflags)
        f, P, v_alt, k, noise, v_peak = spectral_twa(beats)
        w_rr = np.median(np.diff(keep[s0:s0 + size])) / fs
        windows.append({"start_beat": int(s0), "hr": float(60.0 / w_rr), "v_alt": v_alt, "k": k,
                        "noise": noise, "v_peak": v_peak, "ectopic": float(wflags.mean()),
                        "mma": mma_twa(beats), "_f": f, "_P": P})
    if not all(np.isfinite(w["k"]) for w in windows):
        raise ValueError("noise band estimate unavailable; the result would be meaningless")

    valid = [w for w in windows if w["ectopic"] <= MAX_ECTOPIC_FRACTION]
    significant = [w for w in valid if w["k"] >= K_MIN and w["v_alt"] >= V_ALT_MIN_UV]
    sig_k = [w for w in valid if w["k"] >= K_MIN]
    estimate = max((w["v_alt"] for w in sig_k), default=0.0)

    if not valid:
        outcome, reason = "indeterminate", "ectopy"
    elif any(w["hr"] <= HR_ONSET_MAX and w["noise"] <= NOISE_MAX_UV for w in significant):
        outcome, reason = "positive", "criterion_met"
    elif any(w["hr"] > HR_ONSET_MAX for w in significant):
        outcome, reason = "indeterminate", "hr_too_high"
    elif significant:
        outcome, reason = "indeterminate", "noise"
    else:
        clean = [w for w in valid if w["noise"] <= NOISE_MAX_UV]
        if not clean:
            outcome, reason = "indeterminate", "noise"
        elif max(w["hr"] for w in clean) >= HR_NEGATIVE_MIN:
            outcome, reason = "negative", "criterion_not_met"
        else:
            outcome, reason = "indeterminate", "hr_too_low"

    pool = significant or valid or windows
    best = max(pool, key=lambda w: w["v_alt"])
    return TWAResult(
        n_beats=int(len(keep)), heart_rate_bpm=float(hr), v_alt_uv=best["v_alt"], v_alt_peak_uv=best["v_peak"],
        k_score=best["k"], noise_uv=best["noise"],
        mma_uv=max((w["mma"] for w in valid), default=best["mma"]),
        positive=outcome == "positive", outcome=outcome, reason=reason, estimate_uv=float(estimate),
        window_start_beat=best["start_beat"], window_hr_bpm=best["hr"], ectopic_fraction=best["ectopic"],
        spectrum_f=best["_f"].round(5).tolist(), spectrum_p_uv2=best["_P"].round(6).tolist(),
        windows=[{k: round(v, 4) if isinstance(v, float) else v for k, v in w.items() if not k.startswith("_")}
                 for w in windows],
    )
