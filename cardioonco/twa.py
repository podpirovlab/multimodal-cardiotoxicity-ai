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
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class TWAResult:
    n_beats: int
    heart_rate_bpm: float
    v_alt_uv: float           # Spectral-Method alternans voltage (RMS over the ST-T window)
    v_alt_peak_uv: float      # alternans amplitude at the most alternating sample of ST-T
    k_score: float            # Spectral-Method signal-to-noise ratio
    noise_uv: float           # sqrt of mean noise-band power
    mma_uv: float             # Modified-Moving-Average TWA estimate
    positive: bool            # V_alt >= 1.9 uV and K >= 3
    spectrum_f: list          # cycles/beat
    spectrum_p_uv2: list      # aggregate power, uV^2

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
    return float(np.max(np.abs(A - B)) / 2.0)   # half-difference, comparable to V_alt


def analyze(x_mv: np.ndarray, fs: float, r_peaks: np.ndarray | None = None,
            n_beats: int = 128) -> TWAResult:
    """Full pipeline on one ECG lead: filter -> R peaks -> beat matrix -> TWA."""
    from .preprocess import bandpass, detect_r_peaks, heart_rate_bpm

    xf = bandpass(x_mv, fs, 0.5, 40.0)
    if r_peaks is None:
        r_peaks = detect_r_peaks(x_mv, fs)
    hr = heart_rate_bpm(r_peaks, fs)
    rr = 60.0 / hr if np.isfinite(hr) else 0.8
    # ST-T window adapts to heart rate (QT ~ sqrt(RR))
    scale = np.sqrt(rr / 0.8)
    m = beat_matrix(xf, r_peaks, fs, 0.10 * scale, 0.42 * scale)
    m = m[: n_beats] if len(m) >= n_beats else m
    if len(m) % 2:                          # keep an even count so f = 0.5 is a bin
        m = m[:-1]
    if len(m) < 16:
        raise ValueError(f"need at least 16 beats for TWA, got {len(m)}")
    f, P, v_alt, k, noise, v_peak = spectral_twa(m)
    return TWAResult(
        n_beats=len(m), heart_rate_bpm=float(hr), v_alt_uv=v_alt, v_alt_peak_uv=v_peak, k_score=k,
        noise_uv=noise, mma_uv=mma_twa(m), positive=bool(v_alt >= 1.9 and k >= 3.0),
        spectrum_f=f.round(5).tolist(), spectrum_p_uv2=P.round(6).tolist(),
    )
