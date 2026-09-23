"""Parametric synthetic ECG with controllable T-wave alternans.

Every heartbeat is modelled as a sum of five Gaussian waves (P, Q, R, S, T):

    x_beat(t) = sum_k  a_k * exp( -((t - mu_k) / sigma_k)^2 )

This is a simplified, time-domain relative of the dynamical ECG model of
McSharry et al. (2003, IEEE TBME 50:289).  T-wave alternans (TWA) is injected
as an ABAB modulation of the T-wave amplitude:

    a_T[n] = a_T + (-1)^n * alt_mV

so that even and odd beats differ by 2*alt_mV and the *alternans amplitude*
(the quantity the Spectral Method estimates as V_alt) equals alt_mV.

All units: time in seconds, voltage in millivolts (mV).  1 mV = 1000 uV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthConfig:
    fs: int = 500                 # sampling frequency, Hz
    n_beats: int = 128            # 128 beats = standard Spectral-Method window
    heart_rate: float = 75.0      # beats per minute
    hrv_std: float = 0.01         # beat-to-beat RR jitter (s)
    alternans_uv: float = 0.0     # TWA amplitude, microvolts
    t_amp: float = 0.35           # T-wave amplitude, mV (anthracycline stress: flatter T)
    t_width: float = 0.05         # T-wave Gaussian width, s
    noise_uv: float = 10.0        # white (EMG-like) noise std, microvolts
    wander_mv: float = 0.10       # respiratory baseline wander amplitude, mV (0.25 Hz)
    mains_uv: float = 0.0         # 50 Hz power-line interference, microvolts
    mains_hz: float = 50.0
    seed: int | None = 0


# (amplitude mV, centre as fraction of a 0.8 s reference beat, width s)
_WAVES = {
    "P": (0.15, 0.20, 0.030),
    "Q": (-0.10, 0.30, 0.010),
    "R": (1.20, 0.32, 0.015),
    "S": (-0.25, 0.35, 0.015),
}


def _beat(t: np.ndarray, rr: float, t_amp: float, t_width: float) -> np.ndarray:
    """One PQRST complex on local time axis t (s, 0 = beat onset)."""
    x = np.zeros_like(t)
    for a, mu, s in _WAVES.values():
        x += a * np.exp(-(((t - mu)) / s) ** 2)
    # QT shortens with heart rate: T-wave centre scaled by sqrt(RR) (Bazett-like).
    t_mu = 0.32 + 0.23 * np.sqrt(rr / 0.8)
    x += t_amp * np.exp(-(((t - t_mu)) / t_width) ** 2)
    return x


def generate_ecg(cfg: SynthConfig | None = None):
    """Return (time_s, ecg_mV, r_peak_indices) for a single synthetic lead."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    rr_mean = 60.0 / cfg.heart_rate
    rr = np.clip(rng.normal(rr_mean, cfg.hrv_std, cfg.n_beats), 0.3, 2.0)
    onsets = np.concatenate([[0.0], np.cumsum(rr)[:-1]])
    total = float(onsets[-1] + rr[-1] + 0.5)
    n = int(total * cfg.fs)
    t = np.arange(n) / cfg.fs
    x = np.zeros(n)
    r_idx = []
    alt_mv = cfg.alternans_uv / 1000.0
    for k, (t0, rrk) in enumerate(zip(onsets, rr)):
        i0 = int(t0 * cfg.fs)
        i1 = min(n, i0 + int(1.2 * cfg.fs))
        local = t[i0:i1] - t0
        amp_t = cfg.t_amp + ((-1) ** k) * alt_mv
        x[i0:i1] += _beat(local, rrk, amp_t, cfg.t_width)
        r_idx.append(i0 + int(round(_WAVES["R"][1] * cfg.fs)))
    x += cfg.wander_mv * np.sin(2 * np.pi * 0.25 * t)
    x += (cfg.mains_uv / 1000.0) * np.sin(2 * np.pi * cfg.mains_hz * t)
    x += rng.normal(0.0, cfg.noise_uv / 1000.0, n)
    return t, x, np.asarray(r_idx)


# ----------------------------------------------------------------------------- 12-lead
# Physics: to first approximation the heart is a single current dipole H(t) located in a
# homogeneous volume conductor (Einthoven's model). A lead with direction vector e measures
# the projection  V_lead(t) = e . H(t).  Axes: x = patient's left, y = inferior, z = anterior.
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_PRECORDIAL = np.array([[-0.50, 0.00, 0.85], [-0.20, 0.00, 1.00], [0.20, 0.05, 1.00],
                        [0.55, 0.10, 0.85], [0.85, 0.10, 0.50], [1.00, 0.10, 0.15]])
# dipole components of one beat: (amplitude mV, centre s, width s, direction)
_DIPOLE = [
    (0.15, 0.20, 0.030, (0.50, 0.60, 0.10)),   # P: atria, down and to the left
    (0.25, 0.30, 0.010, (-0.40, 0.10, 0.60)),  # septal activation (left -> right, anterior)
    (1.40, 0.32, 0.015, (0.60, 0.75, -0.10)),  # main LV free-wall depolarisation, axis ~50 deg
    (0.35, 0.35, 0.013, (-0.30, -0.60, -0.40)),  # late basal activation
]
_T_DIR = np.array([0.60, 0.70, 0.35])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def heart_vector(t: np.ndarray, rr: float, t_amp: float, t_width: float) -> np.ndarray:
    """(len(t), 3) heart-dipole trajectory for one beat."""
    H = np.zeros((len(t), 3))
    for a, mu, s, d in _DIPOLE:
        H += a * np.exp(-((t - mu) / s) ** 2)[:, None] * _unit(d)
    t_mu = 0.32 + 0.23 * np.sqrt(rr / 0.8)
    H += t_amp * np.exp(-((t - t_mu) / t_width) ** 2)[:, None] * _unit(_T_DIR)
    return H


def dipole_to_12lead(H: np.ndarray) -> np.ndarray:
    """Project a (n, 3) dipole onto the 12 standard leads (Einthoven + Goldberger + Wilson)."""
    e_I, e_II, e_III = np.array([1.0, 0, 0]), np.array([0.5, np.sqrt(3) / 2, 0]), np.array([-0.5, np.sqrt(3) / 2, 0])
    I, II, III = H @ e_I, H @ e_II, H @ e_III
    aVR, aVL, aVF = -(I + II) / 2, (I - III) / 2, (II + III) / 2
    V = H @ np.array([_unit(v) for v in _PRECORDIAL]).T
    return np.column_stack([I, II, III, aVR, aVL, aVF, V])


def generate_12lead(cfg: SynthConfig | None = None, return_dipole: bool = False):
    """Synthetic 12-lead ECG from a moving dipole. Returns (t, X[n,12], r_idx[, H])."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    rr_mean = 60.0 / cfg.heart_rate
    rr = np.clip(rng.normal(rr_mean, cfg.hrv_std, cfg.n_beats), 0.3, 2.0)
    onsets = np.concatenate([[0.0], np.cumsum(rr)[:-1]])
    n = int((onsets[-1] + rr[-1] + 0.5) * cfg.fs)
    t = np.arange(n) / cfg.fs
    H = np.zeros((n, 3))
    alt_mv = cfg.alternans_uv / 1000.0
    for k, (t0, rrk) in enumerate(zip(onsets, rr)):
        i0 = int(t0 * cfg.fs)
        i1 = min(n, i0 + int(1.2 * cfg.fs))
        H[i0:i1] += heart_vector(t[i0:i1] - t0, rrk, cfg.t_amp + ((-1) ** k) * alt_mv, cfg.t_width)
    X = dipole_to_12lead(H)
    X += cfg.wander_mv * np.sin(2 * np.pi * 0.25 * t)[:, None]
    X += rng.normal(0.0, cfg.noise_uv / 1000.0, X.shape)
    r_idx = (onsets * cfg.fs).astype(int) + int(round(0.32 * cfg.fs))
    return (t, X, r_idx, H) if return_dipole else (t, X, r_idx)
