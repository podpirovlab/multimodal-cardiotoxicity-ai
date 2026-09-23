"""Unit tests for the signal-processing and TWA mathematics."""
import numpy as np
import pytest

from cardioonco.preprocess import detect_r_peaks, heart_rate_bpm
from cardioonco.synth import SynthConfig, generate_ecg
from cardioonco.twa import analyze, spectral_twa


def test_pure_alternating_series_gives_exact_amplitude():
    # s[n] = a * (-1)^n  ->  P(0.5) = a^2  ->  V_alt = a
    a_mv = 0.010                           # 10 uV
    n, L = 128, 50
    alt = a_mv * (-1.0) ** np.arange(n)
    rng = np.random.default_rng(0)
    beats = alt[:, None] + rng.normal(0, 1e-5, (n, L))
    _, _, v_alt, k, _, _ = spectral_twa(beats)
    assert v_alt == pytest.approx(10.0, rel=0.02)
    assert k > 100


@pytest.mark.parametrize("hr", [55, 75, 110])
def test_r_peak_detection(hr):
    _, x, r_true = generate_ecg(SynthConfig(heart_rate=hr, noise_uv=30, seed=1))
    r = detect_r_peaks(x, 500)
    assert abs(len(r) - len(r_true)) <= 1
    assert heart_rate_bpm(r, 500) == pytest.approx(hr, rel=0.03)


def test_no_alternans_is_negative():
    _, x, _ = generate_ecg(SynthConfig(alternans_uv=0, noise_uv=15, seed=2))
    res = analyze(x, 500)
    assert not res.positive
    assert res.k_score < 3


def test_strong_alternans_is_positive_and_monotonic():
    vals = []
    for alt in (10, 20, 40):
        _, x, _ = generate_ecg(SynthConfig(alternans_uv=alt, noise_uv=15, seed=4))
        res = analyze(x, 500)
        assert res.positive
        vals.append(res.v_alt_uv)
    assert vals[0] < vals[1] < vals[2]
    # peak estimator should recover the injected amplitude within ~15 %
    _, x, _ = generate_ecg(SynthConfig(alternans_uv=40, noise_uv=5, seed=5))
    assert analyze(x, 500).v_alt_peak_uv == pytest.approx(40, rel=0.15)
