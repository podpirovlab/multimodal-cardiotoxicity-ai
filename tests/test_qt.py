"""QT measurement (cardioonco.qt) on synthetic ECGs whose T-wave geometry is known exactly.

In the generator a T wave is a Gaussian A*exp(-((t - t_T)/w)^2) centred t_T = 0.23*sqrt(RR/0.8) s
after R.  Its steepest descent is at t_T + w/sqrt(2), and the tangent there meets the baseline
at t_T + sqrt(2)*w, which is where the tangent method must put the T end.
"""
import numpy as np
import pytest

from cardioonco import qt as Q
from cardioonco.synth import SynthConfig, generate_12lead


def measure(**kw):
    cfg = dict(fs=500, n_beats=16, noise_uv=5, wander_mv=0.02, hrv_std=0.0, seed=4)
    cfg.update(kw)
    _, X, _ = generate_12lead(SynthConfig(**cfg))
    return Q.measure_qt(X.T, cfg["fs"])


def test_t_end_follows_the_tangent_geometry():
    narrow, wide = measure(t_width=0.05), measure(t_width=0.07)
    # the T end moves by sqrt(2) * (0.07 - 0.05) s = 28.3 ms; the QRS onset does not move
    assert wide.qt_tangent_ms - narrow.qt_tangent_ms == pytest.approx(1000 * np.sqrt(2) * 0.02, abs=4)


def test_qt_shortens_with_heart_rate_as_generated():
    slow, fast = measure(heart_rate=60), measure(heart_rate=100)
    expected = 1000 * 0.23 * (np.sqrt(1.0 / 0.8) - np.sqrt(0.6 / 0.8))       # 58 ms
    assert slow.qt_tangent_ms - fast.qt_tangent_ms == pytest.approx(expected, abs=6)
    assert slow.heart_rate_bpm == pytest.approx(60, rel=0.02) and fast.heart_rate_bpm == pytest.approx(100, rel=0.02)


def test_rate_corrections_and_the_12sl_offset():
    r = measure()
    rr = r.rr_ms / 1000
    assert r.qt_ms == pytest.approx(r.qt_tangent_ms + Q.QT_OFFSET_MS)
    assert r.qtcf_ms == pytest.approx(r.qt_ms / rr ** (1 / 3))
    assert r.qtcb_ms == pytest.approx(r.qt_ms / rr ** 0.5)


def test_clean_recording_is_reliable_and_flat_t_is_not_measured():
    assert measure().reliable
    with pytest.raises(ValueError):
        measure(t_amp=0.0)


def test_one_lead_is_measured_but_never_flagged_reliable():
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=16, noise_uv=5, wander_mv=0.02, hrv_std=0.0, seed=4))
    one = Q.measure_qt(X[:, 1], 500)                     # lead II alone
    assert one.n_leads_used == 1 and not one.reliable
    assert one.qt_ms == pytest.approx(Q.measure_qt(X.T, 500).qt_ms, abs=15)


def test_long_recording_is_measured_in_ten_second_segments():
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=60, heart_rate=75, noise_uv=5, wander_mv=0.02,
                                          hrv_std=0.0, seed=4))                  # 48 s: four segments
    res = Q.measure_qt_segments(X.T, 500)
    assert res.n_segments == 4 and res.n_segments_reliable == 4
    assert res.segment_start_s in (0.0, 10.0, 20.0, 30.0)
    assert res.qtcf_ms == pytest.approx(Q.measure_qt(X[:5000].T, 500).qtcf_ms, abs=5)
    assert Q.segment_starts(15 * 500, 500) == [0]          # under 20 s: one segment, the whole recording
    assert len(Q.segment_starts(3600 * 500, 500)) == 60    # an hour: 60 segments spread over it


def test_coarse_recording_is_interpolated_before_measuring():
    """At 100 Hz (PTB-XL's low-rate copy) the leads are interpolated to 500 Hz first."""
    kw = dict(n_beats=16, noise_uv=5, wander_mv=0.02, hrv_std=0.0, seed=4)
    fine = Q.measure_qt(generate_12lead(SynthConfig(fs=500, **kw))[1].T, 500)
    coarse = Q.measure_qt(generate_12lead(SynthConfig(fs=100, **kw))[1].T, 100)
    assert coarse.reliable and coarse.qt_ms == pytest.approx(fine.qt_ms, abs=6)
