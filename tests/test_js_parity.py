"""The browser (assets/js/dsp.js) and Python (cardioonco/) implementations must agree."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cardioonco.synth import SynthConfig, generate_ecg
from cardioonco.twa import analyze

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def run_js(tmp_path: Path, x: np.ndarray, fs: float) -> dict:
    (tmp_path / "x.json").write_text(json.dumps(x.tolist()))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const x=Float64Array.from(require({json.dumps(str(tmp_path / 'x.json'))}));"
              f"const r=D.analyzeTWA(x,{fs});"
              "console.log(JSON.stringify({v:r.vAlt,k:r.k,n:r.nBeats,hr:r.hr,est:r.estimate,"
              "outcome:r.outcome,reason:r.reason,windows:r.nWindows}));")
    return json.loads(subprocess.check_output([NODE, "-e", script], text=True))


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
@pytest.mark.parametrize("alt,noise,hr,beats", [(5, 10, 75, 128), (20, 15, 108, 200), (20, 60, 80, 128),
                                                (0, 10, 108, 200)])
def test_js_matches_python(tmp_path, alt, noise, hr, beats):
    _, x, _ = generate_ecg(SynthConfig(alternans_uv=alt, noise_uv=noise, heart_rate=hr, n_beats=beats,
                                       seed=3, mains_uv=20))
    py = analyze(x, 500)
    js = run_js(tmp_path, x, 500)
    assert abs(js["n"] - py.n_beats) <= 1
    assert js["windows"] == len(py.windows)
    assert js["hr"] == pytest.approx(py.heart_rate_bpm, rel=0.01)
    assert js["v"] == pytest.approx(py.v_alt_uv, rel=0.10, abs=0.2)
    assert js["est"] == pytest.approx(py.estimate_uv, rel=0.10, abs=0.2)
    assert (js["outcome"], js["reason"]) == (py.outcome, py.reason)


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
@pytest.mark.parametrize("alt", [0, 10])
def test_js_matches_python_at_chest_strap_rate(tmp_path, alt):
    """At 130 Hz both implementations interpolate before the analysis and agree."""
    _, x, _ = generate_ecg(SynthConfig(fs=130, alternans_uv=alt, noise_uv=5, heart_rate=108, n_beats=160, seed=7))
    py = analyze(x, 130)
    js = run_js(tmp_path, x, 130)
    assert abs(js["n"] - py.n_beats) <= 1
    assert js["v"] == pytest.approx(py.v_alt_uv, rel=0.10, abs=0.2)
    assert (js["outcome"], js["reason"]) == (py.outcome, py.reason)


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
def test_js_finds_small_beats_next_to_large_ones(tmp_path):
    """Both implementations find every beat of a bigeminy whose large beats are 3x the small ones."""
    from cardioonco.preprocess import detect_r_peaks
    from test_twa import bigeminy_like
    x, small = bigeminy_like(2, 3)
    (tmp_path / "x.json").write_text(json.dumps(x.tolist()))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const x=Float64Array.from(require({json.dumps(str(tmp_path / 'x.json'))}));"
              "console.log(JSON.stringify(D.detectRPeaks(x,500).r));")
    js = np.array(json.loads(subprocess.check_output([NODE, "-e", script], text=True)))
    py = detect_r_peaks(x, 500)
    assert len(js) == len(py) == 150
    assert np.all(np.abs(np.sort(js) - np.sort(py)) <= 5)      # within 10 ms of each other


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
@pytest.mark.parametrize("up", [2, 4, 5])
def test_js_interpolation_matches_scipy(tmp_path, up):
    """Coarse recordings are interpolated with the same polyphase filter in both implementations."""
    from scipy.signal import resample_poly
    x = np.random.default_rng(up).standard_normal(500)
    (tmp_path / "x.json").write_text(json.dumps(x.tolist()))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const x=Float64Array.from(require({json.dumps(str(tmp_path / 'x.json'))}));"
              f"console.log(JSON.stringify(Array.from(D.resamplePoly(x,{up}))));")
    js = np.array(json.loads(subprocess.check_output([NODE, "-e", script], text=True)))
    np.testing.assert_allclose(js, resample_poly(x, up, 1), atol=1e-12)


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
@pytest.mark.parametrize("hr,width", [(60, 0.05), (95, 0.06)])
def test_js_qt_matches_python(tmp_path, hr, width):
    """QT, QTcF and the reliability flag agree between the browser and Python (same R peaks)."""
    from cardioonco.preprocess import consensus_r_peaks
    from cardioonco.qt import measure_qt
    from cardioonco.synth import generate_12lead
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=16, heart_rate=hr, t_width=width, noise_uv=5,
                                          wander_mv=0.02, hrv_std=0.0, seed=4))
    leads = X.T
    r = consensus_r_peaks(leads, 500)
    py = measure_qt(leads, 500, r_peaks=r)
    (tmp_path / "x.json").write_text(json.dumps({"leads": leads.tolist(), "r": r.tolist()}))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const d=require({json.dumps(str(tmp_path / 'x.json'))});"
              "const q=D.measureQT(d.leads.map(l=>Float64Array.from(l)),500,d.r);"
              "console.log(JSON.stringify(q));")
    js = json.loads(subprocess.check_output([NODE, "-e", script], text=True))
    assert js["qtMs"] == pytest.approx(py.qt_ms, abs=6)
    assert js["qtcfMs"] == pytest.approx(py.qtcf_ms, abs=7)
    assert js["reliable"] == py.reliable


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
def test_js_qt_segments_and_consensus_match_python(tmp_path):
    """On a 48-s 12-lead recording both pick the same shared R peaks and agree on the QT."""
    from cardioonco.preprocess import consensus_r_peaks
    from cardioonco.qt import measure_qt_segments
    from cardioonco.synth import generate_12lead
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=60, heart_rate=75, noise_uv=8, wander_mv=0.05,
                                          hrv_std=0.02, seed=11))
    leads = X.T
    py_r, py = consensus_r_peaks(leads, 500), measure_qt_segments(leads, 500)
    (tmp_path / "x.json").write_text(json.dumps({"leads": leads.tolist()}))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const d=require({json.dumps(str(tmp_path / 'x.json'))});"
              "const L=d.leads.map(l=>Float64Array.from(l));"
              "console.log(JSON.stringify({r:D.consensusRPeaks(L,500),q:D.measureQTSegments(L,500)}));")
    js = json.loads(subprocess.check_output([NODE, "-e", script], text=True))
    assert len(js["r"]) == len(py_r) and np.all(np.abs(np.array(js["r"]) - py_r) <= 2)
    q = js["q"]
    # segments differ by about a millisecond here, so the median one may be a different segment
    assert (q["nSegments"], q["nSegmentsReliable"]) == (py.n_segments, py.n_segments_reliable)
    assert q["qtMs"] == pytest.approx(py.qt_ms, abs=6) and q["qtcfMs"] == pytest.approx(py.qtcf_ms, abs=7)
    assert q["reliable"] == py.reliable


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
def test_js_qt_at_100_hz_matches_python(tmp_path):
    from cardioonco.qt import measure_qt
    from cardioonco.synth import generate_12lead
    _, X, _ = generate_12lead(SynthConfig(fs=100, n_beats=16, noise_uv=5, wander_mv=0.02, hrv_std=0.0, seed=4))
    py = measure_qt(X.T, 100)
    (tmp_path / "x.json").write_text(json.dumps({"leads": X.T.tolist()}))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const d=require({json.dumps(str(tmp_path / 'x.json'))});"
              "console.log(JSON.stringify(D.measureQT(d.leads.map(l=>Float64Array.from(l)),100)));")
    js = json.loads(subprocess.check_output([NODE, "-e", script], text=True))
    assert js["qtMs"] == pytest.approx(py.qt_ms, abs=6) and js["reliable"] == py.reliable


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
@pytest.mark.parametrize("alt,hr", [(15, 100), (0, 108)])
def test_js_multilead_rule_matches_python(tmp_path, alt, hr):
    """All 12 leads and the two-adjacent-precordial rule give the same verdict in both."""
    from cardioonco.synth import generate_12lead
    from cardioonco.twa import analyze_leads
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=160, heart_rate=hr, alternans_uv=alt, noise_uv=8, seed=5))
    py = analyze_leads(X.T, 500, names)
    (tmp_path / "x.json").write_text(json.dumps({"leads": X.T.tolist(), "names": names}))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const d=require({json.dumps(str(tmp_path / 'x.json'))});"
              "console.log(JSON.stringify(D.analyzeLeads(d.leads.map(l=>Float64Array.from(l)),d.names,500)));")
    js = json.loads(subprocess.check_output([NODE, "-e", script], text=True))
    assert (js["outcome"], js["reason"]) == (py.outcome, py.reason)
    same = sum(js["leads"][n]["outcome"] == py.leads[n]["outcome"] for n in names)
    assert same >= 11                                  # a borderline lead may differ (filters differ, §7)
    assert js["pairEstimate"] == pytest.approx(py.pair_estimate_uv, rel=0.15, abs=0.3)
