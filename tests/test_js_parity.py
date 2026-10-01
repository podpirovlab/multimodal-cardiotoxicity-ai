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
