"""The browser (assets/js/dsp.js) and Python (cardioonco/) implementations must agree."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cardioonco.synth import SynthConfig, generate_ecg
from cardioonco.twa import analyze

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node.js not installed")
@pytest.mark.parametrize("alt,noise", [(5, 10), (20, 15), (20, 60)])
def test_js_matches_python(tmp_path, alt, noise):
    _, x, _ = generate_ecg(SynthConfig(alternans_uv=alt, noise_uv=noise, seed=3, mains_uv=20))
    py = analyze(x, 500)
    (tmp_path / "x.json").write_text(json.dumps(x.tolist()))
    script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
              f"const x=Float64Array.from(require({json.dumps(str(tmp_path / 'x.json'))}));"
              "const r=D.analyzeTWA(x,500);console.log(JSON.stringify({v:r.vAlt,k:r.k,n:r.r.length,hr:r.hr}));")
    js = json.loads(subprocess.check_output([NODE, "-e", script], text=True))
    assert js["n"] == 128
    assert js["hr"] == pytest.approx(py.heart_rate_bpm, rel=0.01)
    assert js["v"] == pytest.approx(py.v_alt_uv, rel=0.10, abs=0.2)
    assert (js["k"] >= 3) == (py.k_score >= 3)
