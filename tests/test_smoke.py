"""Every entry point runs end to end: the command-line tool, the Gradio lab and the
teaching scripts in legacy/.  These used to be separate CI steps; as tests they run the
same way locally and in CI."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd):
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)


def test_predict_demo_writes_a_research_fhir_report(tmp_path):
    out = tmp_path / "report.json"
    run([ROOT / "predict.py", "--demo", "--alternans", "20", "--fhir-out", out], cwd=ROOT)
    rep = json.loads(out.read_text())
    assert rep["resourceType"] == "DiagnosticReport" and rep["status"] == "preliminary"
    outcome = next(c for c in rep["contained"] if c.get("id") == "outcome")
    assert outcome["valueCodeableConcept"]["coding"][0]["code"] == "positive"


@pytest.mark.parametrize("script", ["federated_fhir_core.py", "main_model.py"])
def test_legacy_script_runs_from_any_directory(tmp_path, script):
    run([ROOT / "legacy" / script], cwd=tmp_path)


def test_legacy_pytorch_teaching_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("advanced_model", ROOT / "legacy" / "advanced_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sig, fs = m.load_real_hospital_edf(None)
    assert sig.shape == (12, 1250) and fs == 250
    assert m.EnhancedCardioOncoNet()(torch.randn(4, 2), torch.randn(4, 2)).shape == (4, 1)


def test_gradio_lab_reports_all_three_outcomes():
    pytest.importorskip("gradio")
    sys.path.insert(0, str(ROOT))
    import app
    assert app.run_lab(20, 15, 75, 0.35)[0].startswith("TWA criterion met")
    assert app.run_lab(0, 15, 108, 0.35)[0].startswith("No significant TWA")
    assert app.run_lab(0, 15, 75, 0.35)[0].startswith("Indeterminate (hr too low)")
