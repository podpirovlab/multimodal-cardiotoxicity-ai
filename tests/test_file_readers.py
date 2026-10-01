"""The browser file readers (assets/js/edf.js) on files written by an independent library."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pyedflib = pytest.importorskip("pyedflib")
needs_node = pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")


def run_js(body: str) -> dict:
    script = f"const E=require({json.dumps(str(ROOT / 'assets/js/edf.js'))});const fs=require('fs');{body}"
    return json.loads(subprocess.check_output([NODE, "-e", script], text=True))


def write_file(path: Path, file_type: int, digital: int, units=("uV", "mV")) -> np.ndarray:
    fs, n = 500, 5000
    t = np.arange(n) / fs
    sig = np.vstack([800 * np.sin(2 * np.pi * 1.3 * t), 0.4 * np.cos(2 * np.pi * 0.7 * t)])
    headers = [pyedflib.highlevel.make_signal_header(f"ch{i}", dimension=u, sample_frequency=fs,
                                                     physical_min=-1000 if u == "uV" else -1,
                                                     physical_max=1000 if u == "uV" else 1,
                                                     digital_min=-digital, digital_max=digital)
               for i, u in enumerate(units)]
    pyedflib.highlevel.write_edf(str(path), sig, headers, file_type=file_type)
    return sig


@needs_node
@pytest.mark.parametrize("kind,file_type,digital,tol_mv", [("edf", pyedflib.FILETYPE_EDFPLUS, 32767, 2e-4),
                                                           ("bdf", pyedflib.FILETYPE_BDFPLUS, 8388607, 1e-6)])
def test_edf_and_bdf_values_in_mv(tmp_path, kind, file_type, digital, tol_mv):
    path = tmp_path / f"x.{kind}"
    sig = write_file(path, file_type, digital)
    r = run_js(f"const b=fs.readFileSync({json.dumps(str(path))});"
               "const e=E.parseEDF(b.buffer.slice(b.byteOffset,b.byteOffset+b.length));"
               "const ch=e.signals.map((s,i)=>s.annotations?null:Array.from(e.getChannel(i)));"
               "console.log(JSON.stringify({format:e.format,fs:e.signals.map(s=>s.fs),ch}));")
    assert r["format"] == kind.upper()
    assert r["fs"][:2] == [500, 500]
    np.testing.assert_allclose(r["ch"][0], sig[0] / 1000, atol=tol_mv)   # uV channel -> mV
    np.testing.assert_allclose(r["ch"][1], sig[1], atol=tol_mv)          # mV channel unchanged


@needs_node
def test_discontinuous_edf_is_refused(tmp_path):
    path = tmp_path / "x.edf"
    write_file(path, pyedflib.FILETYPE_EDFPLUS, 32767)
    raw = bytearray(path.read_bytes())
    raw[192:236] = b"EDF+D".ljust(44)
    path.write_bytes(bytes(raw))
    r = run_js(f"const b=fs.readFileSync({json.dumps(str(path))});"
               "try{E.parseEDF(b.buffer.slice(b.byteOffset,b.byteOffset+b.length));console.log('{}')}"
               "catch(e){console.log(JSON.stringify({code:e.code}))}")
    assert r == {"code": "discontinuous"}


@needs_node
def test_unknown_unit_is_reported():
    r = run_js("console.log(JSON.stringify(['uV','µV','mV','V','mmHg',''].map(E.unitToMv)))")
    assert r == [0.001, 0.001, 1, 1000, None, None]


@needs_node
@pytest.mark.parametrize("text,col,expected", [
    ("time;ecg\n0;0,125\n0,002;-0,3\n", 1, [0.125, -0.3]),        # Russian export: ";" and decimal comma
    ("0,12\t0,30\n0,13\t0,31\n", 1, [0.30, 0.31]),                 # tab-separated, decimal comma
    ("t,ecg\n0.000,0.12\n0.002,-0.30\n", 1, [0.12, -0.30]),        # comma-separated, decimal point
    ("0,12\n-0,08\n", 0, [0.12, -0.08]),                          # one column with decimal comma
    ("0.12 0.5\n1e-2 0.6\n", 0, [0.12, 0.01]),                    # spaces, exponent
])
def test_csv_column(text, col, expected):
    r = run_js(f"console.log(JSON.stringify(Array.from(E.parseCSV({json.dumps(text)},{col}))))")
    np.testing.assert_allclose(r, expected)


@needs_node
def test_lead_name_from_channel_label():
    labels = ["ECG II", "EKG V5", "Lead aVL", "v3", "AVR", "I", "ECG1", "MLII", "Resp", "ЭКГ III", ""]
    r = run_js(f"console.log(JSON.stringify({json.dumps(labels)}.map(E.leadFromLabel)))")
    assert r == ["II", "V5", "aVL", "V3", "aVR", "I", "", "", "", "III", ""]
