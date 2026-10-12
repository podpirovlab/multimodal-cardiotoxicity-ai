"""De-identifying EDF/EDF+ and BDF files: header fields and annotation text go, the signal stays,
and the browser (assets/js/edf.js) writes byte-for-byte the same file as cardioonco/edf.py."""
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from cardioonco.edf import anonymize_edf

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def write(path: Path, file_type, annotations=True) -> np.ndarray:
    fs, t = 250, np.arange(2500) / 250
    sig = np.vstack([np.sin(2 * np.pi * 1.1 * t), 0.5 * np.cos(2 * np.pi * 0.6 * t)])
    hdr = [pyedflib.highlevel.make_signal_header(n, dimension="mV", sample_frequency=fs, physical_min=-2,
                                                 physical_max=2) for n in ("II", "V5")]
    header = pyedflib.highlevel.make_header(patientname="Ivanov_Ivan", patientcode="MCH-0234567", sex="Male",
                                            birthdate="02 May 1961", technician="Petrova",
                                            equipment="ECG-12", startdate=datetime.datetime(2026, 3, 2, 14, 30, 5))
    if annotations:
        header["annotations"] = [[1.5, -1, "Ivanov fainted"], [4.0, 2.0, "dose 60 mg/m2"]]
    pyedflib.highlevel.write_edf(str(path), sig, hdr, header, file_type=file_type)
    return sig


@pytest.mark.parametrize("file_type,ext", [(pyedflib.FILETYPE_EDFPLUS, "edf"), (pyedflib.FILETYPE_BDFPLUS, "bdf"),
                                           (pyedflib.FILETYPE_EDF, "edf")])
def test_identifying_text_is_removed_and_the_signal_is_untouched(tmp_path, file_type, ext):
    src, dst = tmp_path / f"in.{ext}", tmp_path / f"out.{ext}"
    plus = file_type != pyedflib.FILETYPE_EDF
    sig = write(src, file_type, annotations=plus)
    out, removed = anonymize_edf(src.read_bytes())
    dst.write_bytes(out)
    assert "Ivanov_Ivan" in removed["patient"] and removed["startdate"] == "02.03.26"
    assert removed["annotations"] == (2 if plus else 0)
    raw = out[:256].decode("latin-1")
    assert "Ivanov" not in out.decode("latin-1") and "MCH-0234567" not in raw and "1961" not in raw
    with pyedflib.EdfReader(str(dst)) as f:                 # still a valid file for a standard reader
        assert f.getStartdatetime() == datetime.datetime(1985, 1, 1, 0, 0, 0)
        assert len(f.readAnnotations()[0]) == 0
        for i in range(2):
            np.testing.assert_allclose(f.readSignal(i), sig[i], atol=1e-3)
    with pyedflib.EdfReader(str(src)) as a, pyedflib.EdfReader(str(dst)) as b:
        for i in range(2):                                  # bit-for-bit the same samples
            np.testing.assert_array_equal(a.readSignal(i, digital=True), b.readSignal(i, digital=True))


@pytest.mark.skipif(NODE is None and not os.environ.get("CI"), reason="node.js not installed")
@pytest.mark.parametrize("file_type,ext", [(pyedflib.FILETYPE_EDFPLUS, "edf"), (pyedflib.FILETYPE_BDFPLUS, "bdf")])
def test_browser_writes_the_same_bytes(tmp_path, file_type, ext):
    src = tmp_path / f"in.{ext}"
    write(src, file_type)
    py, removed = anonymize_edf(src.read_bytes())
    script = (f"const E=require({json.dumps(str(ROOT / 'assets/js/edf.js'))});const fs=require('fs');"
              f"const b=fs.readFileSync({json.dumps(str(src))});"
              "const r=E.anonymizeEDF(b.buffer.slice(b.byteOffset,b.byteOffset+b.length));"
              f"fs.writeFileSync({json.dumps(str(tmp_path / 'js.bin'))},Buffer.from(r.buffer));"
              "console.log(JSON.stringify(r.removed));")
    js_removed = json.loads(subprocess.check_output([NODE, "-e", script], text=True))
    assert (tmp_path / "js.bin").read_bytes() == py
    assert js_removed == removed
