"""scripts/batch_analyse.py: a folder of mixed files -> one CSV with the web tool's columns."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pyedflib

from cardioonco.synth import SynthConfig, generate_12lead

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_script():
    spec = importlib.util.spec_from_file_location("batch_analyse", ROOT / "scripts" / "batch_analyse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_folder_becomes_one_table(tmp_path):
    _, X, _ = generate_12lead(SynthConfig(fs=500, n_beats=160, heart_rate=100, alternans_uv=15, noise_uv=8, seed=5))
    hdr = [pyedflib.highlevel.make_signal_header(n, dimension="uV", sample_frequency=500, physical_min=-5000,
                                                 physical_max=5000) for n in NAMES]
    pyedflib.highlevel.write_edf(str(tmp_path / "twelve.edf"), np.ascontiguousarray(X.T) * 1000, hdr)
    (tmp_path / "broken.edf").write_text("not an ecg")
    mod = load_script()
    out = tmp_path / "out.csv"
    mod.main([str(tmp_path), "--out", str(out)])
    rows = {r["file"]: r for r in csv.DictReader(open(out))}
    assert list(next(csv.reader(open(out)))) == mod.COLS
    t = rows["twelve.edf"]
    assert (t["twa_outcome"], t["all_leads_outcome"], t["qt_reliable"]) == ("positive", "positive", "1")
    assert float(t["v_alt_uv"]) > 1.9 and "V4-V5" in t["positive_pairs"]          # uV channels read as mV
    b = rows["broken.edf"]
    assert (b["twa_outcome"], b["twa_reason"]) == ("not_analysed", "error") and str(tmp_path) not in b["error"]
