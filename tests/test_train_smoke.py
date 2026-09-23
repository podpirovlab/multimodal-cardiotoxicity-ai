"""End-to-end smoke test of train_ptbxl.py on a tiny fake dataset in the PTB-XL layout.

The fake records reuse the synthetic ECG generator; "STTC" records get a flattened
T wave so the network has something learnable.  This only checks that the pipeline
runs and produces metrics -- real numbers come from the real PTB-XL download.
"""
import json

import numpy as np
import pandas as pd
import pytest

wfdb = pytest.importorskip("wfdb")
torch = pytest.importorskip("torch")

from cardioonco.synth import SynthConfig, generate_ecg  # noqa: E402


def make_fake_ptbxl(root, n=120):
    (root / "records100" / "00000").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    codes = ["NORM", "IMI", "NDT", "LAFB", "LVH"]
    for i in range(1, n + 1):
        k = i % 5
        _, x, _ = generate_ecg(SynthConfig(fs=100, n_beats=14, heart_rate=70 + 10 * rng.random(),
                                           t_amp=0.08 if codes[k] == "NDT" else 0.35, seed=i))
        sig = np.tile(x[:1000, None], (1, 12)) * (0.5 + rng.random(12))
        name = f"{i:05d}_lr"
        wfdb.wrsamp(name, fs=100, units=["mV"] * 12, sig_name=[f"L{j}" for j in range(12)],
                    p_signal=sig, fmt=["16"] * 12, write_dir=str(root / "records100" / "00000"))
        rows.append({"ecg_id": i, "patient_id": i, "age": 30 + i % 60 if i % 17 else 300, "sex": i % 2,
                     "scp_codes": str({codes[k]: 100.0}), "strat_fold": 1 + (i // 5) % 10,
                     "filename_lr": f"records100/00000/{name}"})
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame({"diagnostic": [1] * 5, "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP"]},
                 index=pd.Index(codes, name="")).to_csv(root / "scp_statements.csv")


def test_training_pipeline_runs(tmp_path):
    import train_ptbxl
    make_fake_ptbxl(tmp_path / "ptb")
    m = train_ptbxl.main(["--data", str(tmp_path / "ptb"), "--out", str(tmp_path / "run"),
                          "--epochs", "2", "--batch", "16", "--width", "8", "--device", "cpu"])
    assert set(m["per_class"]) <= {"NORM", "MI", "STTC", "CD", "HYP"} and m["per_class"]
    assert (tmp_path / "run" / "model.pt").exists()
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert 0.0 <= saved["test_macro_auc"] <= 1.0
