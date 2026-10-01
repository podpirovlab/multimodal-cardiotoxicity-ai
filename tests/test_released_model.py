"""The trained PTB-XL model in models/ptbxl-1.0 is the one its model card describes.

These tests need no PTB-XL download: they check the checksums printed in the card, that
the checkpoint loads into the current network code, that the ONNX copy computes the same
logits, and that predict.py runs it.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ptbxl-1.0"
SHA256 = {
    "model.pt": "5d25e587a3bec4cc553a299043428a428220c8592dfb69962d2c589b452e47f5",
    "model.onnx": "8aa31b0d08da3082e04609390255a2421981186d1087a1cb57b8d5fd72b2ac3a",
}


@pytest.mark.parametrize("name", sorted(SHA256))
def test_checksums_match_the_model_card(name):
    assert hashlib.sha256((MODEL / name).read_bytes()).hexdigest() == SHA256[name]
    assert SHA256[name] in (MODEL / "README.md").read_text()


def test_metrics_match_the_model_card():
    m = json.loads((MODEL / "metrics.json").read_text())
    card = (MODEL / "README.md").read_text()
    assert f"{m['test_macro_auc']:.3f}" in card and m["best_epoch"] == 16
    for c, v in m["per_class"].items():
        lo, hi = v["auc_ci95"]
        assert f"{c} | {v['auc']:.3f} [{lo:.3f}, {hi:.3f}]" in card


def _load():
    torch = pytest.importorskip("torch")
    from cardioonco.model import CardioOncoNet
    ck = torch.load(MODEL / "model.pt", map_location="cpu", weights_only=False)
    net = CardioOncoNet(n_classes=len(ck["classes"]), width=ck["width"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return torch, ck, net


def test_checkpoint_loads_into_current_code():
    _, ck, net = _load()
    assert list(ck["classes"]) == ["NORM", "MI", "STTC", "CD", "HYP"]
    assert sum(p.numel() for p in net.parameters()) == 573482


def test_onnx_copy_matches_pytorch():
    ort = pytest.importorskip("onnxruntime")
    torch, _, net = _load()
    rng = np.random.default_rng(2)
    ecg = rng.standard_normal((2, 12, 1000)).astype(np.float32)
    meta = rng.standard_normal((2, 3)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(ecg), torch.from_numpy(meta)).numpy()
    out = ort.InferenceSession(str(MODEL / "model.onnx")).run(["logits"], {"ecg": ecg, "meta": meta})[0]
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_predict_uses_the_released_model(tmp_path):
    pytest.importorskip("torch")
    import predict
    from cardioonco.synth import SynthConfig, generate_ecg
    _, x, _ = generate_ecg(SynthConfig(fs=100, n_beats=14, seed=3))
    csv = tmp_path / "ecg.csv"
    np.savetxt(csv, np.tile(x[:1000, None], (1, 12)), delimiter=",")
    report = predict.main(["--csv", str(csv), "--fs", "100", "--checkpoint", str(MODEL / "model.pt")])
    assert set(report["model"]) == {"NORM", "MI", "STTC", "CD", "HYP"}
    assert all(0.0 <= p <= 1.0 for p in report["model"].values())


def test_predict_refuses_a_missing_checkpoint():
    import predict
    with pytest.raises(SystemExit, match="checkpoint not found"):
        predict.main(["--demo", "--checkpoint", "no/such/model.pt"])
