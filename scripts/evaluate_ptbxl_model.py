#!/usr/bin/env python3
"""Three checks of the released PTB-XL model (models/ptbxl-1.0) on the untouched test fold.

1. INT8.  The ONNX copy is quantised with ONNX Runtime (static, QDQ format, 8-bit weights per
   channel and 8-bit activations, calibrated on 512 training ECGs) and scored on the same 2,158
   test ECGs; the AUC difference to the float model is bootstrapped over records (paired).
2. Calibration.  Does "probability 0.8" mean the diagnosis is right 8 times in 10?  Reliability
   bins, expected calibration error (ECE, 10 equal-width bins) and Brier score on the test fold,
   before and after Platt scaling fitted on the validation fold only.
3. Attributions.  Integrated gradients (Sundararajan et al., 2017; 64 steps, baseline = the mean
   training ECG) for one test ECG labelled STTC only and one labelled MI only, the most confident
   correct prediction of each.  The arrays are small and saved for figure 08.

Outputs: docs/results/ptbxl_evaluation.json, ptbxl_attributions.npz, and ptbxl_test_probabilities.npz
(test labels and the float, INT8 and Platt-calibrated probabilities, for figures 09 and 11).

    python scripts/evaluate_ptbxl_model.py [--data data/ptb-xl]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_ptbxl as T  # noqa: E402
from cardioonco.model import CardioOncoNet  # noqa: E402

MODEL = ROOT / "models" / "ptbxl-1.0"
OUT = ROOT / "docs" / "results"


def load_model():
    ck = torch.load(MODEL / "model.pt", map_location="cpu", weights_only=False)
    net = CardioOncoNet(n_classes=len(ck["classes"]), width=ck["width"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def torch_probs(net, X, M, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + batch]), torch.from_numpy(M[i:i + batch]))).numpy())
    return np.concatenate(out)


def aucs(y, p):
    return [float(roc_auc_score(y[:, j], p[:, j])) for j in range(y.shape[1])]


def paired_boot(y, pa, pb, n=2000, seed=0):
    rng, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if (y[i].min(axis=0) == 1).any() or (y[i].max(axis=0) == 0).any():
            continue
        d.append(np.mean(aucs(y[i], pa[i])) - np.mean(aucs(y[i], pb[i])))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def int8_check(Xtr, Mtr, Xte, Mte, yte, p_float):
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quant_pre_process,
                                          quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self, X, M):
            self.items = iter([{"ecg": X[i:i + 1], "meta": M[i:i + 1]} for i in range(len(X))])

        def get_next(self):
            return next(self.items, None)

    with tempfile.TemporaryDirectory() as tmp:
        pre, q = Path(tmp) / "pre.onnx", Path(tmp) / "int8.onnx"
        quant_pre_process(str(MODEL / "model.onnx"), str(pre), skip_symbolic_shape=True)
        quantize_static(str(pre), str(q), Reader(Xtr, Mtr), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        sizes = {"float32_mb": (MODEL / "model.onnx").stat().st_size / 1e6, "int8_mb": q.stat().st_size / 1e6}
        timing = {}
        logits = {}
        for name, path in (("float32", MODEL / "model.onnx"), ("int8", q)):
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            out = [sess.run(["logits"], {"ecg": Xte[i:i + 1], "meta": Mte[i:i + 1]})[0] for i in range(len(Xte))]
            t0 = time.perf_counter()
            for i in range(200):
                sess.run(["logits"], {"ecg": Xte[i:i + 1], "meta": Mte[i:i + 1]})
            timing[name + "_ms_per_ecg"] = (time.perf_counter() - t0) / 200 * 1000
            logits[name] = np.concatenate(out)
    p_onnx = 1 / (1 + np.exp(-logits["float32"]))
    p_int8 = 1 / (1 + np.exp(-logits["int8"]))
    int8_check.p_int8 = p_int8
    return {
        "onnx_float_vs_torch_max_abs_prob_diff": float(np.abs(p_onnx - p_float).max()),
        "macro_auc_float": float(np.mean(aucs(yte, p_onnx))),
        "macro_auc_int8": float(np.mean(aucs(yte, p_int8))),
        "per_class_auc_int8": dict(zip(T.CLASSES, aucs(yte, p_int8))),
        "auc_difference_int8_minus_float": float(np.mean(aucs(yte, p_int8)) - np.mean(aucs(yte, p_onnx))),
        "auc_difference_ci95": paired_boot(yte, p_int8, p_onnx),
        "max_abs_prob_diff": float(np.abs(p_int8 - p_onnx).max()),
        "median_abs_prob_diff": float(np.median(np.abs(p_int8 - p_onnx))),
        "calibration_ecgs": int(len(Xtr)),
        **sizes, **timing,
    }


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": b, "n": int(m.sum()), "mean_p": float(p[m].mean()), "frac_pos": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return rows, float(ece)


def calibration_check(yva, pva, yte, pte):
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))  # noqa: E731
    res = {}
    calibration_check.p_platt = np.zeros_like(pte)
    for j, c in enumerate(T.CLASSES):
        lr = LogisticRegression(C=1e6).fit(logit(pva[:, j])[:, None], yva[:, j])
        p_cal = lr.predict_proba(logit(pte[:, j])[:, None])[:, 1]
        calibration_check.p_platt[:, j] = p_cal
        bins0, ece0 = reliability(yte[:, j], pte[:, j])
        bins1, ece1 = reliability(yte[:, j], p_cal)
        res[c] = {"ece_raw": ece0, "ece_platt": ece1,
                  "brier_raw": float(np.mean((pte[:, j] - yte[:, j]) ** 2)),
                  "brier_platt": float(np.mean((p_cal - yte[:, j]) ** 2)),
                  "auc_raw": float(roc_auc_score(yte[:, j], pte[:, j])),
                  "auc_platt": float(roc_auc_score(yte[:, j], p_cal)),
                  "platt_slope": float(lr.coef_[0, 0]), "platt_intercept": float(lr.intercept_[0]),
                  "bins_raw": bins0, "bins_platt": bins1}
    return res


def integrated_gradients(net, x, m, baseline, target, steps=64):
    alphas = torch.linspace(0, 1, steps + 1)[1:].view(-1, 1, 1)
    xb = torch.from_numpy(baseline)[None] + alphas * torch.from_numpy(x - baseline)[None]
    xb.requires_grad_(True)
    logit = net(xb, torch.from_numpy(m)[None].repeat(steps, 1))[:, target]
    grad, = torch.autograd.grad(logit.sum(), xb)
    attr = (x - baseline) * grad.mean(0).numpy()
    # completeness check: attributions add up to f(x) - f(baseline)
    with torch.no_grad():
        f_x = net(torch.from_numpy(x)[None], torch.from_numpy(m)[None])[0, target].item()
        f_b = net(torch.from_numpy(baseline)[None], torch.from_numpy(m)[None])[0, target].item()
    return attr.astype(np.float32), float(attr.sum()), f_x - f_b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "ptb-xl"))
    args = ap.parse_args(argv)
    root = Path(args.data)
    db = T.load_labels(root)
    X = T.load_signals(root, db, root / "cache_100hz.npy")
    M = T.metadata(db)
    Y = db[T.CLASSES].values
    fold = db["strat_fold"].values
    tr, va, te = fold <= 8, fold == 9, fold == 10
    net, ck = load_model()
    Xn = ((X - ck["mu"]) / ck["sd"]).astype(np.float32)
    p_va, p_te = torch_probs(net, Xn[va], M[va]), torch_probs(net, Xn[te], M[te])
    result = {"n_test": int(te.sum()), "n_val": int(va.sum()),
              "macro_auc_torch": float(np.mean(aucs(Y[te], p_te)))}
    print(f"float macro-AUC on test: {result['macro_auc_torch']:.4f}")

    rng = np.random.default_rng(0)
    cal_idx = rng.choice(np.flatnonzero(tr), 512, replace=False)
    result["int8"] = int8_check(Xn[cal_idx], M[cal_idx], Xn[te], M[te], Y[te], p_te)
    q = result["int8"]
    print(f"INT8: macro-AUC {q['macro_auc_int8']:.4f} vs float {q['macro_auc_float']:.4f}, "
          f"difference {q['auc_difference_int8_minus_float']:+.4f} {q['auc_difference_ci95']}, "
          f"{q['float32_mb']:.2f} MB -> {q['int8_mb']:.2f} MB")

    result["calibration"] = calibration_check(Y[va], p_va, Y[te], p_te)
    for c, r in result["calibration"].items():
        print(f"{c}: ECE {r['ece_raw']:.3f} -> {r['ece_platt']:.3f}, Brier {r['brier_raw']:.3f} -> {r['brier_platt']:.3f}")

    baseline = np.zeros((12, 1000), np.float32)                 # standardised input: 0 = mean training ECG
    te_idx = np.flatnonzero(te)
    picks = {}
    for target in ("STTC", "MI"):
        j = T.CLASSES.index(target)
        only = [k for k in range(len(te_idx)) if Y[te_idx[k]].sum() == 1 and Y[te_idx[k], j] == 1]
        k = max(only, key=lambda k: p_te[k, j])
        attr, s, delta = integrated_gradients(net, Xn[te_idx[k]], M[te_idx[k]], baseline, j)
        picks[target] = {"ecg_id": int(db.index[te_idx[k]]), "prob": float(p_te[k, j]),
                         "completeness_sum": s, "completeness_target": delta}
        np.save(OUT / f"_attr_{target}.npy", attr)
        np.save(OUT / f"_ecg_{target}.npy", X[te_idx[k]])
        print(f"IG {target}: ecg {picks[target]['ecg_id']}, p={picks[target]['prob']:.3f}, "
              f"sum of attributions {s:.3f} vs logit change {delta:.3f}")
    np.savez_compressed(OUT / "ptbxl_attributions.npz",
                        **{f"attr_{t}": np.load(OUT / f"_attr_{t}.npy").astype(np.float16) for t in picks},
                        **{f"ecg_{t}": np.load(OUT / f"_ecg_{t}.npy").astype(np.float16) for t in picks})
    for t in picks:
        (OUT / f"_attr_{t}.npy").unlink()
        (OUT / f"_ecg_{t}.npy").unlink()
    result["attributions"] = picks
    np.savez_compressed(OUT / "ptbxl_test_probabilities.npz", y=Y[te].astype(np.int8),
                        p_float=p_te.astype(np.float32), p_int8=int8_check.p_int8.astype(np.float32),
                        p_platt=calibration_check.p_platt.astype(np.float32))
    (OUT / "ptbxl_evaluation.json").write_text(json.dumps(result, indent=2))
    print("-> docs/results/ptbxl_evaluation.json, docs/results/ptbxl_attributions.npz")
    return result


if __name__ == "__main__":
    main()
