#!/usr/bin/env python3
"""Paired comparison of two trained PTB-XL models on the same test ECGs (fold 10).

Typical use: is the bilinear fusion with age and sex better than the same network with
the metadata switched off (train_ptbxl.py --no-meta)?  Both models score every test
record; the difference in AUC is bootstrapped over records (paired), so the interval
reflects how much the comparison could move with a different test sample.

    python scripts/compare_ptbxl_runs.py runs/ptbxl runs/ptbxl_nometa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_ptbxl as T  # noqa: E402
from cardioonco.model import CardioOncoNet  # noqa: E402


def test_scores(run: Path, X, M, no_meta: bool) -> np.ndarray:
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    net = CardioOncoNet(n_classes=len(ck["classes"]), width=ck["width"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    Xn = ((X - ck["mu"]) / ck["sd"]).astype(np.float32)
    Mm = np.zeros_like(M) if no_meta else M
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), 256):
            out.append(torch.sigmoid(net(torch.from_numpy(Xn[i:i + 256]), torch.from_numpy(Mm[i:i + 256]))).numpy())
    return np.concatenate(out)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="e.g. runs/ptbxl (with metadata)")
    ap.add_argument("run_b", help="e.g. runs/ptbxl_nometa")
    ap.add_argument("--data", default="data/ptb-xl")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    root = Path(args.data)
    db = T.load_labels(root)
    X = T.load_signals(root, db, root / "cache_100hz.npy")
    M = T.metadata(db)
    Y = db[T.CLASSES].values
    te = db["strat_fold"].values == 10
    meta_b = json.loads((Path(args.run_b) / "metrics.json").read_text()).get("metadata_used", True)
    pa = test_scores(Path(args.run_a), X[te], M[te], no_meta=False)
    pb = test_scores(Path(args.run_b), X[te], M[te], no_meta=not meta_b)
    y = Y[te]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(y), len(y))
        if (y[idx].min(axis=0) == 1).any() or (y[idx].max(axis=0) == 0).any():
            continue
        diffs.append([roc_auc_score(y[idx, j], pa[idx, j]) - roc_auc_score(y[idx, j], pb[idx, j]) for j in range(y.shape[1])])
    d = np.asarray(diffs)
    point = [roc_auc_score(y[:, j], pa[:, j]) - roc_auc_score(y[:, j], pb[:, j]) for j in range(y.shape[1])]
    macro = d.mean(axis=1)
    res = {"n_test": int(te.sum()), "boot": len(d),
           "macro_auc_diff": float(np.mean(point)), "macro_ci95": [float(np.percentile(macro, 2.5)), float(np.percentile(macro, 97.5))],
           "per_class": {c: {"diff": float(point[j]), "ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]}
                         for j, c in enumerate(T.CLASSES)}}
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
