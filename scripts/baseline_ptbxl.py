#!/usr/bin/env python3
"""A simple baseline for the PTB-XL task: does the deep network earn its complexity?

Hand-made features of every 12-lead ECG (100 Hz, 10 s) plus age and sex, then two classical
models, one per diagnostic class:
  * logistic regression on standardised features (L2 penalty);
  * gradient-boosted trees (scikit-learn HistGradientBoostingClassifier).

Features per lead (12 x 17): mean, standard deviation, minimum, maximum, the 5th, 25th, 50th,
75th and 95th percentiles, skewness, kurtosis, mean absolute first difference, rate of
zero crossings of the mean-removed signal, and the share of power in 0.5-4, 4-8, 8-15 and
15-40 Hz.  Per ECG: heart rate and R-R variability from the R-peak detector on lead II, and
age, sex and the age-missing flag.  209 features in all.

Protocol, as for the network: folds 1-8 fit the models, fold 9 chooses the hyper-parameters
(each class separately, by validation AUC), fold 10 is scored once.  The test macro-AUC is
compared with the released network on the same ECGs by a paired bootstrap.

Output: docs/results/ptbxl_baseline.json.

    python scripts/baseline_ptbxl.py [--data data/ptb-xl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_ptbxl as T  # noqa: E402
from cardioonco.preprocess import detect_r_peaks  # noqa: E402

FS = 100.0
BANDS = ((0.5, 4), (4, 8), (8, 15), (15, 40))


def lead_features(x: np.ndarray) -> list[float]:
    f, p = welch(x, FS, nperseg=256)
    total = p[(f >= 0.5) & (f <= 40)].sum() + 1e-12
    xc = x - x.mean()
    return ([x.mean(), x.std(), x.min(), x.max()] + list(np.percentile(x, [5, 25, 50, 75, 95]))
            + [float(stats.skew(x)), float(stats.kurtosis(x)), float(np.abs(np.diff(x)).mean()),
               float(np.mean(np.diff(np.sign(xc)) != 0))]
            + [p[(f >= lo) & (f < hi)].sum() / total for lo, hi in BANDS])


def ecg_features(ecg: np.ndarray) -> list[float]:
    feats = [v for lead in ecg for v in lead_features(lead.astype(float))]
    r = detect_r_peaks(ecg[1].astype(float), FS)               # lead II
    rr = np.diff(r) / FS
    hr = 60.0 / np.median(rr) if len(rr) >= 2 else np.nan
    return feats + [hr, float(np.std(rr)) if len(rr) >= 3 else np.nan]


def build_features(X: np.ndarray, M: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        F = np.load(cache)
        if F.shape[0] == len(X):
            return F
    t0, rows = time.time(), []
    for i, ecg in enumerate(X):
        rows.append(ecg_features(ecg))
        if i % 4000 == 0:
            print(f"  features {i:>6}/{len(X)} ({time.time() - t0:.0f} s)", flush=True)
    F = np.column_stack([np.asarray(rows, dtype=np.float32), M]).astype(np.float32)
    F[~np.isfinite(F)] = np.nan
    np.save(cache, F)
    return F


def fit_best(make, grid, Xtr, ytr, Xva, yva):
    best = None
    for params in grid:
        model = make(**params).fit(Xtr, ytr)
        auc = roc_auc_score(yva, model.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, params, model)
    return best


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def paired_ci(y, pa, pb, n=2000, seed=0):
    rng, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if (y[i].min(axis=0) == 1).any() or (y[i].max(axis=0) == 0).any():
            continue
        d.append(macro(y[i], pa[i]) - macro(y[i], pb[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


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
    F = build_features(X, M, root / "features_baseline_v1.npy")
    print(f"{F.shape[1]} features for {F.shape[0]} ECGs")
    imputer_median = np.nanmedian(F[tr], axis=0)                  # training-fold medians only
    F = np.where(np.isnan(F), imputer_median, F)

    models = {
        "logistic": (lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)),
                     [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]),
        "boosting": (lambda learning_rate, max_depth: HistGradientBoostingClassifier(
                         learning_rate=learning_rate, max_depth=max_depth, max_iter=300, random_state=0),
                     [{"learning_rate": lr, "max_depth": d} for lr in (0.05, 0.1) for d in (3, 6)]),
    }
    res = {"n_features": int(F.shape[1]), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())}
    probs = {}
    for name, (make, grid) in models.items():
        p_te, chosen = np.zeros((te.sum(), len(T.CLASSES))), {}
        for j, c in enumerate(T.CLASSES):
            val_auc, params, model = fit_best(make, grid, F[tr], Y[tr, j], F[va], Y[va, j])
            p_te[:, j] = model.predict_proba(F[te])[:, 1]
            chosen[c] = {"params": params, "val_auc": float(val_auc), "test_auc": float(roc_auc_score(Y[te, j], p_te[:, j]))}
        probs[name] = p_te
        res[name] = {"test_macro_auc": macro(Y[te], p_te), "per_class": chosen}
        print(f"{name:>9}: test macro-AUC {res[name]['test_macro_auc']:.4f}  "
              + "  ".join(f"{c} {chosen[c]['test_auc']:.3f}" for c in T.CLASSES), flush=True)
    net = np.load(ROOT / "docs" / "results" / "ptbxl_test_probabilities.npz")
    assert (net["y"] == Y[te]).all(), "test fold order differs from the stored network predictions"
    res["network_macro_auc"] = macro(Y[te], net["p_float"])
    for name in models:
        res[name]["network_minus_baseline"] = res["network_macro_auc"] - res[name]["test_macro_auc"]
        res[name]["network_minus_baseline_ci95"] = paired_ci(Y[te], net["p_float"], probs[name])
        print(f"network - {name}: {res[name]['network_minus_baseline']:+.4f} {res[name]['network_minus_baseline_ci95']}")
    out = ROOT / "docs" / "results" / "ptbxl_baseline.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"-> {out.relative_to(ROOT)}")
    return res


if __name__ == "__main__":
    main()
