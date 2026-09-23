"""Train CardioOncoNet on real ECGs from PTB-XL (PhysioNet).

PTB-XL (Wagner et al., Scientific Data 2020): 21,799 clinical 12-lead, 10-second ECGs
from 18,869 patients, labelled by cardiologists.  We use the official 5 diagnostic
super-classes and the official patient-wise split (folds 1-8 train, 9 val, 10 test):

    NORM  normal ECG
    MI    myocardial infarction
    STTC  ST/T change            <- repolarisation abnormalities, the class closest to
                                    the anthracycline-cardiotoxicity ECG phenotype
    CD    conduction disturbance
    HYP   hypertrophy

IMPORTANT: PTB-XL has no chemotherapy labels.  This is a *proxy* task that proves the
pipeline works on real clinical data; it is NOT a cardiotoxicity detector.  See ROADMAP.md.

Usage
-----
    bash scripts/download_ptbxl.sh            # ~1.7 GB, once
    python train_ptbxl.py --data data/ptb-xl --epochs 30
    # Apple Silicon: uses the GPU through Metal (mps) automatically.

Outputs (in --out, default runs/ptbxl):
    model.pt        best checkpoint (by validation macro-AUC) + normalisation stats
    metrics.json    test-set AUC per class with bootstrap 95 % CI, sens/spec, etc.
    roc_test.png    ROC curves on the untouched test fold
    model.onnx      (with --export-onnx) portable model for the browser / edge devices
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from cardioonco.model import CardioOncoNet, count_parameters

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ----------------------------------------------------------------------------- data
def load_labels(root: Path) -> pd.DataFrame:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]

    def superclasses(codes: dict) -> list:
        return sorted({scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index})

    db["superclass"] = db["scp_codes"].apply(superclasses)
    db = db[db["superclass"].map(len) > 0].copy()
    for c in CLASSES:
        db[c] = db["superclass"].apply(lambda s, c=c: int(c in s))
    return db


def load_signals(root: Path, db: pd.DataFrame, cache: Path) -> np.ndarray:
    """(N, 12, 1000) float32 in mV at 100 Hz, cached as .npy after the first run."""
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape[0] == len(db):
            return np.asarray(arr)
    import wfdb
    out = np.zeros((len(db), 12, 1000), dtype=np.float32)
    t0 = time.time()
    for i, fn in enumerate(db["filename_lr"]):
        sig, _ = wfdb.rdsamp(str(root / fn))
        out[i] = np.nan_to_num(sig[:1000].T.astype(np.float32))
        if i % 2000 == 0:
            print(f"  read {i:>6}/{len(db)} records  ({time.time() - t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def metadata(db: pd.DataFrame) -> np.ndarray:
    age = db["age"].astype(float).copy()
    age[age > 120] = 90.0                  # PTB-XL encodes age > 89 as 300 (de-identification)
    missing = age.isna().astype(float)
    age = age.fillna(62.0)
    age_z = (age - 62.0) / 17.0            # approx. PTB-XL mean / std
    sex = db["sex"].astype(float).fillna(0.5)   # 0 = male, 1 = female
    return np.stack([age_z, sex, missing], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- training
def augment(x: torch.Tensor) -> torch.Tensor:
    """Physiologically plausible noise: amplitude scaling, baseline wander, white noise."""
    b, _, t = x.shape
    scale = 1.0 + 0.1 * torch.randn(b, 1, 1, device=x.device)
    tt = torch.linspace(0, 10, t, device=x.device)
    phase = 2 * np.pi * torch.rand(b, 1, 1, device=x.device)
    wander = 0.1 * torch.rand(b, 1, 1, device=x.device) * torch.sin(2 * np.pi * 0.3 * tt + phase)
    return x * scale + wander + 0.02 * torch.randn_like(x)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for xb, mb, yb in loader:
        ps.append(torch.sigmoid(model(xb.to(device), mb.to(device))).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def macro_auc(y, p):
    """Mean one-vs-rest ROC-AUC over classes that have both positives and negatives."""
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min(axis=0).max() == 1 or (y[idx].max(axis=0) == 0).any():
            continue
        stats.append([roc_auc_score(y[idx, j], p[idx, j]) for j in range(y.shape[1])])
    s = np.asarray(stats)
    return np.percentile(s, 2.5, axis=0), np.percentile(s, 97.5, axis=0)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ptb-xl", help="folder with ptbxl_database.csv")
    ap.add_argument("--out", default="runs/ptbxl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--no-meta", action="store_true", help="ablation: zero out age/sex")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"device: {device}")

    db = load_labels(root)
    X = load_signals(root, db, root / "cache_100hz.npy")
    M = metadata(db)
    if args.no_meta:
        M[:] = 0.0
    Y = db[CLASSES].values.astype(np.float32)
    fold = db["strat_fold"].values
    tr, va, te = fold <= 8, fold == 9, fold == 10
    print(f"records: train {tr.sum()}, val {va.sum()}, test {te.sum()}")

    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    def loader(mask, shuffle):
        ds = TensorDataset(torch.from_numpy(Xn[mask]), torch.from_numpy(M[mask]), torch.from_numpy(Y[mask]))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, drop_last=bool(shuffle and mask.sum() > args.batch))

    dl_tr, dl_va, dl_te = loader(tr, True), loader(va, False), loader(te, False)

    model = CardioOncoNet(n_classes=len(CLASSES), width=args.width).to(device)
    print(f"parameters: {count_parameters(model):,}")
    pos = Y[tr].mean(axis=0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device).clamp(max=10) ** 0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=max(1, len(dl_tr)))

    best, best_ep, history = -1.0, -1, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for xb, mb, yb in dl_tr:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            loss = loss_fn(model(augment(xb), mb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(xb)
        p_va, y_va = predict(model, dl_va, device)
        auc_va = macro_auc(y_va, p_va)
        history.append({"epoch": ep, "train_loss": run / tr.sum(), "val_macro_auc": auc_va})
        print(f"epoch {ep:>3}  loss {run / tr.sum():.4f}  val macro-AUC {auc_va:.4f}  ({time.time() - t0:.0f}s)")
        if best_ep < 0 or auc_va > best:
            best, best_ep = auc_va, ep
            torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "classes": CLASSES,
                        "width": args.width, "fs": 100, "epoch": ep}, out / "model.pt")
        elif ep - best_ep >= args.patience:
            print("early stopping")
            break

    ckpt = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    p_va, y_va = predict(model, dl_va, device)
    p_te, y_te = predict(model, dl_te, device)
    try:
        lo, hi = bootstrap_ci(y_te, p_te)
    except (ValueError, IndexError):
        lo = hi = np.full(len(CLASSES), np.nan)

    per_class = {}
    for j, c in enumerate(CLASSES):
        if not (0 < y_va[:, j].sum() < len(y_va)) or not (0 < y_te[:, j].sum() < len(y_te)):
            continue                                         # class absent in a split (tiny data only)
        fpr, tpr, thr = roc_curve(y_va[:, j], p_va[:, j])
        t_star = float(min(thr[np.argmax(tpr - fpr)], 1.0))  # Youden J on VALIDATION only
        yhat = p_te[:, j] >= t_star
        tp = int((yhat & (y_te[:, j] == 1)).sum()); fn = int((~yhat & (y_te[:, j] == 1)).sum())
        tn = int((~yhat & (y_te[:, j] == 0)).sum()); fp = int((yhat & (y_te[:, j] == 0)).sum())
        per_class[c] = {
            "auc": float(roc_auc_score(y_te[:, j], p_te[:, j])),
            "auc_ci95": [float(lo[j]), float(hi[j])],
            "threshold": t_star,
            "sensitivity": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp),
            "prevalence": float(y_te[:, j].mean()),
        }
    metrics = {"dataset": "PTB-XL v1.0.3, 100 Hz, folds 1-8/9/10", "best_epoch": best_ep,
               "val_macro_auc": best, "test_macro_auc": macro_auc(y_te, p_te),
               "n_test": int(te.sum()), "parameters": count_parameters(model),
               "metadata_used": not args.no_meta, "per_class": per_class, "history": history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for j, c in enumerate(CLASSES):
            if c not in per_class:
                continue
            fpr, tpr, _ = roc_curve(y_te[:, j], p_te[:, j])
            ax.plot(fpr, tpr, lw=2, label=f"{c}  AUC={per_class[c]['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="PTB-XL test fold (fold 10)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "roc_test.png", dpi=150)
    except Exception as exc:  # plotting is optional
        print(f"(ROC plot skipped: {exc})")

    if args.export_onnx:
        import inspect
        model_cpu = model.to("cpu").eval()
        extra = {"external_data": False} if "external_data" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(model_cpu, (torch.zeros(1, 12, 1000), torch.zeros(1, 3)), str(out / "model.onnx"),
                          input_names=["ecg", "meta"], output_names=["logits"], opset_version=18,
                          dynamic_axes={"ecg": {0: "batch"}, "meta": {0: "batch"}, "logits": {0: "batch"}}, **extra)
        np.savez(out / "norm_stats.npz", mu=mu, sd=sd)
        print(f"ONNX model -> {out / 'model.onnx'}")
    return metrics


if __name__ == "__main__":
    main()
