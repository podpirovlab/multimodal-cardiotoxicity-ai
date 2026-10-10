#!/usr/bin/env python3
"""Five training seeds per configuration on PTB-XL: how much of a result is luck?

Seed 42 is the released run (runs/ptbxl, runs/ptbxl_nometa); seeds 1-4 come from
scripts/train_seeds.sh.  Every model scores the same 2,158 test ECGs.  Reported:

* macro-AUC of every run, and mean and standard deviation per configuration;
* the age/sex effect paired by seed (with minus without metadata, same seed), with a
  t-interval over the five seeds and a pooled paired bootstrap over test ECGs;
* a deep ensemble (Lakshminarayanan et al., 2017): the mean probability of the five
  with-metadata models, against the single models.

Output: docs/results/ptbxl_seeds.json.

    python scripts/compare_seeds.py [--data data/ptb-xl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_ptbxl as T  # noqa: E402
from compare_ptbxl_runs import test_scores  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]


def run_dir(cfg: str, seed: int) -> Path:
    if seed == 42:
        return ROOT / "runs" / ("ptbxl" if cfg == "meta" else "ptbxl_nometa")
    return ROOT / "runs" / "seeds" / f"{cfg}_s{seed}"


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "ptb-xl"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    root = Path(args.data)
    db = T.load_labels(root)
    X = T.load_signals(root, db, root / "cache_100hz.npy")
    M = T.metadata(db)
    te = db["strat_fold"].values == 10
    y = db[T.CLASSES].values[te]
    probs = {(cfg, s): test_scores(run_dir(cfg, s), X[te], M[te], no_meta=(cfg == "nometa"))
             for cfg in ("meta", "nometa") for s in SEEDS}
    auc = {k: macro(y, p) for k, p in probs.items()}
    for (cfg, s), a in sorted(auc.items()):
        print(f"{cfg:>6} seed {s:>2}: macro-AUC {a:.4f}")
    res = {"n_test": int(te.sum()), "seeds": SEEDS, "runs": {f"{c}_s{s}": a for (c, s), a in auc.items()}}
    for cfg in ("meta", "nometa"):
        v = np.array([auc[(cfg, s)] for s in SEEDS])
        res[cfg] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    d = np.array([auc[("meta", s)] - auc[("nometa", s)] for s in SEEDS])
    t_ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    # pooled paired bootstrap: resample test ECGs, average the five paired differences
    rng, boot = np.random.default_rng(0), []
    for _ in range(args.boot):
        i = rng.integers(0, len(y), len(y))
        if (y[i].min(axis=0) == 1).any() or (y[i].max(axis=0) == 0).any():
            continue
        boot.append(np.mean([macro(y[i], probs[("meta", s)][i]) - macro(y[i], probs[("nometa", s)][i]) for s in SEEDS]))
    res["metadata_effect"] = {"per_seed": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                              "t_ci95_over_seeds": [float(t_ci[0]), float(t_ci[1])],
                              "bootstrap_ci95_pooled": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                              "seeds_with_gain": int((d > 0).sum())}
    ens = np.mean([probs[("meta", s)] for s in SEEDS], axis=0)
    singles = np.array([auc[("meta", s)] for s in SEEDS])
    rng, boot_e = np.random.default_rng(1), []
    for _ in range(args.boot):
        i = rng.integers(0, len(y), len(y))
        if (y[i].min(axis=0) == 1).any() or (y[i].max(axis=0) == 0).any():
            continue
        boot_e.append(macro(y[i], ens[i]) - np.mean([macro(y[i], probs[("meta", s)][i]) for s in SEEDS]))
    spread = np.std([probs[("meta", s)] for s in SEEDS], axis=0)       # disagreement between members
    res["ensemble"] = {"macro_auc": macro(y, ens), "per_class": {c: float(roc_auc_score(y[:, j], ens[:, j]))
                                                                 for j, c in enumerate(T.CLASSES)},
                       "gain_over_mean_single": float(macro(y, ens) - singles.mean()),
                       "gain_ci95": [float(np.percentile(boot_e, 2.5)), float(np.percentile(boot_e, 97.5))],
                       "member_sd_median": float(np.median(spread)), "member_sd_p95": float(np.percentile(spread, 95))}
    out = ROOT / "docs" / "results" / "ptbxl_seeds.json"
    out.write_text(json.dumps(res, indent=2))
    m, n, e = res["meta"], res["nometa"], res["metadata_effect"]
    print(f"with age/sex   {m['mean']:.4f} ± {m['sd']:.4f}\nwithout        {n['mean']:.4f} ± {n['sd']:.4f}")
    print(f"paired effect  {e['mean']:+.4f}, t-CI {e['t_ci95_over_seeds']}, bootstrap CI {e['bootstrap_ci95_pooled']}, "
          f"gain in {e['seeds_with_gain']} of 5 seeds")
    print(f"ensemble       {res['ensemble']['macro_auc']:.4f} (+{res['ensemble']['gain_over_mean_single']:.4f} "
          f"over a single model, CI {res['ensemble']['gain_ci95']})")
    return res


if __name__ == "__main__":
    main()
