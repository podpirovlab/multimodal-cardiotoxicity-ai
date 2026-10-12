#!/usr/bin/env python3
"""Compare cardioonco.qt with the GE 12SL measurements of PTB-XL+ (500 Hz records).

The 12SL algorithm is used in GE's clinical electrocardiographs; PTB-XL+ publishes its global
QT and QTcF for every PTB-XL ECG (Strodthoff et al., Scientific Data 2023).  It is a reference,
not ground truth: two automatic algorithms are compared, and their disagreement is reported
as a bias (mean difference), the standard deviation of the differences, the mean absolute
difference, and the share within 10 and 20 ms (a Bland-Altman analysis).

Protocol: the rule that combines per-lead T ends (cardioonco.qt.combine) is chosen on fold 9
(`--fold 9`, which prints every rule); fold 10 is then scored once with the chosen rule.

Data: python scripts/fetch_ptbxl_100hz.py --records500-folds 9 10, and the PTB-XL+ feature
tables in data/ptb-xl-plus/ (CC BY 4.0).

    python scripts/validate_qt.py --fold 9
    python scripts/validate_qt.py --fold 10
    python scripts/validate_qt.py --fold 9 --hz100     # how much is lost at 100 Hz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco import qt as Q  # noqa: E402
from cardioonco.preprocess import consensus_r_peaks  # noqa: E402

RULES = ("median", "p75", "second_latest", "latest")
SINGLE_LEADS = {"I": 0, "II": 1}          # lead I is what a smartwatch records


def summary(ours: np.ndarray, ref: np.ndarray) -> dict:
    d = ours - ref
    return {"n": int(len(d)), "bias_ms": float(d.mean()), "sd_ms": float(d.std(ddof=1)),
            "mae_ms": float(np.abs(d).mean()), "within_10_ms": float(np.mean(np.abs(d) <= 10)),
            "within_20_ms": float(np.mean(np.abs(d) <= 20)), "pearson_r": float(np.corrcoef(ours, ref)[0, 1]),
            "loa_ms": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "ptb-xl"))
    ap.add_argument("--plus", default=str(ROOT / "data" / "ptb-xl-plus"))
    ap.add_argument("--hz100", action="store_true",
                    help="development folds only: the 100 Hz records, interpolated to 500 Hz before measuring")
    args = ap.parse_args(argv)
    if args.hz100 and args.fold == 10:
        raise SystemExit("--hz100 is a development study; fold 10 has been scored once")
    data, plus = Path(args.data), Path(args.plus)
    db = list(csv.DictReader(open(data / "ptbxl_database.csv")))
    ref = {r["ecg_id"]: r for r in csv.DictReader(open(plus / "12sl_features.csv"))}
    rows, failed, no_ref = [], 0, 0
    for r in (r for r in db if int(r["strat_fold"]) == args.fold):
        f = ref.get(r["ecg_id"])
        try:
            qt_ref, qtcf_ref = float(f["QT_Int_Global"]), float(f["QT_IntFridericia_Global"])
            assert qt_ref > 0 and qtcf_ref > 0
        except (TypeError, ValueError, AssertionError, KeyError):
            no_ref += 1
            continue
        rec = wfdb.rdrecord(str(data / r["filename_lr" if args.hz100 else "filename_hr"]))
        x, fs = np.nan_to_num(rec.p_signal.T.astype(float)), rec.fs
        if args.hz100:
            x, fs = resample_poly(x, 5, 1, axis=1), fs * 5
        try:
            res = Q.measure_qt(x, fs, r_peaks=consensus_r_peaks(x, fs))
        except ValueError:
            failed += 1
            continue
        ends = np.array(res.t_end_per_lead_ms) / 1000
        row = {"ecg_id": int(r["ecg_id"]), "qt_12sl": qt_ref, "qtcf_12sl": qtcf_ref, "rr_ms": res.rr_ms,
               "iqr": res.t_end_iqr_ms, "n_leads": res.n_leads_used, "reliable": res.reliable,
               "qt_ours": res.qt_ms, "qtcf_ours": res.qtcf_ms}
        for rule in RULES:                    # raw tangent QT under every combination rule
            row[f"raw_{rule}"] = Q.combine(ends, rule) * 1000 - res.qrs_onset_ms
        if args.fold != 10:                   # development: one lead alone, as from a single-channel device
            for name, i in SINGLE_LEADS.items():
                try:
                    row[f"lead_{name}"] = Q.measure_qt(x[i:i + 1], fs).qt_ms
                except ValueError:
                    row[f"lead_{name}"] = float("nan")
        rows.append(row)
    rows = [r for r in rows if np.isfinite(r["qt_ours"])]
    out = {"fold": args.fold, "records": "100 Hz, interpolated to 500 Hz" if args.hz100 else "500 Hz", "measured": len(rows), "failed": failed, "no_12sl_value": no_ref,
           "rule": Q.T_END_RULE, "qt_offset_ms": Q.QT_OFFSET_MS, "max_t_end_iqr_ms": Q.MAX_T_END_IQR_MS}
    ref_qt = np.array([r["qt_12sl"] for r in rows])
    if args.fold != 10:                       # development: compare rules, fit the offset, study the flag
        for rule in RULES:
            v = np.array([r[f"raw_{rule}"] for r in rows])
            ok = np.isfinite(v)
            d = v[ok] - ref_qt[ok]
            out[f"raw_{rule}"] = {"median_diff_ms": float(np.median(d)), "robust_sd_ms": float(1.4826 * np.median(np.abs(d - np.median(d)))),
                                   "sd_ms": float(d.std(ddof=1))}
            print(f"raw {rule:>14}: median diff {np.median(d):+.1f} ms, robust SD {out[f'raw_{rule}']['robust_sd_ms']:.1f}, SD {d.std(ddof=1):.1f}")
        d = np.array([r[f"raw_{Q.T_END_RULE}"] for r in rows]) - ref_qt
        out["fitted_offset_ms"] = float(-np.median(d))
        iqr = np.array([r["iqr"] for r in rows])
        for thr in (30, 40, 60, 80, 100):
            keep = iqr <= thr
            c = d[keep] - np.median(d)
            print(f"IQR <= {thr:>3} ms keeps {100 * keep.mean():5.1f}%: |error| > 50 ms in {100 * np.mean(np.abs(c) > 50):.1f}%, SD {c.std():.1f}")
        for name in SINGLE_LEADS:             # with the 12-lead offset, as the site reports it
            v = np.array([r[f"lead_{name}"] for r in rows])
            ok = np.isfinite(v)
            out[f"single_lead_{name}"] = dict(summary(v[ok], ref_qt[ok]), measured_share=float(ok.mean()))
            q = out[f"single_lead_{name}"]
            print(f"lead {name} alone ({100 * ok.mean():.0f}% measured): bias {q['bias_ms']:+.1f} ms, SD {q['sd_ms']:.1f}, "
                  f"MAE {q['mae_ms']:.1f}, within 20 ms {100 * q['within_20_ms']:.1f}%")
    for subset, sel in (("all", np.ones(len(rows), bool)), ("reliable", np.array([r["reliable"] for r in rows]))):
        if sel.sum() < 3:
            continue
        out[subset] = {"share": float(sel.mean()),
                       "qt": summary(np.array([r["qt_ours"] for r in rows])[sel], ref_qt[sel]),
                       "qtcf": summary(np.array([r["qtcf_ours"] for r in rows])[sel], np.array([r["qtcf_12sl"] for r in rows])[sel])}
        q = out[subset]["qt"]
        print(f"{subset:>8} ({100 * sel.mean():.0f}%): QT bias {q['bias_ms']:+.1f} ms, SD {q['sd_ms']:.1f}, MAE {q['mae_ms']:.1f}, "
              f"within 20 ms {100 * q['within_20_ms']:.1f}%, r {q['pearson_r']:.3f}")
    print(f"measured {len(rows)}, failed {failed}, no 12SL value {no_ref}")
    res_dir = ROOT / "docs" / "results"
    stem = f"qt_ptbxl_fold{args.fold}" + ("_100hz" if args.hz100 else "")
    (res_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))
    with open(res_dir / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ecg_id", "qt_12sl_ms", "qtcf_12sl_ms", "rr_ms", "qt_ours_ms", "qtcf_ours_ms", "reliable"])
        for r in rows:
            w.writerow([r["ecg_id"], f"{r['qt_12sl']:.0f}", f"{r['qtcf_12sl']:.0f}", f"{r['rr_ms']:.0f}",
                        f"{r['qt_ours']:.1f}", f"{r['qtcf_ours']:.1f}", int(r["reliable"])])
    return out


if __name__ == "__main__":
    main()
