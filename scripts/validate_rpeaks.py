#!/usr/bin/env python3
"""Score the R-peak detector on the MIT-BIH Arrhythmia Database (48 half-hour records,
every beat annotated by cardiologists), the usual benchmark for QRS detectors.

A detection counts as correct when it lies within 150 ms of an annotated beat, the match
window of ANSI/AAMI EC57 and of PhysioNet's bxb; each annotated beat can be matched once.
The whole record is scored (no 5-minute learning period), and the ventricular flutter
episode of record 207 is left out, as EC57 does for QRS detection.

Detectors compared, all on the first lead (MLII in most MIT-BIH records):
  ours          cardioonco.preprocess.detect_r_peaks;
  browser       the same algorithm in assets/js/dsp.js, run with Node (--js);
  neurokit2     NeuroKit2's default cleaning and peak detection, a widely used reference.
(With two leads, consensus_r_peaks cannot break the tie between them and returns the first
lead's beats, so it is not scored separately.)

Any WFDB database with a RECORDS file and beat annotations ('atr') can be scored with
--data, for example the MIT-BIH Supraventricular Arrhythmia Database (svdb, 128 Hz), which
was not used while developing the detector.

Sensitivity  Se = TP / (TP + FN)   -- share of real beats found
Precision    +P = TP / (TP + FP)   -- share of detections that are real beats

Data (ODC-By 1.0; cite Moody GB, Mark RG. IEEE Eng Med Biol 2001;20(3):45-50):
    python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mitdb')"

Usage:
    python scripts/validate_rpeaks.py [--data data/mitdb] [--out docs/results/rpeaks_mitdb.csv]
    python scripts/validate_rpeaks.py --data data/svdb --out docs/results/rpeaks_svdb.csv
    python scripts/validate_rpeaks.py --js        # also the browser code, with Node on PATH
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco.preprocess import detect_r_peaks  # noqa: E402

BEATS = set("NLRBAaJSVrFejnE/fQ?")
TOL_S = 0.150


def excluded_intervals(ann) -> list[tuple[int, int]]:
    """Ventricular flutter/fibrillation episodes, marked '[' ... ']' in the annotations."""
    out, start = [], None
    for s, sym in zip(ann.sample, ann.symbol):
        if sym == "[":
            start = s
        elif sym == "]" and start is not None:
            out.append((start, s))
            start = None
    return out


def match(ref: np.ndarray, det: np.ndarray, tol: int) -> tuple[int, int, int]:
    """One-to-one matching in time order; returns TP, FN, FP."""
    det = np.sort(det)
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        i = np.searchsorted(det, r)
        best, best_d = -1, tol + 1
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(det) and not used[j] and abs(int(det[j]) - int(r)) < best_d:
                best, best_d = j, abs(int(det[j]) - int(r))
        if best >= 0 and best_d <= tol:
            used[best] = True
            tp += 1
    return tp, len(ref) - tp, int((~used).sum())


def browser_peaks(x: np.ndarray, fs: float, node: str) -> np.ndarray:
    """R peaks from the browser implementation (assets/js/dsp.js) run under Node."""
    with tempfile.TemporaryDirectory() as tmp:
        sig = Path(tmp) / "x.f32"
        np.asarray(x, np.float32).tofile(sig)
        script = (f"const D=require({json.dumps(str(ROOT / 'assets/js/dsp.js'))});"
                  f"const b=require('fs').readFileSync({json.dumps(str(sig))});"
                  "const x=Float64Array.from(new Float32Array(b.buffer,b.byteOffset,b.length/4));"
                  f"console.log(JSON.stringify(D.detectRPeaks(x,{fs}).r));")
        return np.asarray(json.loads(subprocess.check_output([node, "-e", script], text=True)), dtype=int)


def neurokit_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    import neurokit2 as nk
    _, info = nk.ecg_peaks(nk.ecg_clean(x, sampling_rate=fs), sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "mitdb"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "rpeaks_mitdb.csv"))
    ap.add_argument("--no-neurokit", action="store_true", help="skip the NeuroKit2 comparison")
    ap.add_argument("--js", action="store_true", help="also score the browser implementation (needs Node)")
    args = ap.parse_args(argv)
    node = shutil.which("node") if args.js else None
    if args.js and node is None:
        raise SystemExit("--js needs node on PATH")
    data = Path(args.data)
    names = (data / "RECORDS").read_text().split()
    detectors = ["ours"] + (["browser"] if args.js else []) + ([] if args.no_neurokit else ["neurokit2"])
    rows, total = [], {d: np.zeros(3, int) for d in detectors}
    for name in names:
        rec = wfdb.rdrecord(str(data / name))
        ann = wfdb.rdann(str(data / name), "atr")
        fs = float(rec.fs)
        sig = np.nan_to_num(rec.p_signal.T.astype(float))
        ref = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEATS])
        skip = excluded_intervals(ann)
        def keep(idx):
            idx = np.asarray(idx, int)
            return idx[[not any(a <= i <= b for a, b in skip) for i in idx]] if skip else idx
        ref = keep(ref)
        found = {"ours": detect_r_peaks(sig[0], fs)}
        if args.js:
            found["browser"] = browser_peaks(sig[0], fs, node)
        if not args.no_neurokit:
            found["neurokit2"] = neurokit_peaks(sig[0], fs)
        row = [name, str(len(ref))]
        for d in detectors:
            tp, fn, fp = match(ref, keep(found[d]), int(round(TOL_S * fs)))
            total[d] += (tp, fn, fp)
            row += [str(tp), str(fn), str(fp)]
        rows.append(row)
        print(name, " ".join(f"{d}: FN {r[1]:>4} FP {r[2]:>4}" for d, r in
                              zip(detectors, [rows[-1][2 + 3 * k: 5 + 3 * k] for k in range(len(detectors))])), flush=True)
    header = ["record", "beats"] + [f"{d}_{m}" for d in detectors for m in ("tp", "fn", "fp")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(",".join(r) for r in [header] + rows) + "\n")
    summary = {}
    print(f"\n{'detector':<13}{'beats':>8}{'FN':>7}{'FP':>7}{'Se %':>9}{'+P %':>9}")
    for d in detectors:
        tp, fn, fp = total[d]
        se, pp = 100 * tp / (tp + fn), 100 * tp / (tp + fp)
        summary[d] = {"tp": int(tp), "fn": int(fn), "fp": int(fp), "se": se, "ppv": pp}
        print(f"{d:<13}{tp + fn:>8}{fn:>7}{fp:>7}{se:>9.2f}{pp:>9.2f}")
    print(f"per-record table -> {out}")
    return summary


if __name__ == "__main__":
    main()
