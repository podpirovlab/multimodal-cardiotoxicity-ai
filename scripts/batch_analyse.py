#!/usr/bin/env python3
"""Analyse every ECG file in a folder and write one CSV table, with the same columns as the web
tool's "Choose several files" (assets/js/site.js, BATCH_COLS).

For each file: TWA in the default lead (the first labelled ECG or lead II, else the first signal),
the two-adjacent-chest-leads rule over all standard leads when there are at least two chest leads
(the first 10 minutes), and QT/QTc when there are at least six standard leads.  EDF/EDF+ and
BDF/BDF+ are read with pyedflib, WFDB records (.hea) with wfdb; CSV/TXT files hold one column
of numbers and need --fs (and --col, --unit).

    python scripts/batch_analyse.py my_ecgs/ --out results.csv
    python scripts/batch_analyse.py csv_folder/ --fs 500 --unit uv --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco.edf import lead_from_label, unit_to_mv  # noqa: E402
from cardioonco.qt import MIN_LEADS, measure_qt_segments  # noqa: E402
from cardioonco.twa import PRECORDIAL, analyze, analyze_leads  # noqa: E402

COLS = ["file", "format", "fs_hz", "duration_s", "n_leads", "lead", "n_beats", "heart_rate_bpm",
        "v_alt_uv", "k_score", "noise_uv", "twa_outcome", "twa_reason", "all_leads_outcome", "all_leads_reason",
        "positive_pairs", "qt_ms", "qtcf_ms", "qtcb_ms", "qt_reliable", "error"]
MULTI_MAX_S = 600.0
SUFFIXES = (".edf", ".bdf", ".hea", ".csv", ".txt")


def read_signals(path: Path, args) -> tuple[str, list[str], list[np.ndarray], list[float]]:
    """(format, labels, signals in mV, sampling rates) of one file."""
    suffix = path.suffix.lower()
    if suffix in (".edf", ".bdf"):
        import pyedflib
        with pyedflib.EdfReader(str(path)) as f:
            n = f.signals_in_file
            labels = [f.getLabel(i).strip() for i in range(n)]
            sig = [f.readSignal(i) * (unit_to_mv(f.getPhysicalDimension(i)) or 1.0) for i in range(n)]
            rates = [float(f.getSampleFrequency(i)) for i in range(n)]
        return ("BDF" if suffix == ".bdf" else "EDF"), labels, sig, rates
    if suffix == ".hea":
        import wfdb
        rec = wfdb.rdrecord(str(path.with_suffix("")))
        x = np.nan_to_num(rec.p_signal.T.astype(float))
        return "WFDB", list(rec.sig_name), [x[i] * (unit_to_mv(u) or 1.0) for i, u in enumerate(rec.units)], [float(rec.fs)] * rec.n_sig
    if args.fs is None:
        raise ValueError("a CSV/TXT file needs --fs")
    x = np.loadtxt(path, delimiter=",", usecols=[args.col], ndmin=1, comments="#", skiprows=args.skip)
    return "CSV", [f"column {args.col}"], [x / 1000 if args.unit == "uv" else x], [float(args.fs)]


def analyse_file(path: Path, args) -> dict:
    row = {"file": path.name}
    fmt, labels, sig, rates = read_signals(path, args)
    guess = next((i for i, lab in enumerate(labels) if re.search(r"ecg|ekg|\bii\b|lead", lab, re.I)), 0)
    fs = rates[guess]
    row.update(format=fmt, fs_hz=f"{fs:.1f}", duration_s=f"{len(sig[guess]) / fs:.1f}", n_leads=len(labels),
               lead=labels[guess])
    try:
        r = analyze(sig[guess], fs)
        row.update(n_beats=r.n_beats, heart_rate_bpm=f"{r.heart_rate_bpm:.1f}", v_alt_uv=f"{r.v_alt_uv:.2f}",
                   k_score=f"{r.k_score:.2f}", noise_uv=f"{r.noise_uv:.2f}", twa_outcome=r.outcome, twa_reason=r.reason)
    except ValueError as exc:
        code = "too_short" if "too short" in str(exc) else "no_noise_estimate" if "noise band" in str(exc) else "error"
        row.update(twa_outcome="not_analysed", twa_reason=code, error=str(exc))   # the codes of dsp.js
    # the standard leads that share the commonest sampling rate, one channel per lead
    by_rate: dict[float, dict[str, int]] = {}
    for i, lab in enumerate(labels):
        lead = lead_from_label(lab)
        if lead:
            by_rate.setdefault(rates[i], {}).setdefault(lead, i)
    if by_rate:
        rate, leads = max(by_rate.items(), key=lambda kv: len(kv[1]))
        names, idx = list(leads), list(leads.values())
        if sum(n in PRECORDIAL for n in names) >= 2 and row.get("twa_outcome") != "not_analysed":
            cap = int(round(MULTI_MAX_S * rate))
            m = analyze_leads(np.array([sig[i][:cap] for i in idx]), rate, names)
            row.update(all_leads_outcome=m.outcome, all_leads_reason=m.reason,
                       positive_pairs=" ".join("-".join(p) for p in m.positive_pairs))
        if len(idx) >= MIN_LEADS:
            try:
                q = measure_qt_segments(np.array([sig[i] for i in idx]), rate)
                row.update(qt_ms=f"{q.qt_ms:.1f}", qtcf_ms=f"{q.qtcf_ms:.1f}", qtcb_ms=f"{q.qtcb_ms:.1f}",
                           qt_reliable=int(q.reliable))
            except ValueError:
                pass
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="cardioonco_batch.csv")
    ap.add_argument("--fs", type=float, help="sampling rate of CSV/TXT files, Hz")
    ap.add_argument("--col", type=int, default=0, help="column of CSV/TXT files (0 = first)")
    ap.add_argument("--unit", choices=["mv", "uv"], default="mv")
    ap.add_argument("--skip", type=int, default=0, help="header lines to skip in CSV/TXT files")
    args = ap.parse_args(argv)
    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in SUFFIXES)
    rows = []
    for k, path in enumerate(files, 1):
        print(f"[{k}/{len(files)}] {path.name}", file=sys.stderr)
        try:
            rows.append(analyse_file(path, args))
        except Exception as exc:          # one unreadable file must not stop the batch
            msg = str(exc).replace(str(path), path.name)      # no local folder names in the table
            rows.append({"file": path.name, "twa_outcome": "not_analysed", "twa_reason": "error", "error": msg})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    counts = {o: sum(r.get("twa_outcome") == o for r in rows) for o in ("positive", "negative", "indeterminate", "not_analysed")}
    print(f"{len(rows)} files -> {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
