#!/usr/bin/env python3
"""Convert one lead of a PhysioNet WFDB record (.hea/.dat, e.g. PTB-XL) to a
single-column CSV that the browser lab (index.html / ru.html) can upload directly.

Usage:
    python scripts/wfdb_to_csv.py path/to/00001_lr --lead II --out ecg.csv

`path/to/00001_lr` is the record name without extension (both 00001_lr.hea
and 00001_lr.dat must sit next to each other, exactly as PhysioNet ships them).
"""
import argparse
import csv
import sys

import wfdb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", help="WFDB record path without extension, e.g. records100/00000/00001_lr")
    ap.add_argument("--lead", default="II", help="Lead name to export (default: II). Case-insensitive.")
    ap.add_argument("--out", default="ecg.csv", help="Output CSV path (default: ecg.csv)")
    args = ap.parse_args(argv)

    rec = wfdb.rdrecord(args.record)
    names = [s.strip() for s in rec.sig_name]
    lower = [n.lower() for n in names]
    want = args.lead.strip().lower()
    if want in lower:
        idx = lower.index(want)
    else:
        print(f"Lead '{args.lead}' not found. Available leads: {', '.join(names)}. Using {names[0]} instead.",
              file=sys.stderr)
        idx = 0

    signal = rec.p_signal[:, idx]  # physical units, normally mV for PTB-XL
    unit = (rec.units[idx] or "mV").strip()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        for v in signal:
            w.writerow([f"{v:.6f}"])

    print(f"Wrote {len(signal)} samples of lead {names[idx]} ({unit}) at {rec.fs} Hz to {args.out}")
    print(f"On the upload form, set Sampling = {int(rec.fs)} Hz and Units = {'mV' if unit.lower() == 'mv' else 'µV'}.")


if __name__ == "__main__":
    main()
