#!/usr/bin/env python3
"""Download the healthy controls of the PTB Diagnostic ECG Database (PhysioNet ptbdb 1.0.0):
the first record of each of the 52 control subjects listed in its CONTROLS file, the 12 standard
leads and the 3 Frank leads at 1000 Hz, about 2 minutes each (~220 MB), every file checked against PhysioNet's SHA-256 sums.

Licence: Open Data Commons Attribution License v1.0.  Cite Bousseljot R, Kreiseler D, Schnabel A.
Nutzung der EKG-Signaldatenbank CARDIODAT der PTB ueber das Internet. Biomedizinische Technik
1995;40(S1):317, and PhysioNet (Goldberger et al., Circulation 2000).

    python scripts/fetch_ptbdb_controls.py            # -> data/ptbdb/
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_ptbxl_100hz import checksums, fetch  # noqa: E402

BASE = "https://physionet-open.s3.amazonaws.com/ptbdb/1.0.0/"


def control_records() -> list[str]:
    """One record per control subject: the first one CONTROLS lists for each patient."""
    seen, out = set(), []
    for line in urllib.request.urlopen(BASE + "CONTROLS", timeout=60).read().decode().split():
        patient = line.split("/")[0]
        if patient not in seen:
            seen.add(patient)
            out.append(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/ptbdb")
    args = ap.parse_args(argv)
    sums, recs = checksums(BASE), control_records()
    dest = Path(args.out)
    failed = 0
    for rec in recs:
        for ext in (".hea", ".dat", ".xyz"):           # .xyz: the Frank leads vx, vy, vz
            try:
                fetch(rec + ext, dest, sums[rec + ext], base=BASE)
            except Exception as exc:
                failed += 1
                print(exc, file=sys.stderr)
    (dest / "CONTROLS_FIRST").write_text("\n".join(recs) + "\n")
    print(f"{len(recs)} control subjects, {failed} failed downloads -> {dest}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
