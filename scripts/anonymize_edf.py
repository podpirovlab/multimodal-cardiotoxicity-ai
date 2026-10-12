#!/usr/bin/env python3
"""Write de-identified copies of EDF/EDF+ and BDF/BDF+ files (cardioonco/edf.py; the web tool's
"Download a de-identified copy" button writes byte-for-byte the same file).

Patient and recording fields become "X", the start date and time 01.01.85 00.00.00, and annotation
text is cleared; the signal is not changed.  The ECG itself can still be personal data.

    python scripts/anonymize_edf.py recording.edf                 # -> recording_anon.edf
    python scripts/anonymize_edf.py data/raw --out data/anon      # every .edf/.bdf in a folder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco.edf import anonymize_edf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="an EDF/BDF file or a folder of them")
    ap.add_argument("--out", help="output folder (default: next to each file, with _anon added)")
    args = ap.parse_args(argv)
    src = Path(args.source)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".edf", ".bdf")) if src.is_dir() else [src]
    if not files:
        print(f"no .edf or .bdf files in {src}", file=sys.stderr)
        return 1
    failed = 0
    for f in files:
        try:
            data, removed = anonymize_edf(f.read_bytes())
        except (ValueError, IndexError) as exc:
            print(f"{f}: skipped ({exc})", file=sys.stderr)
            failed += 1
            continue
        if args.out:
            dest = Path(args.out) / (f.relative_to(src) if src.is_dir() else f.name)
        else:
            dest = f.with_name(f.stem + "_anon" + f.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        kinds = [k for k in ("patient", "recording") if removed[k].strip("X ").removeprefix("Startdate").strip("X ")]
        print(f"{dest}: cleared {', '.join(kinds) or 'no names'}, date {removed['startdate']}, "
              f"{removed['annotations']} annotation texts")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
