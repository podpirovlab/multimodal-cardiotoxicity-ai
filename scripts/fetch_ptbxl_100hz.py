#!/usr/bin/env python3
"""Download only what train_ptbxl.py needs from PTB-XL v1.0.3: the two label tables and the
100 Hz records (about 0.5 GB instead of the 1.7 GB zip that also holds the 500 Hz records).

Files come from PhysioNet's open-data mirror on AWS (s3://physionet-open), many in parallel,
and every file is checked against the SHA-256 sums PhysioNet publishes with the dataset.
PTB-XL is distributed under CC BY 4.0 (Wagner et al., Scientific Data 2020).

Usage (from the repository root):
    python scripts/fetch_ptbxl_100hz.py            # -> data/ptb-xl/
    python scripts/fetch_ptbxl_100hz.py --records500-folds 9 10 --plus
        # also the 500 Hz records of folds 9 and 10 (~0.5 GB) and the GE 12SL measurements of
        # PTB-XL+ v1.0.1 (~90 MB; Strodthoff et al., Scientific Data 2023; CC BY 4.0) in
        # data/ptb-xl-plus/, for scripts/validate_qt.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
PLUS = "https://physionet-open.s3.amazonaws.com/ptb-xl-plus/1.0.1/"
PLUS_FILES = ("features/12sl_features.csv", "features/feature_description.csv", "LICENSE.txt")


def checksums(base: str) -> dict:
    sums = {}
    for line in urllib.request.urlopen(base + "SHA256SUMS.txt", timeout=60).read().decode().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = h
    return sums


def fetch(rel: str, dest: Path, expected: str | None, tries: int = 4, base: str = BASE, flat: bool = False) -> str:
    target = dest / (Path(rel).name if flat else rel)
    if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            data = urllib.request.urlopen(base + rel, timeout=60).read()
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("checksum mismatch")
            target.write_bytes(data)
            return "ok"
        except Exception as exc:  # network hiccup or bad checksum: retry
            if attempt == tries - 1:
                raise RuntimeError(f"{rel}: {exc}") from exc
    return "unreachable"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/ptb-xl")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--records500-folds", type=int, nargs="*", default=[],
                    help="also fetch the 500 Hz records of these strat_folds")
    ap.add_argument("--plus", action="store_true", help="also fetch the PTB-XL+ 12SL feature table")
    ap.add_argument("--plus-out", default="data/ptb-xl-plus")
    args = ap.parse_args(argv)
    dest = Path(args.out)
    if args.plus:
        plus = checksums(PLUS)
        for rel in PLUS_FILES:
            print(rel, fetch(rel, Path(args.plus_out), plus[rel], base=PLUS, flat=True))
    sums = checksums(BASE)
    wanted = [n for n in sums if n in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt")
              or n.startswith("records100/")]
    if args.records500_folds:
        import csv
        import io
        db = urllib.request.urlopen(BASE + "ptbxl_database.csv", timeout=60).read().decode()
        stems = {r["filename_hr"] for r in csv.DictReader(io.StringIO(db))
                 if int(r["strat_fold"]) in set(args.records500_folds)}
        wanted += [n for n in sums if n.rsplit(".", 1)[0] in stems]
    print(f"{len(wanted)} files to fetch into {dest}/")
    done = failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, n, dest, sums[n]): n for n in wanted}
        for f in as_completed(futures):
            try:
                f.result(); done += 1
            except Exception as exc:
                failed += 1; print(exc, file=sys.stderr)
            if (done + failed) % 2000 == 0:
                print(f"  {done + failed}/{len(wanted)}")
    print(f"done: {done} verified, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
