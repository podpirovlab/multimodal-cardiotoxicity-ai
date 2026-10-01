#!/usr/bin/env python3
"""Download only what train_ptbxl.py needs from PTB-XL v1.0.3: the two label tables and the
100 Hz records (about 0.5 GB instead of the 1.7 GB zip that also holds the 500 Hz records).

Files come from PhysioNet's open-data mirror on AWS (s3://physionet-open), many in parallel,
and every file is checked against the SHA-256 sums PhysioNet publishes with the dataset.
PTB-XL is distributed under CC BY 4.0 (Wagner et al., Scientific Data 2020).

Usage (from the repository root):
    python scripts/fetch_ptbxl_100hz.py            # -> data/ptb-xl/
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def fetch(rel: str, dest: Path, expected: str | None, tries: int = 4) -> str:
    target = dest / rel
    if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            data = urllib.request.urlopen(BASE + rel, timeout=60).read()
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
    args = ap.parse_args(argv)
    dest = Path(args.out)
    sums = {}
    for line in urllib.request.urlopen(BASE + "SHA256SUMS.txt", timeout=60).read().decode().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = h
    wanted = [n for n in sums if n in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt")
              or n.startswith("records100/")]
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
