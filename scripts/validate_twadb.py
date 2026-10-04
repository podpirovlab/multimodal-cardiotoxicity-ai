#!/usr/bin/env python3
"""Score the TWA pipeline on the PhysioNet/CinC Challenge 2008 T-Wave Alternans Database.

The challenge asked for one TWA magnitude per record and scored entries by Kendall's
rank correlation (tau) against reference ranks built from the consensus of the entries.
This script does the same with cardioonco.twa.analyze.

Primary metric:
  v0.4.0  V_alt of the first 128 beats, maximum over leads;
  v0.5.0  estimate_uv -- per lead, the largest V_alt among 128-beat windows that pass
          K >= 3 across the whole recording (0 if none), i.e. the Spectral Method's
          significance test applied before taking a magnitude; maximum over leads;
  v0.6.0  the same per lead; per record, the median over leads, so that one noisy lead
          cannot carry the record.
V_alt of the reported window and MMA are secondary (same aggregation).  From v0.5.0
the leads of a record share one set of R peaks (cardioonco.preprocess.consensus_r_peaks),
because the leads are simultaneous and a lead with a tall T wave can otherwise put the
fiducial point on it.  A record where no lead can be analysed
gets 0, because the challenge required an estimate for every record; tau over analysable
records only is reported alongside.

Development protocol:
  1. v0.5.0 was developed on the 32 synthetic records only, frozen, and the 68 real
     records scored once: tau 0.00 on real records (the held-out result for that stage).
  2. The real records were then split in two halves by source, with a fixed seed
     (split_real): 'real-dev' (34) for developing v0.6.0, 'real-test' (34) scored once
     after it was frozen.  Only the real-test number is a held-out result, and not a
     perfectly clean one: the stage-1 tau over all 68 real records had been seen.

Caveats that belong next to any number this prints:
  * the reference is a consensus of 2008 entries, not ground-truth microvolts;
  * 32 of the 100 records are synthetic (sources "syn"/"sync" in about-records.txt).

Data (ODC-By 1.0, cite Moody GB, Computing in Cardiology 2008;35:505-508):
    python -c "import wfdb; wfdb.dl_database('twadb', dl_dir='data/twadb')"
    curl -o data/twadb/reference-ranks https://physionet.org/files/challenge-2008/1.0.0/reference-ranks
    curl -o data/twadb/about-records.txt https://physionet.org/files/challenge-2008/1.0.0/about-records.txt

Usage:
    python scripts/validate_twadb.py [--data data/twadb] [--out runs/twadb]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco import __version__  # noqa: E402
from cardioonco.preprocess import consensus_r_peaks  # noqa: E402
from cardioonco.twa import analyze  # noqa: E402

METRICS = ("estimate_uv", "v_alt_uv", "mma_uv")
SYNTHETIC = {"syn", "sync"}


def read_ranks(path: Path) -> dict[str, int]:
    ranks = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            name, rank = line.split()[:2]
            ranks[name] = int(rank)
    return ranks


def read_sources(path: Path) -> dict[str, str]:
    sources, in_table = {}, False
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts[:1] == ["twadb"]:
            in_table = True
            continue
        if in_table and len(parts) >= 3 and parts[0].startswith("twa"):
            sources[parts[0]] = parts[2]
    return sources


def split_real(sources: dict[str, str], seed: int = 2008) -> dict[str, str]:
    """Split the real records in two halves, stratified by source database, with a fixed
    seed: 'real-dev' for development from v0.6.0 on, 'real-test' scored once per frozen
    version.  Returns {record: 'real-dev' | 'real-test'}."""
    rng = np.random.default_rng(seed)
    out = {}
    for src in sorted({s for s in sources.values() if s not in SYNTHETIC}):
        names = rng.permutation(sorted(n for n, s in sources.items() if s == src))
        for i, n in enumerate(names):
            out[str(n)] = "real-dev" if i % 2 == 0 else "real-test"
    return out


def analyse_record(path: Path) -> dict:
    rec = wfdb.rdrecord(str(path))
    signals = np.nan_to_num(rec.p_signal.T.astype(float))
    r_peaks = consensus_r_peaks(signals, rec.fs)
    leads, failures = [], []
    for j, name in enumerate(rec.sig_name):
        x = signals[j]
        try:
            r = analyze(x, rec.fs, r_peaks=r_peaks)
        except ValueError as e:
            failures.append({"lead": name, "reason": str(e)})
            continue
        leads.append({"lead": name, "n_beats": r.n_beats, "hr": r.heart_rate_bpm, "k": r.k_score,
                      "outcome": r.outcome, "reason": r.reason, **{m: getattr(r, m) for m in METRICS}})
    best = {m: (float(np.median([l[m] for l in leads])) if leads else 0.0) for m in METRICS}
    return {"fs": rec.fs, "n_leads": rec.n_sig, "leads": leads, "failures": failures, "best": best}


def write_scores_csv(path: Path, records: dict, ranks: dict, sources: dict, half: dict) -> None:
    """One row per record: its group, the challenge reference rank and this pipeline's values.
    docs/results/ keeps the copy behind README section 7.1 and figure 12."""
    lines = ["record,group,source,reference_rank," + ",".join(METRICS)]
    for n in sorted(records):
        group = "synthetic" if sources.get(n) in SYNTHETIC else half.get(n, "unknown")
        lines.append(f"{n},{group},{sources.get(n, '')},{ranks[n]},"
                     + ",".join(f"{records[n]['best'][m]:.4f}" for m in METRICS))
    path.write_text("\n".join(lines) + "\n")


def tau(ours: list[float], ref: list[int]) -> float:
    return float(kendalltau(ours, ref).statistic) if len(ours) > 2 else float("nan")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "twadb"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "twadb"))
    ap.add_argument("--only", choices=["all", "synthetic", "real", "real-dev", "real-test"], default="all",
                    help="analyse only this subset (development runs use 'synthetic' and 'real-dev')")
    args = ap.parse_args(argv)
    data, out = Path(args.data), Path(args.out)

    ranks = read_ranks(data / "reference-ranks")
    sources = read_sources(data / "about-records.txt")
    half = split_real(sources)
    wanted = {"all": lambda n: True, "synthetic": lambda n: sources.get(n) in SYNTHETIC,
              "real": lambda n: n in half, "real-dev": lambda n: half.get(n) == "real-dev",
              "real-test": lambda n: half.get(n) == "real-test"}[args.only]
    records = {name: analyse_record(data / name) for name in sorted(ranks) if wanted(name)}

    names = sorted(records)
    analysable = [n for n in names if records[n]["leads"]]
    groups = {
        "all": names,
        "analysable": analysable,
        "real": [n for n in names if n in half],
        "real-dev": [n for n in names if half.get(n) == "real-dev"],
        "real-test": [n for n in names if half.get(n) == "real-test"],
        "synthetic": [n for n in names if sources.get(n) in SYNTHETIC],
    }
    groups = {g: ns for g, ns in groups.items() if ns}
    scores = {g: {m: tau([records[n]["best"][m] for n in ns], [ranks[n] for n in ns]) for m in METRICS}
              for g, ns in groups.items()}

    summary = {
        "algorithm_version": __version__,
        "subset": args.only,
        "primary_metric": "estimate_uv (median over leads)",
        "kendall_tau": scores,
        "n_records": {g: len(ns) for g, ns in groups.items()},
        "records_without_any_analysable_lead": [n for n in names if not records[n]["leads"]],
        "lead_failures": sum(len(r["failures"]) for r in records.values()),
        "caveats": ["reference ranks are a consensus of 2008 challenge entries, not true amplitudes",
                    "32 of 100 records are synthetic"],
    }
    out.mkdir(parents=True, exist_ok=True)
    tag = f"v{__version__}" + ("" if args.only == "all" else f"_{args.only}")
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    (out / f"records_{tag}.json").write_text(json.dumps(records, indent=2))
    write_scores_csv(out / f"scores_{tag}.csv", records, ranks, sources, half)

    print(f"CardioOncoPredict {__version__} on the TWA Challenge Database")
    print(f"{'group':<12}{'n':>5}  " + "".join(f"{m:>16}" for m in METRICS))
    for g, ns in groups.items():
        print(f"{g:<12}{len(ns):>5}  " + "".join(f"{scores[g][m]:>16.3f}" for m in METRICS))
    print(f"records with no analysable lead: {len(summary['records_without_any_analysable_lead'])}, "
          f"lead failures: {summary['lead_failures']}")
    print(f"primary: Kendall tau = {scores['all']['estimate_uv']:.3f} "
          f"(significance-gated V_alt, {len(names)} records, subset '{args.only}')")
    return summary


if __name__ == "__main__":
    main()
