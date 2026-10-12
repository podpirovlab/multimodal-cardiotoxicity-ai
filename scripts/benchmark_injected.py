#!/usr/bin/env python3
"""A benchmark with a known answer: alternans of a set size injected into real ECGs of healthy people.

The 52 healthy controls of the PTB Diagnostic ECG Database (scripts/fetch_ptbdb_controls.py) give
real noise, baseline wander, heart-rate variability and beat-to-beat T-wave variation, which a
synthetic ECG lacks.  Into lead II and lead V5 of each recording, alternans of amplitude a is
added beat by beat: +a on even beats and -a on odd beats, as a Gaussian (sigma 40 ms) centred on
the peak of the T wave of the recording's median beat, so that |A - B| / 2 = a at that peak, the
convention of cardioonco.synth.  The beats come from the clean recording; the analysis then runs
from scratch on the modified signal (its own filtering, R peaks, alignment and windows).

Reported per lead and amplitude: the share of recordings called positive (Wilson 95% CI); the share
in which significant alternans was found at all (V_alt >= 1.9 uV with K >= 3 in some window, the
noise and heart-rate rules aside); and how well the peak estimate V_alt,peak recovers a.  At a = 0 a positive call is a false positive.  At
rest the heart rate stays below 105 bpm, so "negative" cannot occur (§4.4): a recording that is
not positive is indeterminate.

    python scripts/benchmark_injected.py            # -> docs/results/benchmark_injected.{csv,json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco.preprocess import bandpass, detect_r_peaks  # noqa: E402
from cardioonco.twa import analyze  # noqa: E402

AMPLITUDES_UV = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
LEADS = ("ii", "v5")
SIGMA_S = 0.040


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def t_peak_offset(x: np.ndarray, r: np.ndarray, fs: float) -> float:
    """Seconds from R to the peak of the T wave of the median beat."""
    xf = bandpass(x, fs, 0.5, 40.0)
    rr = np.median(np.diff(r)) / fs
    scale = np.sqrt(rr / 0.8)
    a, b = int(0.15 * scale * fs), int(0.45 * scale * fs)
    beats = np.array([xf[k + a:k + b] for k in r if k + b < len(xf)])
    med = np.median(beats, axis=0)
    base = np.median(np.array([xf[k - int(0.08 * fs):k - int(0.04 * fs)] for k in r if k > 0.1 * fs]))
    return (a + int(np.argmax(np.abs(med - base)))) / fs


def inject(x: np.ndarray, r: np.ndarray, fs: float, amp_uv: float, t_peak_s: float) -> np.ndarray:
    y = x.copy()
    half = int(4 * SIGMA_S * fs)
    g = np.exp(-0.5 * (np.arange(-half, half + 1) / (SIGMA_S * fs)) ** 2) * amp_uv / 1000.0
    for k, rk in enumerate(r):
        c = rk + int(round(t_peak_s * fs))
        if c - half >= 0 and c + half < len(y):
            y[c - half:c + half + 1] += g if k % 2 == 0 else -g
    return y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "ptbdb"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "benchmark_injected"))
    args = ap.parse_args(argv)
    data = Path(args.data)
    records = (data / "CONTROLS_FIRST").read_text().split()
    rows = []
    for rec_name in records:
        rec = wfdb.rdrecord(str(data / rec_name), channel_names=list(LEADS))
        for lead in LEADS:
            x = np.nan_to_num(rec.p_signal[:, rec.sig_name.index(lead)].astype(float))
            r = detect_r_peaks(x, rec.fs)
            tp = t_peak_offset(x, r, rec.fs)
            for amp in AMPLITUDES_UV:
                row = {"record": rec_name, "lead": lead.upper() if lead != "ii" else "II", "amp_uv": amp,
                       "t_peak_ms": round(1000 * tp)}
                try:
                    res = analyze(inject(x, r, rec.fs, amp, tp), rec.fs)
                    row.update(hr=round(res.heart_rate_bpm, 1), outcome=res.outcome, reason=res.reason,
                               v_alt_uv=round(res.v_alt_uv, 3), v_alt_peak_uv=round(res.v_alt_peak_uv, 3),
                               k=round(res.k_score, 2), noise_uv=round(res.noise_uv, 3), estimate_uv=round(res.estimate_uv, 3))
                except ValueError as exc:
                    row.update(outcome="not_analysed", reason=str(exc)[:60])
                rows.append(row)
        print(f"{rec_name}: done", file=sys.stderr)
    out = Path(args.out)
    cols = ["record", "lead", "amp_uv", "t_peak_ms", "hr", "outcome", "reason", "v_alt_uv", "v_alt_peak_uv", "k",
            "noise_uv", "estimate_uv"]
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summary = {"records": len(records), "leads": [lead.upper() if lead != "ii" else "II" for lead in LEADS],
               "sigma_ms": 1000 * SIGMA_S, "by_lead_and_amplitude": []}
    for lead in summary["leads"]:
        for amp in AMPLITUDES_UV:
            sel = [r for r in rows if r["lead"] == lead and r["amp_uv"] == amp and r["outcome"] != "not_analysed"]
            pos = sum(r["outcome"] == "positive" for r in sel)
            found = sum(r["estimate_uv"] >= 1.9 for r in sel)
            peak = np.array([r["v_alt_peak_uv"] for r in sel])
            item = {"lead": lead, "amp_uv": amp, "n": len(sel), "positive": pos,
                    "positive_share": pos / len(sel) if sel else float("nan"),
                    "positive_ci95": wilson(pos, len(sel)),
                    "found": found, "found_share": found / len(sel) if sel else float("nan"),
                    "found_ci95": wilson(found, len(sel)),
                    "indeterminate_reasons": {q: sum(r["reason"] == q for r in sel if r["outcome"] == "indeterminate")
                                              for q in sorted({r["reason"] for r in sel if r["outcome"] == "indeterminate"})},
                    "v_alt_peak_median_uv": float(np.median(peak)) if sel else float("nan"),
                    "v_alt_peak_iqr_uv": [float(v) for v in np.percentile(peak, [25, 75])] if sel else None}
            if amp > 0 and sel:
                item["peak_over_injected_median"] = float(np.median(peak / amp))
            summary["by_lead_and_amplitude"].append(item)
            print(f"{lead:>3} {amp:5.1f} uV: positive {pos}/{len(sel)}, found {found}/{len(sel)}, "
                  f"V_alt,peak median {item['v_alt_peak_median_uv']:.2f}")
    hrs = [r["hr"] for r in rows if "hr" in r]
    summary["heart_rate_bpm"] = {"median": float(np.median(hrs)), "range": [float(min(hrs)), float(max(hrs))]}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
