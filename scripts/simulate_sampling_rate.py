#!/usr/bin/env python3
"""How does the sampling rate change what the TWA analysis finds?  (README section 4.5)

Synthetic single-lead recordings (160 beats at 108 bpm, white noise) are made at
500 Hz (clinical ECG), 250 Hz (many Holter recorders) and 130 Hz (a Polar H10 chest strap),
with 0, 5 or 10 uV of alternans, 24 recordings per cell with fixed seeds.  At 130 Hz the
analysis is run twice: as it was up to version 1.2, on the samples as recorded, and as it
is from 1.3, after interpolation to 520 Hz (cardioonco.twa.MIN_ANALYSIS_FS).

    python scripts/simulate_sampling_rate.py [--out docs/results/sampling_rate.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cardioonco.twa as twa  # noqa: E402
from cardioonco.synth import SynthConfig, generate_ecg  # noqa: E402

N = 24


def positives(fs: int, noise: float, alt: float, interpolate: bool) -> int:
    saved = twa.MIN_ANALYSIS_FS
    twa.MIN_ANALYSIS_FS = saved if interpolate else 0.0
    try:
        n = 0
        for seed in range(N):
            _, x, _ = generate_ecg(SynthConfig(fs=fs, alternans_uv=alt, noise_uv=noise, heart_rate=108,
                                               n_beats=160, seed=3000 + seed))
            try:
                n += twa.analyze(x, fs).outcome == "positive"
            except ValueError:
                pass
        return n
    finally:
        twa.MIN_ANALYSIS_FS = saved


def main(argv=None) -> list[list]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "sampling_rate.csv"))
    args = ap.parse_args(argv)
    rows = []
    for fs, interpolate in ((500, True), (250, True), (130, False), (130, True)):
        for noise in (5, 20):
            row = [fs, "yes" if interpolate and fs < twa.MIN_ANALYSIS_FS else "no", noise]
            row += [positives(fs, noise, alt, interpolate) for alt in (0, 5, 10)]
            rows.append(row)
            print(f"{fs:>4} Hz  interpolated {row[1]:<3}  noise {noise:>2} uV | called positive: "
                  f"0 uV {row[3]:>2}/{N}  5 uV {row[4]:>2}/{N}  10 uV {row[5]:>2}/{N}", flush=True)
    header = "fs_hz,interpolated,noise_uv,positive_0uv,positive_5uv,positive_10uv,n_per_cell"
    Path(args.out).write_text("\n".join([header] + [",".join(map(str, r + [N])) for r in rows]) + "\n")
    return rows


if __name__ == "__main__":
    main()
