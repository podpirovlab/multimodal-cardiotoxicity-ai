"""Analyse one ECG: TWA mathematics + (optionally) the trained PTB-XL model + FHIR report.

Examples
--------
    python predict.py --demo --alternans 25                   # synthetic Holter strip
    python predict.py --wfdb data/ptb-xl/records100/00000/00001_lr --age 56 --sex female \
                      --checkpoint runs/ptbxl/model.pt
    python predict.py --csv my_ecg.csv --fs 500 --lead 0      # one column per lead, mV

TWA needs a long recording (>= 64 beats, ideally 128, i.e. about 2 minutes); a 10-second
12-lead ECG has only ~12 beats, so for those only the neural-network output is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from cardioonco import fhir
from cardioonco.twa import analyze


def load_input(args):
    if args.demo:
        from cardioonco.synth import SynthConfig, generate_ecg
        _, x, _ = generate_ecg(SynthConfig(alternans_uv=args.alternans, noise_uv=15, seed=7))
        return x[:, None], 500.0
    if args.wfdb:
        import wfdb
        sig, fields = wfdb.rdsamp(args.wfdb)
        return np.nan_to_num(sig), float(fields["fs"])
    if args.csv:
        data = np.loadtxt(args.csv, delimiter=",", skiprows=1 if args.header else 0, ndmin=2)
        return data, float(args.fs)
    raise SystemExit("give --demo, --wfdb PATH or --csv PATH")


def run_model(sig, fs, age, sex, ckpt_path):
    import torch
    from cardioonco.model import CardioOncoNet
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if sig.shape[1] != 12:
        return None, "model skipped: it needs a 12-lead ECG"
    x = resample_poly(sig, 100, int(round(fs)), axis=0) if int(round(fs)) != 100 else sig
    x = x[:1000]
    if len(x) < 1000:
        x = np.pad(x, ((0, 1000 - len(x)), (0, 0)))
    x = ((x.T[None] - ck["mu"]) / ck["sd"]).astype(np.float32)
    a = 90.0 if age > 120 else age
    meta = np.array([[(a - 62.0) / 17.0, 1.0 if sex == "female" else 0.0, 0.0]], dtype=np.float32)
    net = CardioOncoNet(n_classes=len(ck["classes"]), width=ck["width"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(x), torch.from_numpy(meta)))[0].numpy()
    return {c: float(v) for c, v in zip(ck["classes"], p)}, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--alternans", type=float, default=20.0, help="demo only: injected TWA, uV")
    ap.add_argument("--wfdb")
    ap.add_argument("--csv")
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--lead", type=int, default=None, help="lead index for TWA (default: V5 if 12-lead)")
    ap.add_argument("--age", type=float, default=60.0)
    ap.add_argument("--sex", choices=["male", "female"], default="male")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fhir-out", default=None)
    args = ap.parse_args(argv)

    sig, fs = load_input(args)
    lead = args.lead if args.lead is not None else (10 if sig.shape[1] == 12 else 0)
    report = {"fs": fs, "n_samples": int(sig.shape[0]), "n_leads": int(sig.shape[1]), "twa_lead": lead}

    try:
        twa = analyze(sig[:, lead], fs).as_dict()
        report["twa"] = {k: v for k, v in twa.items() if not k.startswith("spectrum")}
    except ValueError as exc:
        twa = None
        report["twa"] = f"not computed: {exc}"

    probs = None
    if args.checkpoint and Path(args.checkpoint).exists():
        probs, status = run_model(sig, fs, args.age, args.sex, args.checkpoint)
        report["model"] = probs if probs else status

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fhir_out:
        doc = fhir.diagnostic_report("Patient/anonymous", twa, probs)
        Path(args.fhir_out).write_text(json.dumps(doc, indent=2))
        print(f"FHIR DiagnosticReport -> {args.fhir_out}")
    return report


if __name__ == "__main__":
    main()
