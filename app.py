"""Gradio lab: real T-wave-alternans mathematics on a controllable synthetic ECG.

Every number shown is *computed* by cardioonco.twa from the generated signal
(R-peak detection -> beat alignment -> spectral method / MMA).  Nothing is hard-coded.
The signal itself is synthetic, so this is a physics/maths laboratory, not a diagnosis.
"""
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cardioonco.preprocess import bandpass, detect_r_peaks  # noqa: E402
from cardioonco.synth import SynthConfig, generate_ecg  # noqa: E402
from cardioonco.twa import analyze, beat_matrix  # noqa: E402

FS = 500


def run_lab(alternans_uv, noise_uv, heart_rate, t_amp_mv, seed=7):
    cfg = SynthConfig(fs=FS, heart_rate=heart_rate, alternans_uv=alternans_uv,
                      noise_uv=noise_uv, t_amp=t_amp_mv, mains_uv=20, seed=int(seed))
    t, x, _ = generate_ecg(cfg)
    res = analyze(x, FS)
    r = detect_r_peaks(x, FS)
    xf = bandpass(x, FS)
    beats = beat_matrix(xf, r, FS, -0.25, 0.55)[:128]

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(t[: 5 * FS], x[: 5 * FS], lw=1, color="#0b3d5c")
    ax[0].plot(r[r < 5 * FS] / FS, x[r[r < 5 * FS]], "o", color="#e63946", ms=5, label="detected R peaks")
    ax[0].set(title="1. Raw synthetic ECG (first 5 s) + Pan-Tompkins R-peak detection", xlabel="s", ylabel="mV")
    ax[0].legend(loc="upper right")
    tb = np.arange(beats.shape[1]) / FS - 0.25
    ax[1].plot(tb, beats[0::2].mean(0), color="#1f77b4", lw=2, label="mean of EVEN beats (A)")
    ax[1].plot(tb, beats[1::2].mean(0), color="#e63946", lw=2, label="mean of ODD beats (B)")
    ax[1].set(title="2. Beats aligned on R: A-B-A-B alternation of the T wave", xlabel="time from R (s)", ylabel="mV")
    ax[1].legend(loc="upper right")
    f, P = np.array(res.spectrum_f), np.array(res.spectrum_p_uv2)
    ax[2].semilogy(f[1:], P[1:], color="#0b3d5c")
    ax[2].axvspan(0.44, 0.49, color="#999", alpha=0.25, label="noise band")
    ax[2].axvline(0.5, color="#e63946", ls="--", label="alternans frequency 0.5 cycles/beat")
    ax[2].set(title="3. Aggregate beat-series spectrum (Spectral Method)", xlabel="cycles / beat", ylabel="uV^2")
    ax[2].legend(loc="upper left")
    fig.tight_layout()

    verdict = "TWA POSITIVE (V_alt >= 1.9 uV and K >= 3)" if res.positive else "TWA negative / indeterminate"
    text = (f"{verdict}\n"
            f"heart rate      {res.heart_rate_bpm:6.1f} bpm   (from {res.n_beats} beats)\n"
            f"V_alt (RMS)     {res.v_alt_uv:6.2f} uV\n"
            f"V_alt (peak)    {res.v_alt_peak_uv:6.2f} uV   (injected: {alternans_uv:.1f} uV)\n"
            f"K-score         {res.k_score:6.1f}\n"
            f"MMA             {res.mma_uv:6.2f} uV\n"
            f"noise floor     {res.noise_uv:6.2f} uV\n\n"
            "Synthetic signal - educational laboratory, not a medical device.")
    return text, fig


demo = gr.Interface(
    fn=run_lab,
    inputs=[
        gr.Slider(0, 60, value=20, step=0.5, label="Injected T-wave alternans, uV"),
        gr.Slider(0, 80, value=15, step=1, label="Muscle (white) noise, uV"),
        gr.Slider(45, 130, value=75, step=1, label="Heart rate, bpm"),
        gr.Slider(0.05, 0.5, value=0.35, step=0.01, label="T-wave amplitude, mV (flattening = repolarisation stress)"),
    ],
    outputs=[gr.Textbox(label="Computed TWA metrics", lines=9), gr.Plot(label="Signal processing")],
    title="CardioOncoPredict - T-wave alternans lab",
    description=("Real signal-processing mathematics (Pan-Tompkins, Spectral Method, MMA) running on a "
                 "controllable synthetic ECG. Research/education prototype - not a medical device."),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
