"""Generate every scientific figure used in README.md / README.ru.md.

All panels are *computed* from the project's own code (cardioonco/*) or from
standard textbook models; nothing is drawn by hand.

    python scripts/make_figures.py                 # both languages -> docs/figures/{en,ru}/
    python scripts/make_figures.py --checkpoint runs/ptbxl/model.pt   # use trained weights in fig. 08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardioonco.preprocess import bandpass, detect_r_peaks  # noqa: E402
from cardioonco.synth import LEADS_12, SynthConfig, generate_12lead, generate_ecg, heart_vector  # noqa: E402
from cardioonco.twa import analyze, beat_matrix  # noqa: E402

# ----------------------------------------------------------------------------- style
INK, MUTED, LINE = "#16202e", "#5b6573", "#d9d4d1"
RED, BLUE, GREEN, AMBER = "#c8102e", "#1f5aa6", "#1d7a4f", "#b07000"
PAPER, GRID_MIN, GRID_MAJ = "#fbf9f8", "#f3d9d9", "#e6b0b0"
DANGER = (200 / 255, 16 / 255, 46 / 255, 0.10)
CMAP_DIV = LinearSegmentedColormap.from_list("rb", [BLUE, "#ffffff", RED])
CMAP_SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#f3c4cb", RED, "#5a0714"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#ece8e6", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 160, "savefig.bbox": "tight",
})

L = {
    "en": dict(
        t_s="time, s", mv="mV", uv="µV", ms="ms", beat="beat number", fromR="time from R peak, s",
        # 01
        f1_title="From ion channels to the T wave", ap="membrane potential, mV", ph0="0: Na⁺ in (upstroke)",
        ph1="1: K⁺ out (I_to)", ph2="2: Ca²⁺ in ≈ K⁺ out (plateau)", ph3="3: K⁺ out (I_Kr, I_Ks) = repolarisation",
        ph4="4: rest (I_K1)", normal="normal", ikr="I_Kr reduced (drug / injury)", endo="endocardium (APD longer)",
        epi="epicardium (APD shorter)", pseudo="pseudo-ECG  ∝  V_endo − V_epi", twave="T wave = transmural\nrepolarisation gradient",
        qrs="QRS", a_title="a  Ventricular action potential", b_title="b  Two layers of the wall", c_title="c  What the electrode sees",
        # 02
        f2_title="Why the T wave alternates: restitution and period doubling",
        di="diastolic interval DI, ms", apd="APD, ms", apdn="APD_n, ms", apdn1="APD_{n+1}, ms", bcl="pacing cycle length (BCL), ms",
        rest="restitution curve APD = f(DI)", slope1="slope = 1", stable="slow pacing: converges", alt="fast pacing: 2-cycle (alternans)",
        a2="a  Restitution curve", b2="b  Cobweb of the map APD_{n+1} = f(BCL − APD_n)", c2="c  Bifurcation diagram",
        d2="d  Beat-to-beat APD", danger2="alternans zone", healthy="healthy", steep="steeper restitution (injured)",
        # 03
        f3_title="The heart as a dipole: Einthoven's triangle and the 12 leads",
        a3="a  Frontal-plane vector loop", b3="b  Leads are projections of one vector", c3="c  Einthoven's law holds exactly",
        loopq="QRS loop", loopt="T loop", xl3="x (patient's left), mV", yl3="y (inferior), mV", resid="II − (I + III)",
        # 04
        f4_title="Signal pipeline: raw electrode voltage → clean beats → R peaks",
        a4="a  Raw signal: wander + 50 Hz mains + muscle noise", b4="b  Power spectrum before and after the filter",
        c4="c  Zero-phase band-pass 0.5–40 Hz", d4="d  Pan–Tompkins: derivative² and 150 ms integration", e4="e  Detected R peaks",
        raw="raw", filt="filtered", freq="frequency, Hz", psd="power, mV²/Hz", mwi="integrated energy", thr="adaptive threshold",
        qrsband="QRS energy", wander="breathing drift", mains="mains 50 Hz",
        # 05
        f5_title="T-wave alternans: the Spectral Method step by step",
        a5="a  128 aligned beats minus the mean beat", b5="b  Even (A) vs odd (B) average beats", c5="c  One ST-T point across beats",
        d5="d  Aggregate spectrum", stt="ST-T danger zone", even="even beats A", odd="odd beats B", diffx="(A − B) × 10",
        cpb="cycles / beat", pow="power, µV²", noiseband="noise band", altpk="alternans\n0.5 cycles/beat",
        # 06
        f6_title="Detection map: when can alternans be measured?", alt_ax="true alternans amplitude, µV", noise_ax="muscle noise, µV (RMS)",
        kmap="K-score (log colour)", detected="TWA detected\nV_alt ≥ 1.9 µV and K ≥ 3", hidden="indeterminate:\nburied in noise",
        # 07
        f7_title="Continuous wavelet transform (complex Morlet): where the energy sits",
        a7="a  Normal beat", b7="b  Flattened T wave (repolarisation stress)", c7="c  Difference b − a: signal and |W|", fhz="frequency, Hz",
        # 08
        f8_title="What happens to one ECG inside CardioOncoNet", f8_sub_untrained="(random initial weights: tensor shapes are real, values are illustrative)",
        f8_sub_trained="(trained weights)", i8="a  Input: 12 leads × 1000 samples", s8="b  Stem conv: 32 channels × 500",
        k8="c  Block 3: 64 × 250", l8="d  Block 5: 128 × 63", ve8="e  v_e ∈ ℝ⁶⁴ (pooled ECG embedding)",
        vm8="f  v_m ∈ ℝ¹⁶ (age 58, female)", op8="g  v_e ⊗ v_m (64 × 16 block of 65 × 17)", out8="h  Output probabilities",
        chan="channel", samp="time step",
        # 09
        f9_title="INT8 quantisation for wearables: 4× smaller, almost the same numbers",
        a9="a  Weights of one conv layer: float32 vs INT8", b9="b  Quantisation error", c9="c  Model size",
        w="weight value", count="count", err="error, % of the step s", size="MB",
        # 10
        f10_title="Why early detection matters", a10="a  Heart failure vs cumulative doxorubicin dose (Swain et al., 2003)",
        b10="b  When is injury visible? (project hypothesis)", dose="cumulative dose, mg/m²", hf="patients with heart failure, %",
        mol="molecular\n(ROS, iron, TOP2B)", ele="electrical\n(QT, T-wave alternans)", mec="mechanical\n(LVEF ↓ on echo)",
        tgt="CardioOncoPredict\ntarget", std="standard\nmonitoring", time="time since start of chemotherapy (schematic)",
        visible="signal\nstrength",
    ),
    "ru": dict(
        t_s="время, с", mv="мВ", uv="мкВ", ms="мс", beat="номер удара", fromR="время от R-пика, с",
        f1_title="От ионных каналов к зубцу T", ap="мембранный потенциал, мВ", ph0="0: вход Na⁺ (деполяризация)",
        ph1="1: выход K⁺ (I_to)", ph2="2: вход Ca²⁺ ≈ выход K⁺ (плато)", ph3="3: выход K⁺ (I_Kr, I_Ks) = реполяризация",
        ph4="4: покой (I_K1)", normal="норма", ikr="I_Kr снижен (препарат / повреждение)", endo="эндокард (ПД длиннее)",
        epi="эпикард (ПД короче)", pseudo="псевдо-ЭКГ  ∝  V_эндо − V_эпи", twave="зубец T = трансмуральный\nградиент реполяризации",
        qrs="QRS", a_title="a  Потенциал действия желудочка", b_title="b  Два слоя стенки", c_title="c  Что видит электрод",
        f2_title="Почему зубец T чередуется: реституция и удвоение периода",
        di="диастолический интервал DI, мс", apd="ДПД, мс", apdn="ДПД_n, мс", apdn1="ДПД_{n+1}, мс", bcl="период стимуляции (BCL), мс",
        rest="кривая реституции ДПД = f(DI)", slope1="наклон = 1", stable="редкий ритм: сходится", alt="частый ритм: 2-цикл (альтернация)",
        a2="a  Кривая реституции", b2="b  Лестница Ламерея для ДПД_{n+1} = f(BCL − ДПД_n)", c2="c  Бифуркационная диаграмма",
        d2="d  ДПД от удара к удару", danger2="зона альтернации", healthy="здоровая ткань", steep="более крутая реституция (повреждение)",
        f3_title="Сердце как диполь: треугольник Эйнтховена и 12 отведений",
        a3="a  Векторная петля во фронтальной плоскости", b3="b  Отведения — проекции одного вектора", c3="c  Закон Эйнтховена выполняется точно",
        loopq="петля QRS", loopt="петля T", xl3="x (влево от пациента), мВ", yl3="y (вниз), мВ", resid="II − (I + III)",
        f4_title="Обработка сигнала: напряжение электрода → чистые удары → R-пики",
        a4="a  Сырой сигнал: дрейф + сеть 50 Гц + мышечный шум", b4="b  Спектр мощности до и после фильтра",
        c4="c  Фильтр 0,5–40 Гц без сдвига фазы", d4="d  Пан–Томпкинс: производная² и интегрирование 150 мс", e4="e  Найденные R-пики",
        raw="сырой", filt="после фильтра", freq="частота, Гц", psd="мощность, мВ²/Гц", mwi="интегр. энергия", thr="адаптивный порог",
        qrsband="энергия QRS", wander="дыхательный дрейф", mains="сеть 50 Гц",
        f5_title="Альтернация зубца T: спектральный метод шаг за шагом",
        a5="a  128 выровненных ударов минус средний удар", b5="b  Средние чётные (A) и нечётные (B) удары", c5="c  Одна точка ST-T по ударам",
        d5="d  Суммарный спектр", stt="опасная зона ST-T", even="чётные удары A", odd="нечётные удары B", diffx="(A − B) × 10",
        cpb="циклы / удар", pow="мощность, мкВ²", noiseband="полоса шума", altpk="альтернация\n0,5 цикла/удар",
        f6_title="Карта обнаружения: когда альтернацию можно измерить?", alt_ax="истинная амплитуда альтернации, мкВ", noise_ax="мышечный шум, мкВ (RMS)",
        kmap="K-score (лог. шкала)", detected="TWA обнаружена\nV_alt ≥ 1,9 мкВ и K ≥ 3", hidden="неопределённо:\nтонет в шуме",
        f7_title="Непрерывное вейвлет-преобразование (комплексный Морле): где сосредоточена энергия",
        a7="a  Нормальный удар", b7="b  Уплощённый зубец T (стресс реполяризации)", c7="c  Разность b − a: сигнал и |W|", fhz="частота, Гц",
        f8_title="Что происходит с одной ЭКГ внутри CardioOncoNet", f8_sub_untrained="(случайные начальные веса: размерности настоящие, значения иллюстративные)",
        f8_sub_trained="(обученные веса)", i8="a  Вход: 12 отведений × 1000 отсчётов", s8="b  Первая свёртка: 32 канала × 500",
        k8="c  Блок 3: 64 × 250", l8="d  Блок 5: 128 × 63", ve8="e  v_e ∈ ℝ⁶⁴ (вектор ЭКГ после пулинга)",
        vm8="f  v_m ∈ ℝ¹⁶ (58 лет, женщина)", op8="g  v_e ⊗ v_m (блок 64 × 16 из 65 × 17)", out8="h  Выходные вероятности",
        chan="канал", samp="шаг времени",
        f9_title="INT8-квантование для носимых устройств: в 4 раза меньше, числа почти те же",
        a9="a  Веса одного свёрточного слоя: float32 и INT8", b9="b  Ошибка квантования", c9="c  Размер модели",
        w="значение веса", count="количество", err="ошибка, % от шага s", size="МБ",
        f10_title="Почему важно раннее выявление", a10="a  Сердечная недостаточность и кумулятивная доза доксорубицина (Swain et al., 2003)",
        b10="b  Когда повреждение становится видно? (гипотеза проекта)", dose="кумулятивная доза, мг/м²", hf="пациентов с СН, %",
        mol="молекулярный\n(АФК, железо, TOP2B)", ele="электрический\n(QT, альтернация T)", mec="механический\n(ФВ ЛЖ ↓ на ЭхоКГ)",
        tgt="цель\nCardioOncoPredict", std="стандартный\nконтроль", time="время от начала химиотерапии (схема)",
        visible="сила\nсигнала",
    ),
}


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)
    print("  ", out / name)


def ecg_paper(ax, xlim, ylim, major_x=0.2, major_y=0.5):
    ax.set_facecolor(PAPER)
    ax.grid(False)
    for x in np.arange(xlim[0], xlim[1] + 1e-9, major_x / 5):
        ax.axvline(x, color=GRID_MIN, lw=0.4, zorder=0)
    for y in np.arange(np.floor(ylim[0] / 0.1) * 0.1, ylim[1] + 1e-9, major_y / 5):
        ax.axhline(y, color=GRID_MIN, lw=0.4, zorder=0)
    for x in np.arange(xlim[0], xlim[1] + 1e-9, major_x):
        ax.axvline(x, color=GRID_MAJ, lw=0.7, zorder=0)
    for y in np.arange(np.floor(ylim[0] / major_y) * major_y, ylim[1] + 1e-9, major_y):
        ax.axhline(y, color=GRID_MAJ, lw=0.7, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


# ============================================================================ 01
def action_potential(t, apd, v_rest=-85.0, v_peak=30.0, plateau=15.0, t0=0.0):
    """Phenomenological ventricular AP (phases 0-4) with a given duration APD (ms)."""
    s = t - t0
    up = 1 / (1 + np.exp(-s / 0.6))                                     # phase 0
    spike = (v_peak - plateau) * np.exp(-np.clip(s, 0, None) / 6.0) * up  # phase 1 notch
    repol = 1 / (1 + np.exp((s - apd) / (0.09 * apd)))                  # phase 3
    return v_rest + (plateau - v_rest) * up * repol + spike * repol


def fig01(lang, out):
    T = L[lang]
    t = np.linspace(-50, 500, 3000)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    v_n, v_k = action_potential(t, 250), action_potential(t, 310)
    ax[0].plot(t, v_n, color=INK, lw=2.2, label=T["normal"])
    ax[0].plot(t, v_k, color=RED, lw=2, ls="--", label=T["ikr"])
    ann = [(2, 10, T["ph0"], (40, -30)), (8, 24, T["ph1"], (22, 38)), (120, 15, T["ph2"], (150, 27)),
           (245, -35, T["ph3"].replace(" = ", "\n= "), (285, -12)), (420, -85, T["ph4"], (330, -70))]
    for x, y, txt, xy in ann:
        ax[0].annotate(txt, (x, y), xytext=xy, fontsize=8.5, color=MUTED,
                       arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax[0].annotate("", xy=(310, -60), xytext=(250, -60), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax[0].text(262, -56, "ΔAPD", color=RED, fontsize=9)
    ax[0].set(title=T["a_title"], xlabel=T["ms"], ylabel=T["ap"], ylim=(-95, 45))
    ax[0].legend(loc="lower left")

    v_endo = action_potential(t, 280, t0=10)
    v_epi = action_potential(t, 240, t0=18)
    ax[1].plot(t, v_endo, color=BLUE, lw=2, label=T["endo"])
    ax[1].plot(t, v_epi, color=RED, lw=2, label=T["epi"])
    ax[1].fill_between(t, v_endo, v_epi, where=(t > 150), color=DANGER, lw=0)
    ax[1].set(title=T["b_title"], xlabel=T["ms"], ylim=(-95, 45))
    ax[1].legend(loc="lower left")

    pseudo = v_endo - v_epi
    ecg_paper(ax[2], (-50, 500), (-60, 140), major_x=100, major_y=50)
    ax[2].plot(t, pseudo, color=INK, lw=2)
    ax[2].axvspan(150, 340, color=DANGER, lw=0)
    ax[2].annotate(T["twave"], (265, pseudo[np.argmin(np.abs(t - 265))]), xytext=(330, 95), fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax[2].text(20, 120, T["qrs"], fontsize=9, fontweight="bold")
    ax[2].set(title=T["c_title"], xlabel=T["ms"])
    ax[2].text(-40, -52, T["pseudo"], fontsize=8.5, color=MUTED)
    fig.suptitle(T["f1_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
    save(fig, out, "01_electrophysiology.png")


# ============================================================================ 02
def restitution(di, apd_max=260.0, a=200.0, tau=55.0):
    return apd_max - a * np.exp(-np.clip(di, 1, None) / tau)


def iterate(bcl, n=300, apd0=200.0, **kw):
    apd = [apd0]
    for _ in range(n):
        di = bcl - apd[-1]
        apd.append(restitution(max(di, 1.0), **kw))
    return np.array(apd)


def fig02(lang, out):
    T = L[lang]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))
    di = np.linspace(5, 400, 500)
    ax[0, 0].plot(di, restitution(di), color=INK, lw=2.2, label=T["healthy"])
    ax[0, 0].plot(di, restitution(di, a=260, tau=45), color=RED, lw=2, ls="--", label=T["steep"])
    # slope = 1 point
    slope = 200 / 55 * np.exp(-di / 55)
    d1 = di[np.argmin(np.abs(slope - 1))]
    ax[0, 0].axvspan(0, d1, color=DANGER, lw=0)
    ax[0, 0].text(d1 * 0.5, 80, T["danger2"], ha="center", color=RED, fontsize=9)
    x = np.linspace(d1 - 40, d1 + 40, 10)
    ax[0, 0].plot(x, restitution(d1) + (x - d1), color=MUTED, lw=1, ls=":", label=T["slope1"])
    ax[0, 0].set(title=T["a2"], xlabel=T["di"], ylabel=T["apd"], xlim=(0, 400), ylim=(40, 280))
    ax[0, 0].legend(loc="lower right")

    a = np.linspace(60, 300, 400)
    ax[0, 1].plot(a, a, color=LINE, lw=1)
    for bcl, col, lab in [(420, BLUE, T["stable"]), (270, RED, T["alt"])]:
        ax[0, 1].plot(a, restitution(np.clip(bcl - a, 1, None)), color=col, lw=2, label=f"BCL {bcl} {T['ms']}: {lab}")
        seq = iterate(bcl, n=40, apd0=150)
        xs, ys = [seq[0]], [seq[0]]
        for k in range(len(seq) - 1):
            xs += [seq[k], seq[k + 1]]
            ys += [seq[k + 1], seq[k + 1]]
        ax[0, 1].plot(xs, ys, color=col, lw=0.8, alpha=0.7)
    ax[0, 1].set(title=T["b2"], xlabel=T["apdn"], ylabel=T["apdn1"], xlim=(60, 300), ylim=(60, 300))
    ax[0, 1].legend(loc="upper left")

    bcls = np.linspace(258, 480, 500)
    for b in bcls:
        seq = iterate(b, n=400)[-40:]
        ax[1, 0].plot(np.full(len(seq), b), seq, ",", color=INK, alpha=0.6, ms=1)
        ax[1, 0].plot([b, b], [seq.min(), seq.max()], color=INK, lw=0.4, alpha=0.4)
    alt_b = [b for b in bcls if np.ptp(iterate(b, n=400)[-20:]) > 1]
    if alt_b:
        ax[1, 0].axvspan(min(alt_b), max(alt_b), color=DANGER, lw=0)
        ax[1, 0].text(np.mean(alt_b), 262, T["danger2"], ha="center", color=RED, fontsize=9)
    ax[1, 0].set(title=T["c2"], xlabel=T["bcl"], ylabel=T["apd"], ylim=(60, 275))
    ax[1, 0].invert_xaxis()

    for bcl, col in [(420, BLUE), (270, RED)]:
        seq = iterate(bcl, n=30, apd0=150)
        ax[1, 1].plot(np.arange(len(seq)), seq, "o-", color=col, ms=4, lw=1.2, label=f"BCL {bcl} {T['ms']}")
    ax[1, 1].set(title=T["d2"], xlabel=T["beat"], ylabel=T["apd"])
    ax[1, 1].legend()
    fig.suptitle(T["f2_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, out, "02_restitution_alternans.png")


# ============================================================================ 03
def fig03(lang, out):
    T = L[lang]
    fs = 1000
    t = np.arange(0, 0.8, 1 / fs)
    H = heart_vector(t, 0.8, 0.35, 0.05)
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.05, 1.2, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    qrs, tw = (t > 0.27) & (t < 0.40), (t > 0.42) & (t < 0.72)
    ax0.plot(H[qrs, 0], H[qrs, 1], color=INK, lw=2, label=T["loopq"])
    ax0.plot(H[tw, 0], H[tw, 1], color=RED, lw=2, label=T["loopt"])
    r = 1.6
    tri = np.array([[-r, -0.55 * r], [r, -0.55 * r], [0, 1.15 * r]])
    ax0.add_patch(Polygon(tri, closed=True, fill=False, ec=MUTED, lw=1, ls="--"))
    for ang, name in [(0, "I"), (60, "II"), (120, "III"), (90, "aVF"), (-30, "aVL"), (-150, "aVR")]:
        v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * 1.45
        ax0.add_patch(FancyArrowPatch((0, 0), tuple(v), arrowstyle="-|>", mutation_scale=10, color=BLUE, lw=1))
        ax0.text(*(v * 1.1), name, color=BLUE, ha="center", va="center", fontsize=9, fontweight="bold")
    ax0.set(title=T["a3"], xlim=(-1.9, 1.9), ylim=(1.9, -1.2), aspect="equal", xlabel=T["xl3"], ylabel=T["yl3"])
    ax0.legend(loc="lower left")

    from cardioonco.synth import dipole_to_12lead
    X = dipole_to_12lead(H)
    for i, (idx, name) in enumerate([(0, "I"), (1, "II"), (2, "III")]):
        a = fig.add_subplot(gs[i, 1])
        ecg_paper(a, (0.1, 0.8), (-0.6, 1.6), major_x=0.2, major_y=0.5)
        a.plot(t, X[:, idx], color=INK, lw=1.6)
        a.text(0.12, 1.25, name, fontweight="bold")
        a.set_xticklabels([]) if i < 2 else a.set_xlabel(T["t_s"])
        if i == 0:
            a.set_title(T["b3"])
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.plot(t, X[:, 1], color=INK, lw=2, label="II")
    ax2.plot(t, X[:, 0] + X[:, 2], color=RED, lw=1.2, ls="--", label="I + III")
    ax2.plot(t, (X[:, 1] - X[:, 0] - X[:, 2]) * 1e12, color=GREEN, lw=1, label=T["resid"] + " × 10¹²")
    ax2.set(title=T["c3"], xlabel=T["t_s"], ylabel=T["mv"], xlim=(0.1, 0.8))
    ax2.legend(loc="upper right")
    fig.suptitle(T["f3_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, out, "03_dipole_einthoven.png")


# ============================================================================ 04
def fig04(lang, out):
    from scipy.signal import welch
    T = L[lang]
    fs = 500
    t, x, _ = generate_ecg(SynthConfig(fs=fs, n_beats=40, heart_rate=72, noise_uv=45, wander_mv=0.35, mains_uv=120, seed=4))
    xf = bandpass(x, fs)
    r = detect_r_peaks(x, fs)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.45)
    win = (t >= 2) & (t < 7)
    a = fig.add_subplot(gs[0, 0])
    ecg_paper(a, (2, 7), (-0.8, 1.8), major_y=0.5)
    a.plot(t[win], x[win], color=INK, lw=0.9)
    a.set(title=T["a4"], ylabel=T["mv"])
    b = fig.add_subplot(gs[0:2, 1])
    f, p = welch(x, fs, nperseg=4096)
    f2, p2 = welch(xf, fs, nperseg=4096)
    b.semilogy(f, p, color=MUTED, lw=1.2, label=T["raw"])
    b.semilogy(f2, p2, color=RED, lw=1.5, label=T["filt"])
    b.axvspan(0.5, 40, color=(29 / 255, 122 / 255, 79 / 255, 0.08), lw=0)
    b.axvspan(5, 25, color=DANGER, lw=0)
    b.text(12, p.max() * 0.3, T["qrsband"], color=RED, fontsize=9)
    b.annotate(T["wander"], (0.25, p[np.argmin(np.abs(f - 0.25))]), xytext=(3, p.max() * 3), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    b.annotate(T["mains"], (50, p[np.argmin(np.abs(f - 50))]), xytext=(60, p.max() * 0.5), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    b.set(title=T["b4"], xlabel=T["freq"], ylabel=T["psd"], xscale="log", xlim=(0.1, 250), ylim=(p2.max() * 1e-9, p.max() * 20))
    b.legend(loc="lower left")
    c = fig.add_subplot(gs[1, 0])
    ecg_paper(c, (2, 7), (-0.8, 1.8), major_y=0.5)
    c.plot(t[win], xf[win], color=INK, lw=1.1)
    c.set(title=T["c4"], ylabel=T["mv"])
    # Pan-Tompkins internals
    from cardioonco.preprocess import bandpass as bp
    q = bp(x, fs, 5, 15, order=2)
    d = np.zeros_like(q)
    d[1:-1] = q[2:] - q[:-2]
    e = d ** 2
    w = int(0.15 * fs)
    mwi = np.convolve(e, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(mwi, 99)
    dd = fig.add_subplot(gs[2, 0])
    dd.plot(t[win], e[win] / e.max(), color=MUTED, lw=0.8, label="d[n]²")
    dd.plot(t[win], mwi[win] / mwi.max(), color=BLUE, lw=1.8, label=T["mwi"])
    dd.axhline(thr / mwi.max(), color=RED, ls="--", lw=1.2, label=T["thr"])
    dd.set(title=T["d4"], xlabel=T["t_s"], xlim=(2, 7))
    dd.legend(loc="upper right", ncol=3)
    ee = fig.add_subplot(gs[2, 1])
    beats = beat_matrix(xf, r, fs, -0.25, 0.55)
    tb = np.arange(beats.shape[1]) / fs - 0.25
    for row in beats:
        ee.plot(tb, row, color=INK, lw=0.5, alpha=0.25)
    ee.plot(tb, beats.mean(0), color=RED, lw=2)
    ee.axvline(0, color=RED, lw=0.8, ls=":")
    ee.set(title=f"{T['e4']}: {len(r)}", xlabel=T["fromR"], ylabel=T["mv"])
    fig.suptitle(T["f4_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.96)
    save(fig, out, "04_signal_pipeline.png")


# ============================================================================ 05
def fig05(lang, out):
    T = L[lang]
    fs = 500
    _, x, _ = generate_ecg(SynthConfig(fs=fs, alternans_uv=15, noise_uv=8, heart_rate=90, hrv_std=0.004, t_amp=0.28, seed=11))
    res = analyze(x, fs)
    r = detect_r_peaks(x, fs)
    xf = bandpass(x, fs)
    B = beat_matrix(xf, r, fs, -0.25, 0.55)[:128]
    tb = np.arange(B.shape[1]) / fs - 0.25
    rr = 60 / res.heart_rate_bpm
    sc = np.sqrt(rr / 0.8)
    w0, w1 = 0.10 * sc, 0.42 * sc
    dev = (B - B.mean(0)) * 1000
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.18)
    a = fig.add_subplot(gs[0, 0])
    lim = np.percentile(np.abs(dev), 99)
    im = a.imshow(dev, aspect="auto", cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -lim, lim),
                  extent=[tb[0], tb[-1], len(B), 0], interpolation="nearest")
    a.axvline(w0, color=INK, lw=1, ls="--")
    a.axvline(w1, color=INK, lw=1, ls="--")
    a.text((w0 + w1) / 2, 8, T["stt"], ha="center", color=RED, fontsize=9, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=RED, lw=0.8))
    a.set(title=T["a5"], xlabel=T["fromR"], ylabel=T["beat"])
    a.grid(False)
    cb = fig.colorbar(im, ax=a, pad=0.01)
    cb.set_label(T["uv"])
    b = fig.add_subplot(gs[0, 1])
    A, Bo = B[0::2].mean(0), B[1::2].mean(0)
    b.axvspan(w0, w1, color=DANGER, lw=0, label=T["stt"])
    b.plot(tb, A, color=BLUE, lw=2, label=T["even"])
    b.plot(tb, Bo, color=RED, lw=2, label=T["odd"])
    b.plot(tb, (A - Bo) * 10, color=MUTED, lw=1.2, ls="--", label=T["diffx"])
    b.set(title=T["b5"], xlabel=T["fromR"], ylabel=T["mv"])
    b.legend(loc="upper right")
    c = fig.add_subplot(gs[1, 0])
    k0 = np.searchsorted(tb, w0)
    k1 = np.searchsorted(tb, w1)
    alt_strength = np.abs(((B[:, k0:k1] - B[:, k0:k1].mean(0)) * ((-1.0) ** np.arange(len(B)))[:, None]).sum(0))
    kbest = k0 + int(np.argmax(alt_strength))
    s = (B[:, kbest] - B[:, kbest].mean()) * 1000
    c.bar(np.arange(len(s)), s, color=[BLUE if i % 2 == 0 else RED for i in range(len(s))], width=0.8)
    c.axhline(0, color=INK, lw=0.6)
    c.set(title=f"{T['c5']} (t = {tb[kbest]:.3f} {T['t_s'].split(', ')[-1]})", xlabel=T["beat"], ylabel=T["uv"], xlim=(-1, len(s)))
    d = fig.add_subplot(gs[1, 1])
    f, P = np.array(res.spectrum_f), np.array(res.spectrum_p_uv2)
    d.axvspan(0.44, 0.49, color=(31 / 255, 90 / 255, 166 / 255, 0.12), lw=0, label=T["noiseband"])
    d.semilogy(f[1:], P[1:], color=INK, lw=1.3)
    d.plot(0.5, P[-1], "o", color=RED, ms=9)
    d.annotate(T["altpk"], (0.5, P[-1]), xytext=(0.33, P[-1] * 0.6), color=RED, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=RED))
    d.text(0.02, 0.95, f"V_alt = {res.v_alt_uv:.1f} µV   K = {res.k_score:.0f}\nMMA = {res.mma_uv:.1f} µV   HR = {res.heart_rate_bpm:.0f}",
           transform=d.transAxes, va="top", family="DejaVu Sans Mono", fontsize=9.5,
           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=LINE))
    d.set(title=T["d5"], xlabel=T["cpb"], ylabel=T["pow"], xlim=(0, 0.51))
    d.legend(loc="lower left")
    fig.suptitle(T["f5_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.97)
    save(fig, out, "05_twa_spectral_method.png")


# ============================================================================ 06
def detection_grid():
    alts = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
    noises = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    K = np.zeros((len(noises), len(alts)))
    V = np.zeros_like(K)
    for i, nz in enumerate(noises):
        for j, al in enumerate(alts):
            ks, vs = [], []
            for seed in (1, 2, 3, 4):
                _, x, _ = generate_ecg(SynthConfig(alternans_uv=al, noise_uv=nz, seed=seed + 10 * i + 100 * j))
                r = analyze(x, 500)
                ks.append(r.k_score)
                vs.append(r.v_alt_uv)
            K[i, j], V[i, j] = np.median(ks), np.median(vs)
    return alts, noises, K, V


_GRID = None


def fig06(lang, out):
    global _GRID
    T = L[lang]
    if _GRID is None:
        print("   computing detection grid (280 full analyses)...")
        _GRID = detection_grid()
    alts, noises, K, V = _GRID
    fig, ax = plt.subplots(figsize=(10, 6))
    Kc = np.clip(K, 0.1, None)
    im = ax.pcolormesh(alts, noises, np.log10(Kc), cmap=CMAP_SEQ, shading="gouraud", vmin=-1, vmax=3.3)
    from scipy.ndimage import gaussian_filter
    Ks, Vs = 10 ** gaussian_filter(np.log10(Kc), 0.7), gaussian_filter(V, 0.7)
    with plt.rc_context({"hatch.color": (1, 1, 1, 0.45), "hatch.linewidth": 0.6}):
        ax.contourf(alts, noises, ((Ks >= 3) & (Vs >= 1.9)).astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
    cs = ax.contour(alts, noises, Ks, levels=[3], colors=[INK], linewidths=2.2)
    ax.clabel(cs, fmt={3: "K = 3"}, fontsize=9)
    cv = ax.contour(alts, noises, Vs, levels=[1.9], colors=[BLUE], linewidths=1.6, linestyles="--")
    ax.clabel(cv, fmt={1.9: "V_alt = 1.9"}, fontsize=9)
    ax.text(21, 18, T["detected"], color="white", fontsize=11, fontweight="bold", ha="center")
    ax.text(2.2, 88, T["hidden"], color=INK, fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LINE))
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(T["kmap"])
    cb.set_ticks([-1, 0, 1, 2, 3])
    cb.set_ticklabels(["0.1", "1", "10", "100", "1000"])
    ax.set(xlabel=T["alt_ax"], ylabel=T["noise_ax"], title=T["f6_title"])
    ax.grid(False)
    save(fig, out, "06_detection_map.png")


# ============================================================================ 07
def fig07(lang, out):
    import pywt
    T = L[lang]
    fs = 500
    _, xa, ra = generate_ecg(SynthConfig(fs=fs, n_beats=3, noise_uv=3, wander_mv=0, t_amp=0.35, seed=1))
    _, xb, rb = generate_ecg(SynthConfig(fs=fs, n_beats=3, noise_uv=3, wander_mv=0, t_amp=0.10, t_width=0.08, seed=1))
    seg = slice(ra[1] - int(0.35 * fs), ra[1] + int(0.6 * fs))
    xa, xb = xa[seg], xb[seg]
    t = np.arange(len(xa)) / fs - 0.35
    freqs = np.geomspace(0.8, 60, 80)
    wav = "cmor1.5-1.0"
    scales = pywt.central_frequency(wav) * fs / freqs
    Wa = np.abs(pywt.cwt(xa, scales, wav, sampling_period=1 / fs)[0])
    Wb = np.abs(pywt.cwt(xb, scales, wav, sampling_period=1 / fs)[0])
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5), gridspec_kw={"height_ratios": [1, 2.2]}, sharex=True)
    vmax = max(Wa.max(), Wb.max())
    for j, (x, W, ttl) in enumerate([(xa, Wa, T["a7"]), (xb, Wb, T["b7"])]):
        ecg_paper(ax[0, j], (t[0], t[-1]), (-0.4, 1.4), major_y=0.5)
        ax[0, j].plot(t, x, color=INK, lw=1.4)
        ax[0, j].set_title(ttl)
        ax[1, j].pcolormesh(t, freqs, W, cmap=CMAP_SEQ, vmin=0, vmax=vmax, shading="auto")
        ax[1, j].set(yscale="log", ylabel=T["fhz"] if j == 0 else None, xlabel=T["fromR"])
        ax[1, j].grid(False)
    ax[0, 2].plot(t, xb - xa, color=RED, lw=1.4)
    ax[0, 2].set_title(T["c7"])
    dW = Wb - Wa
    lim = np.abs(dW).max()
    ax[1, 2].pcolormesh(t, freqs, dW, cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
    ax[1, 2].set(yscale="log", xlabel=T["fromR"])
    ax[1, 2].grid(False)
    for a in ax[1]:
        a.axvspan(0.1, 0.42, color=(1, 1, 1, 0), ec=INK, ls="--", lw=1)
    fig.suptitle(T["f7_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, out, "07_cwt_scalogram.png")


# ============================================================================ 08
def fig08(lang, out, checkpoint=None):
    import torch
    from cardioonco.model import CardioOncoNet
    T = L[lang]
    torch.manual_seed(0)
    net = CardioOncoNet()
    sub = T["f8_sub_untrained"]
    mu = sd = None
    if checkpoint and Path(checkpoint).exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        net = CardioOncoNet(n_classes=len(ck["classes"]), width=ck["width"])
        net.load_state_dict(ck["state_dict"])
        mu, sd = ck["mu"], ck["sd"]
        sub = T["f8_sub_trained"]
    net.eval()
    _, X, _ = generate_12lead(SynthConfig(fs=100, n_beats=14, heart_rate=78, noise_uv=15, t_amp=0.2, seed=5))
    X = X[:1000].T.astype(np.float32)
    Xn = (X - mu[0]) / sd[0] if mu is not None else X / X.std()
    acts = {}
    enc = net.ecg_encoder
    hooks = [enc.stem.register_forward_hook(lambda m, i, o: acts.__setitem__("stem", o)),
             enc.blocks[2].register_forward_hook(lambda m, i, o: acts.__setitem__("b3", o)),
             enc.blocks[4].register_forward_hook(lambda m, i, o: acts.__setitem__("b5", o))]
    meta = torch.tensor([[(58 - 62) / 17, 1.0, 0.0]])
    with torch.no_grad():
        xe = torch.from_numpy(Xn.astype(np.float32))[None]
        ve = net.ecg_encoder(xe)
        vm = net.meta_encoder(meta)
        e1 = torch.cat([ve, torch.ones(1, 1)], 1)
        m1 = torch.cat([vm, torch.ones(1, 1)], 1)
        outer = (e1.T @ m1).numpy()
        probs = torch.sigmoid(net(xe, meta))[0].numpy()
    for h in hooks:
        h.remove()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1], hspace=0.5, wspace=0.42)
    a = fig.add_subplot(gs[0, :2])
    tt = np.arange(1000) / 100
    for i in range(12):
        a.plot(tt, X[i] - i * 1.6, color=INK, lw=0.7)
        a.text(-0.35, -i * 1.6, LEADS_12[i], fontsize=7.5, ha="right", va="center", color=MUTED)
    a.set(title=T["i8"], xlabel=T["t_s"], yticks=[], xlim=(0, 10))
    a.grid(False)
    for key, pos, ttl in [("stem", gs[0, 2:], T["s8"]), ("b3", gs[1, 0:2], T["k8"]), ("b5", gs[1, 2:], T["l8"])]:
        ax = fig.add_subplot(pos)
        A = acts[key][0].numpy()
        ax.imshow(A, aspect="auto", cmap=CMAP_SEQ, interpolation="nearest")
        ax.set(title=ttl, xlabel=T["samp"], ylabel=T["chan"])
        ax.grid(False)
    b = fig.add_subplot(gs[2, 0])
    b.bar(np.arange(64), ve[0].numpy(), color=BLUE, width=0.8)
    b.set(title=T["ve8"], xlim=(-1, 64))
    c = fig.add_subplot(gs[2, 1])
    c.bar(np.arange(16), vm[0].numpy(), color=AMBER, width=0.8)
    c.set(title=T["vm8"])
    d = fig.add_subplot(gs[2, 2])
    d.imshow(outer[:-1, :-1], aspect="auto", cmap=CMAP_SEQ, interpolation="nearest",
             vmax=np.percentile(outer[:-1, :-1], 99.5) + 1e-9)
    d.set(title=T["op8"], xlabel="v_m", ylabel="v_e")
    d.grid(False)
    e = fig.add_subplot(gs[2, 3])
    cls = ["NORM", "MI", "STTC", "CD", "HYP"]
    e.barh(cls[::-1], probs[::-1], color=[RED if c == "STTC" else MUTED for c in cls[::-1]])
    e.set(title=T["out8"], xlim=(0, 1))
    for i, p in enumerate(probs[::-1]):
        e.text(p + 0.02, i, f"{p:.2f}", va="center", fontsize=9)
    fig.suptitle(f"{T['f8_title']}  {sub}", x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.95)
    save(fig, out, "08_network_dataflow.png")


# ============================================================================ 09
def fig09(lang, out):
    import torch
    from cardioonco.model import CardioOncoNet, count_parameters
    T = L[lang]
    torch.manual_seed(0)
    net = CardioOncoNet()
    w = net.ecg_encoder.blocks[3].body[0].weight.detach().numpy().ravel()
    lo, hi = w.min(), w.max()
    s = (hi - lo) / 255
    z = np.round(-lo / s)
    q = np.clip(np.round(w / s) + z, 0, 255)
    wq = (q - z) * s
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 0.8]})
    ax[0].hist(w, bins=120, color=MUTED, alpha=0.6, label="float32")
    ax[0].hist(wq, bins=120, color=RED, histtype="step", lw=1.3, label="INT8 → float")
    ax[0].set(title=T["a9"], xlabel=T["w"], ylabel=T["count"])
    ax[0].legend()
    ax[0].text(0.02, 0.95, "q  = round(x / s) + z\nx' = (q − z)·s\ns  = (max − min) / 255", transform=ax[0].transAxes,
               va="top", family="DejaVu Sans Mono", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=LINE))
    ax[1].hist((w - wq) / s * 100, bins=60, color=BLUE, alpha=0.8)
    ax[1].set(title=T["b9"], xlabel=T["err"])
    n = count_parameters(net)
    sizes = [n * 4 / 1e6, n / 1e6]
    ax[2].bar(["float32", "INT8"], sizes, color=[MUTED, RED], width=0.6)
    for i, v in enumerate(sizes):
        ax[2].text(i, v + 0.05, f"{v:.2f} {T['size']}", ha="center", fontsize=10, fontweight="bold")
    ax[2].set(title=T["c9"], ylabel=T["size"], ylim=(0, sizes[0] * 1.25))
    fig.suptitle(T["f9_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, out, "09_int8_quantization.png")


# ============================================================================ 10
def fig10(lang, out):
    T = L[lang]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    d = np.array([400, 550, 700])
    p = np.array([5, 26, 48])
    ax[0].bar(d, p, width=70, color=RED, alpha=0.9)
    for x, y in zip(d, p):
        ax[0].text(x, y + 1.5, f"{y}%", ha="center", fontweight="bold")
    ax[0].set(title=T["a10"], xlabel=T["dose"], ylabel=T["hf"], xlim=(300, 780), ylim=(0, 58))
    tt = np.linspace(0, 10, 500)
    sig = lambda c, w: 1 / (1 + np.exp(-(tt - c) / w))  # noqa: E731
    ax[1].plot(tt, sig(1.5, 0.35), color=MUTED, lw=2, label=T["mol"])
    ax[1].plot(tt, sig(3.5, 0.5), color=RED, lw=2.6, label=T["ele"])
    ax[1].plot(tt, sig(7.0, 0.6), color=BLUE, lw=2, label=T["mec"])
    ax[1].axvspan(3.0, 6.0, color=DANGER, lw=0)
    ax[1].text(4.5, 1.08, T["tgt"], ha="center", color=RED, fontsize=9, fontweight="bold")
    ax[1].text(7.6, 1.08, T["std"], ha="center", color=BLUE, fontsize=9)
    ax[1].set(title=T["b10"], xlabel=T["time"], ylabel=T["visible"], ylim=(-0.05, 1.25), xticks=[], yticks=[])
    ax[1].legend(loc="center right", fontsize=8.5)
    fig.suptitle(T["f10_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, out, "10_clinical_motivation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["en", "ru"])
    ap.add_argument("--only", nargs="*", default=None, help="e.g. 05 06")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    figs = {"01": fig01, "02": fig02, "03": fig03, "04": fig04, "05": fig05, "06": fig06,
            "07": fig07, "08": fig08, "09": fig09, "10": fig10}
    for lang in args.lang:
        out = ROOT / "docs" / "figures" / lang
        print(f"[{lang}]")
        for k, fn in figs.items():
            if args.only and k not in args.only:
                continue
            fn(lang, out, args.checkpoint) if k == "08" else fn(lang, out)


if __name__ == "__main__":
    main()
