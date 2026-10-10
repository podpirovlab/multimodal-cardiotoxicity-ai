"""Generate every scientific figure used in README.md / README.ru.md.

All panels are *computed* from the project's own code (cardioonco/*) or from
standard textbook models; nothing is drawn by hand.

    python scripts/make_figures.py                 # both languages -> docs/figures/{en,ru}/
    python scripts/make_figures.py --checkpoint runs/ptbxl/model.pt   # your own weights in figs 08-09

Figures 06, 08, 09, 11, 12 and 13 are drawn from the result tables in docs/results/ (written by
scripts/evaluate_ptbxl_model.py, compare_seeds.py, validate_twadb.py and validate_rpeaks.py) and
from the released model in models/ptbxl-1.0/, so no dataset download is needed to rebuild them.
Figure 02 simulates the Mitchell-Schaeffer cell model; figure 06 recomputes its table if missing.
"""
from __future__ import annotations

import argparse
import locale
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "models" / "ptbxl-1.0"
sys.path.insert(0, str(ROOT))
from cardioonco.preprocess import bandpass, detect_r_peaks  # noqa: E402
from cardioonco.synth import LEADS_12, SynthConfig, generate_12lead, generate_ecg, heart_vector  # noqa: E402
from cardioonco.twa import align_beats, analyze, beat_matrix  # noqa: E402

# ----------------------------------------------------------------------------- style
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

for _font in (ROOT / "scripts" / "fonts").glob("*.ttf"):     # PT Sans, the website's text face (OFL)
    font_manager.fontManager.addfont(str(_font))
# Use only the bundled PT Sans: macOS ships PT Sans as a .ttc collection, from which matplotlib
# renders regular-weight text blank, and other systems may have another version or none.
font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                    if f.name != "PT Sans" or Path(f.fname).parent == ROOT / "scripts" / "fonts"]

# One palette per style; every colour in the figures comes from these names.
STYLES = {
    "clinical": dict(INK="#16202e", MUTED="#5b6573", LINE="#d9d4d1", RED="#c8102e", BLUE="#1f5aa6",
                     GREEN="#1d7a4f", AMBER="#b07000", PAPER="#fbf9f8", GRID_MIN="#f3d9d9", GRID_MAJ="#e6b0b0",
                     BG="#ffffff", GRID="#ece8e6", DANGER=(200 / 255, 16 / 255, 46 / 255, 0.10),
                     DIV=("#1f5aa6", "#ffffff", "#c8102e"), SEQ=("#ffffff", "#f3c4cb", "#c8102e", "#5a0714"),
                     FONT=["DejaVu Sans"], GRID_ON=True),
    "journal": dict(INK="#222222", MUTED="#666666", LINE="#bdbdbd", RED="#D55E00", BLUE="#0072B2",
                    GREEN="#009E73", AMBER="#E69F00", PAPER="#ffffff", GRID_MIN="#f2f2f2", GRID_MAJ="#dedede",
                    BG="#ffffff", GRID="#eeeeee", DANGER=(213 / 255, 94 / 255, 0, 0.08),
                    DIV=("#0072B2", "#ffffff", "#D55E00"), SEQ=("#ffffff", "#fcd9c2", "#D55E00", "#5c2600"),
                    FONT=["PT Sans", "DejaVu Sans"], GRID_ON=False),
    "graphite": dict(INK="#efe9df", MUTED="#aeb5bc", LINE="#4a5662", RED="#f08a80", BLUE="#8fb3e0",
                     GREEN="#a9cdbf", AMBER="#e8c27a", PAPER="#252d36", GRID_MIN="#2e3843", GRID_MAJ="#3f4c59",
                     BG="#1f262e", GRID="#2e3843", DANGER=(240 / 255, 138 / 255, 128 / 255, 0.10),
                     DIV=("#8fb3e0", "#1f262e", "#f08a80"), SEQ=("#1f262e", "#7a4f4c", "#f08a80", "#ffe3dc"),
                     FONT=["PT Sans", "DejaVu Sans"], GRID_ON=True),
}
INK = MUTED = LINE = RED = BLUE = GREEN = AMBER = PAPER = GRID_MIN = GRID_MAJ = BG = None
DANGER = CMAP_DIV = CMAP_SEQ = None


def apply_style(name: str) -> None:
    """Set the module-wide palette and matplotlib defaults for one figure style."""
    global INK, MUTED, LINE, RED, BLUE, GREEN, AMBER, PAPER, GRID_MIN, GRID_MAJ, BG, DANGER, CMAP_DIV, CMAP_SEQ
    st = STYLES[name]
    INK, MUTED, LINE, RED, BLUE = st["INK"], st["MUTED"], st["LINE"], st["RED"], st["BLUE"]
    GREEN, AMBER, PAPER, GRID_MIN, GRID_MAJ, BG = st["GREEN"], st["AMBER"], st["PAPER"], st["GRID_MIN"], st["GRID_MAJ"], st["BG"]
    DANGER = st["DANGER"]
    CMAP_DIV = LinearSegmentedColormap.from_list("rb", list(st["DIV"]))
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq", list(st["SEQ"]))
    plt.rcParams.update({
        "font.family": st["FONT"], "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": st["GRID_ON"], "grid.color": st["GRID"],
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG,
        "axes.facecolor": BG, "savefig.facecolor": BG, "legend.labelcolor": INK, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


apply_style("journal")


def _decade(v, _pos=None) -> str:
    """Log-axis labels as plain text (10⁻¹, 10⁰, 10¹): unlike mathtext, plain text falls back to
    DejaVu Sans for glyphs the text face lacks, so the exponents never disappear."""
    if v <= 0:
        return ""
    e = np.log10(v)
    if abs(e - round(e)) > 1e-9:
        return ""
    return "10" + str(int(round(e))).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


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
        f2_title="Why the T wave alternates: a cardiac cell model (Mitchell & Schaeffer, 2003)",
        ms_normal=r"normal action potential ($\tau_{close}$ = 150 ms)", ms_prolonged=r"prolonged action potential ($\tau_{close}$ = 180 ms)", alt_zone="alternans", block_zone="2:1 block: every other stimulus answered",
        slope_note="dots: restitution slope = 1, where alternans can begin",
        hr_axis="heart rate, bpm", v_axis="membrane variable v (rest 0, peak 1)",
        di="diastolic interval DI, ms", apd="APD, ms", apdn=r"$\mathrm{APD}_n$, ms", apdn1=r"$\mathrm{APD}_{n+1}$, ms",
        bcl="pacing cycle length (BCL), ms",
        rest="restitution curve APD = f(DI)", slope1="slope = 1", stable="slow pacing, converges", alt="fast pacing, 2-cycle (alternans)",
        a2="a  Restitution: action potential duration after a diastolic interval", b2="b  Bifurcation: APD of the last 8 beats at each cycle length",
        c2="c  Membrane voltage at the same rhythm, BCL {bcl} ms ({hr} bpm)",
        d2="d  APD beat by beat at BCL {bcl} ms", danger2="alternans zone", healthy="healthy", steep="steeper restitution (injured)",
        # 03
        f3_title="The heart as a dipole: Einthoven's triangle and the 12 leads",
        a3="a  Frontal-plane vector loop", b3="b  Leads are projections of one vector", c3="c  Einthoven's law holds exactly",
        loopq="QRS loop", loopt="T loop", xl3="x (patient's left), mV", yl3="y (inferior), mV", resid="II − (I + III)",
        # 04
        f4_title="Signal pipeline: raw electrode voltage → clean beats → R peaks",
        a4="a  Raw signal: wander + 50 Hz mains + white noise", b4="b  Power spectrum before and after the filter",
        c4="c  Zero-phase band-pass 0.5–40 Hz", d4="d  QRS blocks (Elgendi): squared 8–20 Hz signal and two moving averages",
        e4="e  {n} R peaks found, beats overlaid",
        raw="raw", filt="filtered", freq="frequency, Hz", psd="power, mV²/Hz", mwi="QRS-length average (97 ms)", thr="beat-length average (611 ms) + 0.08·mean", blk="QRS block",
        qrsband="QRS energy", wander="breathing drift", mains="mains 50 Hz", passband="pass band\n0.5–40 Hz",
        # 05
        f5_title="T-wave alternans: the Spectral Method step by step",
        a5="a  128 aligned beats minus the mean beat", b5="b  Even (A) vs odd (B) average beats", c5="c  One ST-T point across beats",
        d5="d  Aggregate spectrum", stt="ST-T window", even="even beats A", odd="odd beats B", diffx="(A − B) × 10",
        cpb="cycles / beat", pow="power, µV²", noiseband="noise band", altpk="alternans\n0.5 cycles/beat", bpm="bpm",
        # 06
        f6_title="How often is alternans found? 1,440 analyses of synthetic recordings", a6="a  Called positive, % of {n} recordings per cell",
        b6="b  The same with 95% confidence intervals (Wilson)", p_pos="called positive, %", noise_lbl="noise {n} µV",
        five_pct="5%", alt_ax="true alternans amplitude, µV",
        mostly_noisy="hatched: mostly \u201cindeterminate: too noisy\u201d",
        hi_noise_note="at 60 and 100 µV of noise: \u201cindeterminate: too noisy\u201d in {p:.0f}% of recordings",
        noise_ax="white noise (a stand-in for muscle noise), µV RMS",
        kmap="K-score (log colour)", detected="TWA detected\nV_alt ≥ 1.9 µV and K ≥ 3", hidden="indeterminate:\nburied in noise",
        grid6="dots: simulated points, median of 4 recordings each",
        # 07
        f7_title="Continuous wavelet transform (complex Morlet): where the energy sits",
        a7="a  Normal beat", b7="b  Flattened T wave (repolarisation stress)", c7="c  Difference b − a: signal and |W|", fhz="frequency, Hz",
        wabs="|W|, arbitrary units", dwabs="|W_b| − |W_a|, arbitrary units",
        # 08
        f8_title="What does the trained network look at? Integrated gradients on two real PTB-XL test ECGs",
        a8="{letter}  {cls}: ECG {ecg}, predicted probability {p}", b8="{letter}  Sum per lead",
        cls_STTC="ST/T change (STTC)", cls_MI="myocardial infarction (MI)", contrib="contribution to the logit",
        cb8="contribution of each sample to the logit (red pushes towards the diagnosis, blue away from it)",
        completeness="sum {s} = logit change {d}",
        # 09
        f9_title="INT8 quantisation: {r}× smaller, the same accuracy on 2,158 test ECGs",
        b9_test="b  Probabilities, INT8 vs float32 ({n} ECGs × 5 = {k})", p_float="float32 probability",
        p_int8="INT8 probability", diff_note="median |difference| {med}\nlargest {mx}",
        c9_head="c  Whole model (ONNX Runtime, one CPU core)", ms_ecg="ms per ECG on this laptop", auc_diff="macro-AUC, INT8 − float32, 95% CI",
        a9_trained="a  Weights of one trained conv layer: float32 vs INT8", a9_init="a  Weights of one conv layer (untrained): float32 vs INT8",
        zoom9="zoom: x → x'", b9="b  Quantisation error", c9="c  Model size",
        w="weight value", count="count", err="error, % of the step s", size="MB",
        # 11
        f11_title="The network on the PTB-XL test fold (2,158 ECGs): accuracy, chance and calibration",
        released="released model (seed 42), 95% CI", ensemble5="ensemble of 5 seeds", macro_auc="macro-AUC",
        b11s="b  Five training seeds, with and without age/sex",
        effect_note="paired by seed: {d}, 95% CI [{lo}, {hi}]\nwith age/sex better in {k} of 5 seeds",
        c11="c  Calibration: does 0.8 mean 80%? (5 classes pooled)", raw_cal="as trained", platt_cal="Platt-scaled on validation",
        pred_prob="predicted probability", obs_freq="observed share with the diagnosis",
        a11="a  AUC per diagnosis, with 95% CI", b11="b  With age and sex minus without (paired)",
        auc="AUC", dauc="difference in AUC", macro="macro", published="published\nmodels [20]",
        nometa="without age/sex", withmeta="with age/sex",
        f13_title="R-peak detection checked against cardiologists' beat annotations (126 records, 294,077 beats)",
        a13="a  Each detector on two databases", b13="b  MIT-BIH record by record, worst first",
        d_fixed="one fixed threshold (up to 1.1)", d_pt="Pan–Tompkins adaptive (rejected)", d_ours="Elgendi (this version)",
        db_mit="MIT-BIH Arrhythmia, 360 Hz", db_sv="Supraventricular Arrhythmia, 128 Hz",
        se_ax="sensitivity: annotated beats found, %", pp_ax="precision: detections that are beats, %",
        err_ax="missed + false beats, % of the record's beats",
        # 12
        f12_title="The PhysioNet 2008 TWA challenge: synthetic recordings agree, real ones do not",
        a12="a  This pipeline against the challenge reference", b12="b  Kendall τ by group",
        refrank="challenge reference rank (100 = most alternans)", est="this pipeline: significance-gated V_alt, µV",
        g_syn="synthetic (32)", g_dev="real, used for development (34)", g_test="real, held out (34)",
        bar_all="all 100", bar_syn="synthetic", bar_dev="real,\ndevelopment", bar_test="real,\nheld out",
        sigline="organisers' significance line 0.436", entries="the 19 entries\nthat formed\nthe reference",
        # 10
        f10_title="Why early detection matters", a10="a  Heart failure vs cumulative doxorubicin dose (Swain et al., 2003)",
        b10="b  When is injury visible? (project hypothesis)", dose="cumulative dose, mg/m²", hf="patients with heart failure, %",
        mol="molecular\n(ROS, iron, TOP2B)", ele="electrical\n(QT, T-wave alternans)", mec="mechanical\n(LVEF ↓ on echo)",
        tgt="CardioOncoPredict\ntarget", std="standard\nmonitoring", time="time since start of chemotherapy (schematic)",
        visible="signal strength",
    ),
    "ru": dict(
        t_s="время, с", mv="мВ", uv="мкВ", ms="мс", beat="номер удара", fromR="время от R-пика, с",
        f1_title="От ионных каналов к зубцу T", ap="мембранный потенциал, мВ", ph0="0: вход Na⁺\n(деполяризация)",
        ph1="1: выход K⁺ (I_to)", ph2="2: вход Ca²⁺ ≈ выход K⁺ (плато)", ph3="3: выход K⁺ (I_Kr, I_Ks) = реполяризация",
        ph4="4: покой (I_K1)", normal="норма", ikr="I_Kr ↓ (препарат, повреждение)", endo="эндокард (ПД длиннее)",
        epi="эпикард (ПД короче)", pseudo="псевдо-ЭКГ  ∝  V_эндо − V_эпи", twave="зубец T = трансмуральный\nградиент реполяризации",
        qrs="QRS", a_title="a  Потенциал действия желудочка", b_title="b  Два слоя стенки", c_title="c  Что видит электрод",
        f2_title="Почему зубец T чередуется: модель клетки сердца (Mitchell & Schaeffer, 2003)",
        ms_normal=r"нормальный потенциал действия ($\tau_{close}$ = 150 мс)", ms_prolonged=r"удлинённый потенциал действия ($\tau_{close}$ = 180 мс)", alt_zone="альтернация", block_zone="блокада 2:1: ответ на каждый второй стимул",
        slope_note="точки: наклон реституции = 1, отсюда может начаться альтернация",
        hr_axis="ЧСС, уд/мин", v_axis="мембранная переменная v (покой 0, пик 1)",
        di="диастолический интервал DI, мс", apd="ДПД, мс", apdn=r"$\mathrm{ДПД}_n$, мс", apdn1=r"$\mathrm{ДПД}_{n+1}$, мс",
        bcl="период стимуляции (BCL), мс",
        rest="кривая реституции ДПД = f(DI)", slope1="наклон = 1", stable="редкий ритм, сходится", alt="частый ритм, 2-цикл (альтернация)",
        a2="a  Реституция: длительность потенциала действия после диастолы", b2="b  Бифуркация: ДПД последних 8 ударов при каждом цикле",
        c2="c  Мембранный потенциал при одном ритме, BCL {bcl} мс ({hr} уд/мин)",
        d2="d  ДПД от удара к удару при BCL {bcl} мс", danger2="зона альтернации", healthy="здоровая ткань", steep="более крутая реституция (повреждение)",
        f3_title="Сердце как диполь: треугольник Эйнтховена и 12 отведений",
        a3="a  Векторная петля во фронтальной плоскости", b3="b  Отведения — проекции одного вектора", c3="c  Закон Эйнтховена выполняется точно",
        loopq="петля QRS", loopt="петля T", xl3="x (влево от пациента), мВ", yl3="y (вниз), мВ", resid="II − (I + III)",
        f4_title="Обработка сигнала: напряжение электрода → чистые удары → R-пики",
        a4="a  Сырой сигнал: дрейф + сеть 50 Гц + белый шум", b4="b  Спектр мощности до и после фильтра",
        c4="c  Фильтр 0,5–40 Гц без сдвига фазы", d4="d  Блоки QRS (Elgendi): квадрат сигнала 8–20 Гц и два скользящих средних",
        e4="e  Найдено R-пиков: {n}, удары наложены",
        raw="сырой", filt="после фильтра", freq="частота, Гц", psd="мощность, мВ²/Гц", mwi="среднее за QRS (97 мс)", thr="среднее за удар (611 мс) + 0,08·среднее", blk="блок QRS",
        qrsband="энергия QRS", wander="дыхательный дрейф", mains="сеть 50 Гц", passband="полоса\n0,5–40 Гц",
        f5_title="Альтернация зубца T: спектральный метод шаг за шагом",
        a5="a  128 выровненных ударов минус средний удар", b5="b  Средние чётные (A) и нечётные (B) удары", c5="c  Одна точка ST-T по ударам",
        d5="d  Суммарный спектр", stt="окно ST-T", even="чётные удары A", odd="нечётные удары B", diffx="(A − B) × 10",
        cpb="циклы / удар", pow="мощность, мкВ²", noiseband="полоса шума", altpk="альтернация\n0,5 цикла/удар", bpm="уд/мин",
        f6_title="Как часто находится альтернация? 1440 анализов синтетических записей", a6="a  Названо положительным, % из {n} записей в клетке",
        b6="b  То же с 95 % доверительными интервалами (Уилсона)", p_pos="названо положительным, %", noise_lbl="шум {n} мкВ",
        five_pct="5 %", alt_ax="истинная амплитуда альтернации, мкВ",
        mostly_noisy="штриховка: чаще всего «не определено: слишком шумно»",
        hi_noise_note="при шуме 60 и 100 мкВ: «не определено: слишком шумно» в {p:.0f} % записей",
        noise_ax="белый шум (вместо мышечного), мкВ RMS",
        kmap="K-score (лог. шкала)", detected="TWA обнаружена\nV_alt ≥ 1,9 мкВ и K ≥ 3", hidden="неопределённо:\nтонет в шуме",
        grid6="точки — смоделированные условия, медиана по 4 записям",
        f7_title="Непрерывное вейвлет-преобразование (комплексный Морле): где сосредоточена энергия",
        a7="a  Нормальный удар", b7="b  Уплощённый зубец T (стресс реполяризации)", c7="c  Разность b − a: сигнал и |W|", fhz="частота, Гц",
        wabs="|W|, усл. ед.", dwabs="|W_b| − |W_a|, усл. ед.",
        f8_title="На что смотрит обученная сеть? Интегрированные градиенты на двух настоящих ЭКГ из теста PTB-XL",
        a8="{letter}  {cls}: ЭКГ {ecg}, предсказанная вероятность {p}", b8="{letter}  Сумма по отведению",
        cls_STTC="изменения ST/T (STTC)", cls_MI="инфаркт миокарда (MI)", contrib="вклад в логит",
        cb8="вклад каждого отсчёта в логит (красное толкает к диагнозу, синее — от него)",
        completeness="сумма {s} = изменение логита {d}",
        f9_title="INT8-квантование: в {r} раза меньше, та же точность на 2158 тестовых ЭКГ",
        b9_test="b  Вероятности: INT8 против float32 ({n} ЭКГ × 5 = {k})", p_float="вероятность float32",
        p_int8="вероятность INT8", diff_note="медиана |разницы| {med}\nнаибольшая {mx}",
        c9_head="c  Вся модель (ONNX Runtime, одно ядро процессора)", ms_ecg="мс на ЭКГ на этом ноутбуке", auc_diff="macro-AUC, INT8 − float32, 95 % ДИ",
        a9_trained="a  Веса одного обученного свёрточного слоя: float32 и INT8",
        a9_init="a  Веса одного свёрточного слоя (без обучения): float32 и INT8",
        zoom9="увеличено: x → x'", b9="b  Ошибка квантования", c9="c  Размер модели",
        w="значение веса", count="количество", err="ошибка, % от шага s", size="МБ",
        f11_title="Сеть на тестовом фолде PTB-XL (2158 ЭКГ): точность, случайность и калибровка",
        released="опубликованная модель (seed 42), 95 % ДИ", ensemble5="ансамбль из 5 seed", macro_auc="macro-AUC",
        b11s="b  Пять запусков обучения: с возрастом и полом и без",
        effect_note="парно по seed: {d}, 95 % ДИ [{lo}; {hi}]\nс возрастом и полом лучше в {k} из 5 запусков",
        c11="c  Калибровка: значит ли 0,8 «80 %»? (5 классов вместе)", raw_cal="как обучена", platt_cal="после Платта на валидации",
        pred_prob="предсказанная вероятность", obs_freq="доля с этим диагнозом на самом деле",
        a11="a  AUC по диагнозам, с 95 % ДИ", b11="b  С возрастом и полом минус без них (парно)",
        auc="AUC", dauc="разница AUC", macro="среднее", published="опубликованные\nмодели [20]",
        nometa="без возраста и пола", withmeta="с возрастом и полом",
        f13_title="Поиск R-пиков против отметок кардиологов (126 записей, 294 077 ударов)",
        a13="a  Каждый детектор на двух базах", b13="b  MIT-BIH по записям, начиная с худшей",
        d_fixed="один фиксированный порог (до 1.1)", d_pt="адаптивный Пан–Томпкинс (отклонён)", d_ours="Elgendi (эта версия)",
        db_mit="MIT-BIH Arrhythmia, 360 Гц", db_sv="Supraventricular Arrhythmia, 128 Гц",
        se_ax="чувствительность: найдено отмеченных ударов, %", pp_ax="точность: находок, которые оказались ударами, %",
        err_ax="пропущено + лишних, % от ударов записи",
        f12_title="Конкурс PhysioNet 2008 по TWA: синтетические записи согласуются, реальные — нет",
        a12="a  Этот пайплайн против эталона конкурса", b12="b  τ Кендалла по группам",
        refrank="эталонный ранг конкурса (100 = больше всего альтернации)", est="этот пайплайн: V_alt после проверки значимости, мкВ",
        g_syn="синтетические (32)", g_dev="реальные, для разработки (34)", g_test="реальные, отложенные (34)",
        bar_all="все 100", bar_syn="синтетические", bar_dev="реальные,\nразработка", bar_test="реальные,\nотложенные",
        sigline="порог значимости организаторов 0,436", entries="19 участников,\nиз которых\nсобран эталон",
        f10_title="Почему важно раннее выявление", a10="a  Сердечная недостаточность и кумулятивная доза доксорубицина (Swain et al., 2003)",
        b10="b  Когда повреждение становится видно? (гипотеза проекта)", dose="кумулятивная доза, мг/м²", hf="пациентов с СН, %",
        mol="молекулярный\n(АФК, железо, TOP2B)", ele="электрический\n(QT, альтернация T)", mec="механический\n(ФВ ЛЖ ↓ на ЭхоКГ)",
        tgt="цель\nCardioOncoPredict", std="стандартный\nконтроль", time="время от начала химиотерапии (схема)",
        visible="сила сигнала",
    ),
}


def dec(lang: str, s: str) -> str:
    """Decimal comma in Russian figures (0,5 rather than 0.5)."""
    return s.replace(".", ",") if lang == "ru" else s


def save(fig, out: Path, name: str):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_major_formatter(FuncFormatter(_decade))
                axis.set_minor_formatter(NullFormatter())
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
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.12, 0.0))   # clear of the resting line at -85 mV

    v_endo = action_potential(t, 280, t0=10)
    v_epi = action_potential(t, 240, t0=18)
    ax[1].plot(t, v_endo, color=BLUE, lw=2, label=T["endo"])
    ax[1].plot(t, v_epi, color=RED, lw=2, label=T["epi"])
    ax[1].fill_between(t, v_endo, v_epi, where=(t > 150), color=DANGER, lw=0)
    ax[1].set(title=T["b_title"], xlabel=T["ms"], ylim=(-95, 45))
    ax[1].legend(loc="upper right")

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
# Mitchell & Schaeffer (2003): a two-variable cardiac cell model that has restitution and
# alternans built into its equations.  v is the membrane variable (0 = rest, 1 = peak), h the gate.
#   dv/dt = h v^2 (1 - v) / tau_in - v / tau_out + J_stim
#   dh/dt = (1 - h) / tau_open  if v < v_gate,   -h / tau_close  otherwise
MS = dict(tau_in=0.3, tau_out=6.0, tau_open=120.0, v_gate=0.13)
TAU_CLOSE = {"normal": 150.0, "prolonged": 180.0}     # longer tau_close = longer action potential


def ms_restitution(di, tau_close):
    """The model's restitution curve, APD = tau_close * ln(h(DI) / h_min) (Mitchell & Schaeffer 2003)."""
    h_min = 4 * MS["tau_in"] / MS["tau_out"]
    h = 1 - (1 - h_min) * np.exp(-np.asarray(di, float) / MS["tau_open"])
    return tau_close * np.log(np.maximum(h, h_min) / h_min)


def ms_pacing(bcls, tau_close, n_beats=50, dt=0.04, trace_bcl=None):
    """Pace the model cell at every cycle length in `bcls` at once (forward Euler, 1 ms stimuli).
    Returns the APDs (time with v >= 0.1, in ms) of each cycle length and, for `trace_bcl`, the
    time course of v over the last four cycles."""
    bcls = np.asarray(bcls, float)
    v, h = np.zeros(len(bcls)), np.ones(len(bcls))
    above, start = np.zeros(len(bcls), bool), np.zeros(len(bcls))
    apds = [[] for _ in bcls]
    k_tr = int(np.argmin(np.abs(bcls - trace_bcl))) if trace_bcl else None
    t_end = n_beats * bcls
    tr_t, tr_v = [], []
    for i in range(int(n_beats * bcls.max() / dt)):
        t = i * dt
        stim = 0.5 * (((t % bcls) < 1.0) & (t < t_end))
        v, h = (v + dt * (h * v * v * (1 - v) / MS["tau_in"] - v / MS["tau_out"] + stim),
                h + dt * np.where(v < MS["v_gate"], (1 - h) / MS["tau_open"], -h / tau_close))
        now = v >= 0.1
        start[now & ~above] = t
        for k in np.flatnonzero(~now & above):
            apds[k].append(t - start[k])
        above = now
        if k_tr is not None and t_end[k_tr] - 4 * bcls[k_tr] <= t < t_end[k_tr]:
            tr_t.append(t - (t_end[k_tr] - 4 * bcls[k_tr]))
            tr_v.append(v[k_tr])
    return apds, (np.array(tr_t), np.array(tr_v))


_MS_CACHE: dict = {}
BCL_COMPARE = 335.0      # the normal cell is steady here, the prolonged one alternates


def ms_results():
    if not _MS_CACHE:
        print("   simulating the Mitchell-Schaeffer cell (2 x 101 cycle lengths)...")
        bcls = np.arange(250.0, 452.0, 2.0)
        for name, tc in TAU_CLOSE.items():
            _MS_CACHE[name] = (bcls,) + ms_pacing(bcls, tc, trace_bcl=BCL_COMPARE)
    return _MS_CACHE


def fig02(lang, out):
    T = L[lang]
    res = ms_results()
    col = {"normal": BLUE, "prolonged": RED}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.4), layout="constrained")
    # a: restitution curves from the model equations, slope-1 points marked
    a0 = ax[0, 0]
    di = np.linspace(1, 400, 800)
    for name, tc in TAU_CLOSE.items():
        apd = ms_restitution(di, tc)
        slope = np.gradient(apd, di)
        a0.plot(di, apd, color=col[name], lw=2.2, label=T[f"ms_{name}"])
        k = int(np.argmin(np.abs(slope - 1)))
        a0.plot(di[k], apd[k], "o", color=col[name], ms=6, mec=BG, mew=1.2)
        a0.plot(di[k] + np.array([-45, 45]), apd[k] + np.array([-45, 45]), color=col[name], lw=1, ls=":")
    a0.set(title=T["a2"], xlabel=T["di"], ylabel=T["apd"], xlim=(0, 400), ylim=(0, 320))
    a0.text(0.97, 0.06, T["slope_note"], transform=a0.transAxes, ha="right", fontsize=8.5, color=MUTED)
    a0.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12))
    # b: bifurcation diagram from the full simulation, APD of the last 8 beats at each cycle length
    b0 = ax[0, 1]
    for name in TAU_CLOSE:
        bcls, apds, _ = res[name]
        for bcl, seq in zip(bcls, apds):
            tail = np.asarray(seq[-8:])
            b0.scatter(np.full(len(tail), bcl), tail, s=7, color=col[name], lw=0)
        alt = [bcl for bcl, seq in zip(bcls, apds) if len(seq) > 0.8 * 50 and np.ptp(seq[-8:]) > 2]
        if alt:
            b0.axvspan(min(alt) - 1, max(alt) + 1, color=col[name], alpha=0.12, lw=0)
            b0.text(np.mean(alt), 30, T["alt_zone"], color=col[name], ha="center", fontsize=8.5)
        block = [bcl for bcl, seq in zip(bcls, apds) if len(seq) <= 0.8 * 50]
        if name == "prolonged" and block:
            b0.annotate(T["block_zone"], xy=(max(block) - 10, 344), xytext=(250, 372), fontsize=8.5, ha="right",
                        color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    b0.axvline(BCL_COMPARE, color=MUTED, lw=1, ls="--")
    b0.set(title=T["b2"], xlabel=T["bcl"], ylabel=T["apd"], xlim=(452, 248), ylim=(0, 385))
    sec = b0.secondary_xaxis("top", functions=(lambda x: 60000.0 / np.maximum(x, 1), lambda x: 60000.0 / np.maximum(x, 1)))
    sec.set_xlabel(T["hr_axis"], color=MUTED)
    sec.tick_params(colors=MUTED)
    # c: membrane voltage over the last four cycles at the same cycle length
    c0 = ax[1, 0]
    for name in TAU_CLOSE:
        t, v = res[name][2]
        c0.plot(t, v, color=col[name], lw=1.8, label=T[f"ms_{name}"])
    c0.set(title=T["c2"].format(bcl=int(BCL_COMPARE), hr=int(round(60000 / BCL_COMPARE))), xlabel=T["ms"], ylabel=T["v_axis"],
           xlim=(0, 4 * BCL_COMPARE), ylim=(-0.05, 1.12))
    c0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=8)
    # d: APD beat by beat at the same cycle length
    d0 = ax[1, 1]
    for name in TAU_CLOSE:
        bcls, apds, _ = res[name]
        seq = np.asarray(apds[int(np.argmin(np.abs(bcls - BCL_COMPARE)))][:30])
        d0.plot(np.arange(len(seq)), seq, "o-", color=col[name], ms=4, lw=1.2, label=T[f"ms_{name}"])
    d0.set(title=T["d2"].format(bcl=int(BCL_COMPARE)), xlabel=T["beat"], ylabel=T["apd"])
    d0.legend(loc="lower right")
    fig.suptitle(T["f2_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
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
    ax0.set(title=T["a3"], xlim=(-1.9, 1.9), ylim=(1.9, -1.2), xlabel=T["xl3"], ylabel=T["yl3"])
    ax0.set_aspect("equal", adjustable="datalim")   # keeps the panel as tall as its neighbours
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
    b.text(0.55, p2.max() * 3e-8, T["passband"], color=GREEN, fontsize=8.5)
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
    # detector internals (Elgendi 2013), computed exactly as cardioonco.preprocess.qrs_blocks does
    from cardioonco.preprocess import bandpass as bp
    f = bp(x, fs, 8.0, 20.0, order=3)
    y = f ** 2
    n1, n2 = int(round(0.097 * fs)), int(round(0.611 * fs))
    ma_qrs = np.convolve(y, np.ones(n1) / n1, mode="same")
    thr = np.convolve(y, np.ones(n2) / n2, mode="same") + 0.08 * y.mean()
    top = ma_qrs[win].max()
    dd = fig.add_subplot(gs[2, 0])
    dd.fill_between(t[win], 0, 1.0, where=(ma_qrs > thr)[win], color=(31 / 255, 90 / 255, 166 / 255, 0.10),
                    lw=0, label=T["blk"])
    dd.plot(t[win], y[win] / y[win].max(), color=MUTED, lw=0.6, label="y = f²")
    dd.plot(t[win], ma_qrs[win] / top, color=BLUE, lw=1.8, label=T["mwi"])
    dd.plot(t[win], thr[win] / top, color=RED, ls="--", lw=1.2, label=T["thr"])
    dd.set(title=T["d4"], xlabel=T["t_s"], xlim=(2, 7), ylim=(0, 1.32))
    dd.legend(loc="upper right", ncol=2, fontsize=8)
    ee = fig.add_subplot(gs[2, 1])
    beats = beat_matrix(xf, r, fs, -0.25, 0.55)
    tb = np.arange(beats.shape[1]) / fs - 0.25
    for row in beats:
        ee.plot(tb, row, color=INK, lw=0.5, alpha=0.25)
    ee.plot(tb, beats.mean(0), color=RED, lw=2)
    ee.axvline(0, color=RED, lw=0.8, ls=":")
    ee.set(title=T["e4"].format(n=len(r)), xlabel=T["fromR"], ylabel=T["mv"])
    fig.suptitle(T["f4_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.96)
    save(fig, out, "04_signal_pipeline.png")


# ============================================================================ 05
def fig05(lang, out):
    T = L[lang]
    fs = 500
    _, x, _ = generate_ecg(SynthConfig(fs=fs, n_beats=130, alternans_uv=15, noise_uv=8, heart_rate=90, hrv_std=0.004,
                                       t_amp=0.28, seed=11))
    res = analyze(x, fs)
    xf = bandpass(x, fs)
    r = align_beats(xf, detect_r_peaks(x, fs), fs)     # as in the analysis: beats superimposed on the QRS
    # the first and last beats of the recording touch its edges, where the filter has not settled
    B = beat_matrix(xf, r, fs, -0.25, 0.55)[1:-1][:128]
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
           bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=RED, lw=0.8))
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
    c.set(title=f"{T['c5']} (t = {dec(lang, f'{tb[kbest]:.3f}')} {T['t_s'].split(', ')[-1]})", xlabel=T["beat"], ylabel=T["uv"], xlim=(-1, len(s)))
    d = fig.add_subplot(gs[1, 1])
    f, P = np.array(res.spectrum_f), np.array(res.spectrum_p_uv2)
    d.axvspan(0.44, 0.49, color=(31 / 255, 90 / 255, 166 / 255, 0.12), lw=0, label=T["noiseband"])
    d.semilogy(f[1:], P[1:], color=INK, lw=1.3)
    d.plot(0.5, P[-1], "o", color=RED, ms=9)
    d.annotate(T["altpk"], (0.5, P[-1]), xytext=(0.33, P[-1] * 0.6), color=RED, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=RED))
    d.text(0.02, 0.95, dec(lang, f"V_alt = {res.v_alt_uv:.1f} {T['uv']}   K = {res.k_score:.0f}\n"
                                 f"MMA   = {res.mma_uv:.1f} {T['uv']}  HR = {res.heart_rate_bpm:.0f} {T['bpm']}"),
           transform=d.transAxes, va="top", family="DejaVu Sans Mono", fontsize=9.5,
           bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=LINE))
    d.set(title=T["d5"], xlabel=T["cpb"], ylabel=T["pow"], xlim=(0, 0.51))
    d.legend(loc="lower left")
    fig.suptitle(T["f5_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.97)
    save(fig, out, "05_twa_spectral_method.png")


# ============================================================================ 06
DETECTION_CSV = ROOT / "docs" / "results" / "detection_map.csv"
DET_ALTS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15)
DET_NOISES = (5, 10, 20, 40, 60, 100)
DET_N = 24


def detection_grid(path=DETECTION_CSV):
    """Share of synthetic recordings called 'positive' for each alternans amplitude and noise level
    (DET_N recordings per cell, 128 beats at 75 bpm, white noise, fixed seeds).  The table is cached
    in docs/results/ and recomputed when missing (about a minute)."""
    import csv
    if not path.exists():
        print(f"   computing the detection grid ({len(DET_ALTS) * len(DET_NOISES) * DET_N} analyses)...")
        rows = []
        for i, nz in enumerate(DET_NOISES):
            for j, al in enumerate(DET_ALTS):
                pos = noisy = 0
                for k in range(DET_N):
                    _, x, _ = generate_ecg(SynthConfig(alternans_uv=al, noise_uv=nz, seed=50000 + 1000 * i + 100 * j + k))
                    r = analyze(x, 500)
                    pos += r.outcome == "positive"
                    noisy += r.outcome == "indeterminate" and r.reason == "noise"
                rows.append((al, nz, pos, noisy, DET_N))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("alternans_uv,noise_uv,positive,indeterminate_noise,n\n"
                        + "".join(",".join(map(str, r)) + "\n" for r in rows))
    return [(float(r["alternans_uv"]), float(r["noise_uv"]), int(r["positive"]), int(r["indeterminate_noise"]), int(r["n"]))
            for r in csv.DictReader(open(path))]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def fig06(lang, out):
    T = L[lang]
    rows = detection_grid()
    alts, noises = sorted({r[0] for r in rows}), sorted({r[1] for r in rows})
    P, Q = np.zeros((len(noises), len(alts))), np.zeros((len(noises), len(alts)))
    for a_, n_, k, noisy, n in rows:
        P[noises.index(n_), alts.index(a_)] = k / n
        Q[noises.index(n_), alts.index(a_)] = noisy / n
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1, 1.1]}, layout="constrained")
    a = ax[0]
    im = a.imshow(P * 100, origin="lower", aspect="auto", cmap=CMAP_SEQ, vmin=0, vmax=100)
    from matplotlib.patches import Rectangle
    for i in range(len(noises)):
        for j in range(len(alts)):
            if Q[i, j] >= 0.5:      # the analysis mostly declined to answer: too noisy
                a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec=LINE, lw=0))
            v = P[i, j] * 100
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=BG if v > 60 else INK)
    a.add_patch(Rectangle((0, 0), 0, 0, fill=False, hatch="///", ec=LINE, lw=0, label=T["mostly_noisy"]))
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False)
    a.set_xticks(range(len(alts)), [f"{v:g}" for v in alts])
    a.set_yticks(range(len(noises)), [f"{v:g}" for v in noises])
    a.set(xlabel=T["alt_ax"], ylabel=T["noise_ax"], title=T["a6"].format(n=DET_N))
    a.grid(False)
    cb = fig.colorbar(im, ax=a, pad=0.01)
    cb.set_label(T["p_pos"])
    b = ax[1]
    for nz, c, dx in ((5, BLUE, -0.12), (20, GREEN, 0.0), (40, AMBER, 0.12)):
        sel = sorted((r for r in rows if r[1] == nz), key=lambda r: r[0])
        x = np.array([r[0] for r in sel]) + dx
        p = np.array([r[2] / r[4] for r in sel]) * 100
        lo, hi = np.array([wilson(r[2], r[4]) for r in sel]).T * 100
        b.errorbar(x, p, yerr=[p - lo, hi - p], fmt="o-", color=c, lw=1.6, ms=4, capsize=2.5, elinewidth=1,
                   label=T["noise_lbl"].format(n=nz))
    hi_noise = [r for r in rows if r[1] >= 60]
    share = 100 * sum(r[3] for r in hi_noise) / sum(r[4] for r in hi_noise)
    b.text(0.98, 0.04, T["hi_noise_note"].format(p=share), transform=b.transAxes, ha="right", va="bottom",
           fontsize=8.5, color=MUTED)
    b.axhline(5, color=MUTED, lw=0.8, ls=":")
    b.set(xlabel=T["alt_ax"], ylabel=T["p_pos"], title=T["b6"], ylim=(-3, 103), xlim=(-0.5, 15.5))
    b.legend(loc="center right")
    fig.suptitle(T["f6_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
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
    # one beat on a flat baseline: the segment starts and ends at the isoelectric line, so it is padded
    # with zeros; otherwise the slow wavelets would see the cut edges, or the neighbouring beats
    # (the heart rate and its harmonics would then show up as horizontal bands)
    pad = 2 * fs
    freqs = np.geomspace(0.8, 60, 80)
    wav = "cmor1.5-1.0"
    scales = pywt.central_frequency(wav) * fs / freqs
    Wa = np.abs(pywt.cwt(np.pad(xa, pad), scales, wav, sampling_period=1 / fs)[0])[:, pad:-pad]
    Wb = np.abs(pywt.cwt(np.pad(xb, pad), scales, wav, sampling_period=1 / fs)[0])[:, pad:-pad]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5), gridspec_kw={"height_ratios": [1, 2.2]}, sharex=True,
                           layout="constrained")
    vmax = max(Wa.max(), Wb.max())
    for j, (x, W, ttl) in enumerate([(xa, Wa, T["a7"]), (xb, Wb, T["b7"])]):
        ecg_paper(ax[0, j], (t[0], t[-1]), (-0.4, 1.4), major_y=0.5)
        ax[0, j].plot(t, x, color=INK, lw=1.4)
        ax[0, j].set_title(ttl)
        im = ax[1, j].pcolormesh(t, freqs, W, cmap=CMAP_SEQ, vmin=0, vmax=vmax, shading="auto")
        ax[1, j].set(yscale="log", ylabel=T["fhz"] if j == 0 else None, xlabel=T["fromR"])
        ax[1, j].grid(False)
    fig.colorbar(im, ax=ax[1, :2].tolist(), pad=0.01, aspect=30, label=T["wabs"])   # one scale for a and b
    ax[0, 2].plot(t, xb - xa, color=RED, lw=1.4)
    ax[0, 2].set_title(T["c7"])
    dW = Wb - Wa
    lim = np.abs(dW).max()
    imd = ax[1, 2].pcolormesh(t, freqs, dW, cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
    fig.colorbar(imd, ax=ax[1, 2], pad=0.01, aspect=30, label=T["dwabs"])
    ax[1, 2].set(yscale="log", xlabel=T["fromR"])
    ax[1, 2].grid(False)
    for a in ax[1]:
        a.axvspan(0.1, 0.42, color=(1, 1, 1, 0), ec=INK, ls="--", lw=1)
        a.text(0.26, 50, L[lang]["stt"], ha="center", va="top", fontsize=8.5, color=INK)
    fig.suptitle(T["f7_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "07_cwt_scalogram.png")


# ============================================================================ 08
EVAL_JSON = ROOT / "docs" / "results" / "ptbxl_evaluation.json"


def fig08(lang, out, checkpoint=None):
    """What the trained network looks at: integrated gradients on two real test ECGs
    (computed by scripts/evaluate_ptbxl_model.py, stored in docs/results/)."""
    import json
    from matplotlib.collections import LineCollection
    T = L[lang]
    ev = json.loads(EVAL_JSON.read_text())["attributions"]
    npz = np.load(ROOT / "docs" / "results" / "ptbxl_attributions.npz")
    fig = plt.figure(figsize=(15, 11.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1])
    tt = np.arange(1000) / 100
    lim = max(np.percentile(np.abs(npz[f"attr_{t}"].astype(float)), 99.5) for t in ("STTC", "MI"))
    norm = TwoSlopeNorm(0, -lim, lim)
    for row, (target, letter) in enumerate((("STTC", "a"), ("MI", "c"))):
        ecg, attr = npz[f"ecg_{target}"].astype(float), npz[f"attr_{target}"].astype(float)
        info = ev[target]
        ax = fig.add_subplot(gs[row, 0])
        for i in range(12):
            y = ecg[i] - np.median(ecg[i]) - i * 2.0
            ax.plot(tt, y, color=MUTED, lw=0.7, alpha=0.7)          # the ECG itself
            pts = np.column_stack([tt, y]).reshape(-1, 1, 2)
            seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
            rgba = CMAP_DIV(norm(attr[i, :-1]))
            rgba[:, 3] = np.clip(np.abs(attr[i, :-1]) / (0.35 * lim), 0, 1)   # unimportant samples fade out
            ax.add_collection(LineCollection(seg, colors=rgba, lw=2.0))
            ax.text(-0.15, -i * 2.0, LEADS_12[i], ha="right", va="center", fontsize=8.5, color=MUTED)
        ax.set(xlim=(0, 10), ylim=(-23.5, 2.2), yticks=[], xlabel=T["t_s"],
               title=T["a8"].format(letter=letter, cls=T[f"cls_{target}"], ecg=info["ecg_id"],
                                    p=dec(lang, f"{info['prob']:.3f}")))
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        bx = fig.add_subplot(gs[row, 1])
        per_lead = attr.sum(axis=1)
        bx.barh(np.arange(12), per_lead, color=[RED if v > 0 else BLUE for v in per_lead], height=0.7)
        bx.set_yticks(np.arange(12), LEADS_12)
        bx.invert_yaxis()
        bx.axvline(0, color=INK, lw=0.8)
        bx.set(title=T["b8"].format(letter=chr(ord(letter) + 1)), xlabel=T["contrib"])
        bx.text(0.98, 0.02, T["completeness"].format(s=dec(lang, f"{info['completeness_sum']:.2f}"),
                                                       d=dec(lang, f"{info['completeness_target']:.2f}")),
                transform=bx.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DIV)
    cb = fig.colorbar(sm, ax=fig.axes[0::2], location="bottom", shrink=0.45, pad=0.02, aspect=40)
    cb.set_label(T["cb8"])
    fig.suptitle(T["f8_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "08_network_attributions.png")


# ============================================================================ 09
def fig09(lang, out, checkpoint=None):
    """INT8 quantisation: the arithmetic on one trained layer, and the whole model measured on the
    test fold (scripts/evaluate_ptbxl_model.py)."""
    import json
    import torch
    from cardioonco.model import CardioOncoNet
    T = L[lang]
    ck = torch.load(checkpoint or RELEASED / "model.pt", map_location="cpu", weights_only=False)
    net = CardioOncoNet(n_classes=len(ck["classes"]), width=ck["width"])
    net.load_state_dict(ck["state_dict"])
    w = net.ecg_encoder.blocks[3].body[0].weight.detach().numpy().ravel().astype(np.float64)
    lo, hi = w.min(), w.max()
    s = (hi - lo) / 255
    z = np.round(-lo / s)
    q = np.clip(np.round(w / s) + z, 0, 255)
    wq = (q - z) * s
    ev = json.loads(EVAL_JSON.read_text())["int8"]
    pr = np.load(ROOT / "docs" / "results" / "ptbxl_test_probabilities.npz")
    fig = plt.figure(figsize=(15.5, 4.8), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.05, 0.9])
    a = fig.add_subplot(gs[:, 0])
    edges = (np.arange(-1, 257, 2) + 0.5 - z) * s     # each bin holds exactly two INT8 levels
    a.hist(w, bins=edges, color=MUTED, alpha=0.6, label="float32")
    a.hist(wq, bins=edges, color=RED, histtype="step", lw=1.3, label="INT8 → float")
    a.set(title=T["a9_trained"], xlabel=T["w"], ylabel=T["count"])
    a.legend(loc="upper right")
    a.text(0.02, 0.95, "q  = round(x / s) + z\nx' = (q − z)·s\ns  = (max − min) / 255", transform=a.transAxes,
           va="top", family="DejaVu Sans Mono", fontsize=8.5, bbox=dict(boxstyle="round", fc=BG, ec=LINE))
    zoom = a.inset_axes([0.66, 0.3, 0.31, 0.42])
    xs = np.linspace(-3 * s, 3 * s, 400)
    zoom.plot(xs / s, xs / s, color=MUTED, lw=1, ls=":")
    zoom.plot(xs / s, (np.round(xs / s + z) - z), color=RED, lw=1.5)
    zoom.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2], title=T["zoom9"])
    zoom.title.set_fontsize(8)
    zoom.tick_params(labelsize=7)
    b = fig.add_subplot(gs[:, 1])
    pf, pq = pr["p_float"].ravel(), pr["p_int8"].ravel()
    b.scatter(pf, pq, s=3, color=BLUE, alpha=0.25, lw=0)
    b.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=":")
    b.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=T["p_float"], ylabel=T["p_int8"],
          title=T["b9_test"].format(n=len(pr["y"]), k=pf.size))
    b.text(0.04, 0.96, T["diff_note"].format(med=dec(lang, f"{ev['median_abs_prob_diff']:.4f}"),
                                             mx=dec(lang, f"{ev['max_abs_prob_diff']:.2f}")),
           transform=b.transAxes, va="top", fontsize=8.5, bbox=dict(boxstyle="round", fc=BG, ec=LINE))
    rows = [(T["size"], ev["float32_mb"], ev["int8_mb"], "{:.2f}"),
            (T["ms_ecg"], ev["float32_ms_per_ecg"], ev["int8_ms_per_ecg"], "{:.1f}")]
    for r, (lab, f, i8, fmt) in enumerate(rows):
        c = fig.add_subplot(gs[r, 2])
        c.barh([1, 0], [f, i8], color=[MUTED, RED], height=0.6)
        c.set_yticks([1, 0], ["float32", "INT8"])
        for y, v in ((1, f), (0, i8)):
            c.text(v, y, " " + dec(lang, fmt.format(v)), va="center", fontsize=9)
        c.set_xlim(0, f * 1.3)
        c.set_title(lab, fontsize=9.5, loc="left")
        c.grid(False)
        if r == 0:
            c.set_title(T["c9_head"] + "\n" + lab, fontsize=9.5, loc="left")
    c = fig.add_subplot(gs[2, 2])
    d, (l0, h0) = ev["auc_difference_int8_minus_float"], ev["auc_difference_ci95"]
    c.errorbar([d], [0], xerr=[[d - l0], [h0 - d]], fmt="D", color=INK, capsize=4)
    c.axvline(0, color=MUTED, lw=0.8)
    c.set(yticks=[], xlim=(-0.004, 0.004))
    c.set_title(T["auc_diff"], fontsize=9.5, loc="left")
    c.grid(False)
    c.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: dec(lang, f"{v:+.3f}".replace("+0.000", "0"))))
    fig.suptitle(T["f9_title"].format(r=dec(lang, f"{ev['float32_mb'] / ev['int8_mb']:.1f}")), x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
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


# ============================================================================ 11
def fig11(lang, out, run=RELEASED):
    """The network on the PTB-XL test fold: per-class AUC of the released model, five seeds and their
    ensemble, the age/sex effect paired by seed, and calibration (docs/results/ptbxl_*.json)."""
    import json
    T = L[lang]
    res = ROOT / "docs" / "results"
    m = json.loads((run / "metrics.json").read_text())
    sd = json.loads((res / "ptbxl_seeds.json").read_text())
    ev = json.loads((res / "ptbxl_evaluation.json").read_text())["calibration"]
    pr = np.load(res / "ptbxl_test_probabilities.npz")
    cls = ["NORM", "MI", "STTC", "CD", "HYP"]
    num = lambda v, k=3: dec(lang, f"{v:.{k}f}")  # noqa: E731
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw={"width_ratios": [1.25, 0.9, 1]}, layout="constrained")
    # a: per-class AUC of the released model with CI, and the ensemble
    a = ax[0]
    y = np.arange(len(cls))[::-1].astype(float) + 1
    for k, (yi, c) in enumerate(zip(y, cls)):
        v = m["per_class"][c]
        a.plot(v["auc_ci95"], [yi, yi], color=INK, lw=2)
        a.plot(v["auc"], yi, "o", color=INK, ms=7, label=T["released"] if k == 0 else None)
        a.plot(sd["ensemble"]["per_class"][c], yi + 0.25, "*", color=RED, ms=10, label=T["ensemble5"] if k == 0 else None)
    a.axhspan(-0.5, 0.5, color=GRID_MIN, lw=0)
    a.fill_betweenx([-0.45, 0.45], 0.92, 0.93, color=BLUE, alpha=0.18, lw=0)
    a.text(0.9315, -0.22, T["published"].replace("\n", " "), color=BLUE, fontsize=7.5)
    a.plot(m["test_macro_auc"], 0, "D", color=INK, ms=7)
    a.plot(sd["ensemble"]["macro_auc"], 0.25, "*", color=RED, ms=11)
    a.text(0.975, 0, num(m["test_macro_auc"]), va="center", fontsize=9)
    a.text(0.975, 0.3, num(sd["ensemble"]["macro_auc"]), va="center", fontsize=9, color=RED, fontweight="bold")
    a.set(yticks=list(y) + [0], yticklabels=cls + [T["macro"]], xlim=(0.865, 0.99), ylim=(-0.6, 5.6),
          xlabel=T["auc"], title=T["a11"])
    a.legend(loc="upper left", fontsize=8)
    # b: five seeds, with and without age/sex, paired
    b = ax[1]
    seeds = sd["seeds"]
    with_ = [sd["runs"][f"meta_s{s}"] for s in seeds]
    without = [sd["runs"][f"nometa_s{s}"] for s in seeds]
    for w0, w1 in zip(without, with_):
        b.plot([0, 1], [w0, w1], color=LINE, lw=1)
    b.plot(np.zeros(5), without, "o", color=MUTED, ms=6)
    b.plot(np.ones(5), with_, "o", color=INK, ms=6)
    e = sd["metadata_effect"]
    lo_, hi_ = min(without + with_), max(without + with_)
    b.set(xticks=[0, 1], xticklabels=[T["nometa"], T["withmeta"]], xlim=(-0.4, 1.4), ylabel=T["macro_auc"],
          title=T["b11s"], ylim=(lo_ - 0.45 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_)))
    b.text(0.5, 0.04, T["effect_note"].format(d=dec(lang, f"{e['mean']:+.4f}"), lo=dec(lang, f"{e['bootstrap_ci95_pooled'][0]:+.4f}"),
                                             hi=dec(lang, f"{e['bootstrap_ci95_pooled'][1]:+.4f}"), k=e["seeds_with_gain"]),
           transform=b.transAxes, ha="center", fontsize=8.5, bbox=dict(boxstyle="round", fc=BG, ec=LINE))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: num(v)))
    # c: reliability diagram, all five classes pooled, before and after Platt scaling
    c = ax[2]
    c.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    yy = pr["y"].ravel().astype(float)
    for key, colr, lab in (("p_float", MUTED, T["raw_cal"]), ("p_platt", BLUE, T["platt_cal"])):
        p = pr[key].ravel()
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys, ns = [], [], []
        for k in range(10):
            sel = idx == k
            if sel.sum() >= 20:
                xs.append(p[sel].mean()); ys.append(yy[sel].mean()); ns.append(sel.sum())
        ece = np.mean([ev[cc]["ece_raw" if key == "p_float" else "ece_platt"] for cc in cls])
        c.plot(xs, ys, "o-", color=colr, lw=1.6, ms=5, label=f"{lab}: ECE {num(ece)}")
    c.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=T["pred_prob"], ylabel=T["obs_freq"], title=T["c11"])
    c.legend(loc="upper left", fontsize=8.5)
    fig.suptitle(T["f11_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "11_ptbxl_results.png")


# ============================================================================ 12
def fig12(lang, out, csv_path=ROOT / "docs" / "results" / "twadb_v0.6.0.csv"):
    """Section 7.1 as a picture, from the per-record table that scripts/validate_twadb.py writes."""
    import csv
    from scipy.stats import kendalltau
    T = L[lang]
    rows = list(csv.DictReader(open(csv_path)))
    groups = [("synthetic", T["g_syn"], dict(color=BLUE, marker="o", s=34)),
              ("real-dev", T["g_dev"], dict(facecolors=BG, edgecolors=MUTED, marker="o", s=34, linewidths=1.2)),
              ("real-test", T["g_test"], dict(color=AMBER, marker="D", s=30))]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    a = ax[0]
    for g, lab, style in groups:
        sel = [r for r in rows if r["group"] == g]
        a.scatter([int(r["reference_rank"]) for r in sel], [float(r["estimate_uv"]) for r in sel], label=lab, zorder=3, **style)
    a.set_yscale("symlog", linthresh=1)
    a.set_yticks([0, 1, 10])
    a.set_yticklabels(["0", "1", "10"])
    a.set(xlim=(0, 101), ylim=(-0.15, 25), xlabel=T["refrank"], ylabel=T["est"], title=T["a12"])
    a.legend(loc="upper left", bbox_to_anchor=(0.2, 1.0))   # clear of the held-out record twa32 at rank 18
    b = ax[1]
    def tau(sel):
        return kendalltau([float(r["estimate_uv"]) for r in sel], [int(r["reference_rank"]) for r in sel]).statistic
    bars = [(T["bar_all"], tau(rows), INK), (T["bar_syn"], tau([r for r in rows if r["group"] == "synthetic"]), BLUE),
            (T["bar_dev"], tau([r for r in rows if r["group"] == "real-dev"]), MUTED),
            (T["bar_test"], tau([r for r in rows if r["group"] == "real-test"]), AMBER)]
    x = np.arange(len(bars))
    b.bar(x, [v for _, v, _ in bars], color=[c for _, _, c in bars], width=0.55, zorder=3)
    for xi, (_, v, _) in zip(x, bars):
        b.text(xi, v + 0.02, dec(lang, f"{v:.2f}"), ha="center", fontsize=10, fontweight="bold")
    b.axhline(0.436, color=INK, lw=1, ls="--", zorder=2)
    b.text(3.35, 0.448, T["sigline"], ha="right", va="bottom", fontsize=8.5)
    b.plot([-0.42, -0.42], [0.451, 0.911], color=INK, lw=6, alpha=0.18, solid_capstyle="butt")   # entries' range, all 100 only
    b.text(-0.36, 0.68, T["entries"], fontsize=8, va="center", ha="left")
    b.set(xticks=x, xticklabels=[n for n, _, _ in bars], ylim=(0, 1), xlim=(-0.6, 3.5), ylabel="τ", title=T["b12"])
    fig.suptitle(T["f12_title"], x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, out, "12_twadb_check.png")


# ============================================================================ 13
# Totals of the two earlier detectors, from the runs described in README section 4.3 (git history
# of cardioonco/preprocess.py: version 1.1 and the adaptive Pan-Tompkins variant tried for 1.2).
EARLIER_DETECTORS = {
    "fixed": {"mitdb": (99298, 10196, 35), "svdb": (172200, 12383, 96)},
    "pt": {"mitdb": (108661, 833, 541), "svdb": (183372, 1211, 1299)},
}


def fig13(lang, out):
    """R-peak detection on cardiologist-annotated databases (scripts/validate_rpeaks.py)."""
    import csv
    T = L[lang]
    res = ROOT / "docs" / "results"
    tables = {db: list(csv.DictReader(open(res / f"rpeaks_{db}.csv"))) for db in ("mitdb", "svdb")}
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [1, 1.25]}, layout="constrained")
    a = ax[0]
    det = [("fixed", T["d_fixed"], LINE), ("pt", T["d_pt"], AMBER), ("neurokit2", "NeuroKit2", BLUE),
           ("ours", T["d_ours"], RED)]
    for k, (db, title) in enumerate((("mitdb", T["db_mit"]), ("svdb", T["db_sv"]))):
        for name, lab, col in det:
            if name in EARLIER_DETECTORS:
                tp, fn, fp = EARLIER_DETECTORS[name][db]
            else:
                rows = tables[db]
                tp, fn, fp = (sum(int(r[f"{name}_{m}"]) for r in rows) for m in ("tp", "fn", "fp"))
            se, pp = 100 * tp / (tp + fn), 100 * tp / (tp + fp)
            a.scatter(se, pp, s=90 if name == "ours" else 55, marker="o" if k == 0 else "s", color=col, zorder=3,
                      edgecolor=INK if name == "ours" else "none", lw=1,
                      label=lab if k == 0 else None)
    a.scatter([], [], marker="o", color=MUTED, label=T["db_mit"])
    a.scatter([], [], marker="s", color=MUTED, label=T["db_sv"])
    a.set(xlabel=T["se_ax"], ylabel=T["pp_ax"], title=T["a13"], xlim=(89.5, 100.2), ylim=(98.1, 100.05))
    a.legend(loc="lower left", fontsize=8.5)
    b = ax[1]
    rows = sorted(tables["mitdb"], key=lambda r: -(int(r["ours_fn"]) + int(r["ours_fp"])))
    names = [r["record"] for r in rows]
    ours = np.array([int(r["ours_fn"]) + int(r["ours_fp"]) for r in rows])
    nk = np.array([int(r["neurokit2_fn"]) + int(r["neurokit2_fp"]) for r in rows])
    beats = np.array([int(r["beats"]) for r in rows])
    x = np.arange(len(rows))
    b.bar(x - 0.2, 100 * ours / beats, width=0.4, color=RED, label=T["d_ours"])
    b.bar(x + 0.2, 100 * nk / beats, width=0.4, color=BLUE, label="NeuroKit2")
    b.set_xticks(x, names, rotation=90, fontsize=7)
    b.set(ylabel=T["err_ax"], title=T["b13"], xlim=(-0.8, len(rows) - 0.2))
    b.set_yscale("symlog", linthresh=0.1)
    b.set_yticks([0, 0.1, 1, 10, 50])
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: dec(lang, f"{v:g}")))
    b.legend(loc="upper right")
    fig.suptitle(T["f13_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "13_rpeak_benchmark.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["en", "ru"])
    ap.add_argument("--only", nargs="*", default=None, help="e.g. 05 06")
    ap.add_argument("--style", choices=sorted(STYLES), default="journal", help="figure palette and type")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"), help="folder for the en/ and ru/ subfolders")
    ap.add_argument("--checkpoint", default=str(RELEASED / "model.pt"),
                    help="weights for figures 08 and 09 (default: the released model)")
    args = ap.parse_args()
    apply_style(args.style)
    figs = {"01": fig01, "02": fig02, "03": fig03, "04": fig04, "05": fig05, "06": fig06,
            "07": fig07, "08": fig08, "09": fig09, "10": fig10, "11": fig11, "12": fig12, "13": fig13}
    for lang in args.lang:
        out = Path(args.out) / lang
        print(f"[{lang}]")
        # tick labels with a decimal comma in Russian; if the system lacks the locale, points stay
        try:
            locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8" if lang == "ru" else "C")
            plt.rcParams["axes.formatter.use_locale"] = lang == "ru"
        except locale.Error:
            plt.rcParams["axes.formatter.use_locale"] = False
        for k, fn in figs.items():
            if args.only and k not in args.only:
                continue
            fn(lang, out, args.checkpoint) if k in ("08", "09") else fn(lang, out)


if __name__ == "__main__":
    main()
