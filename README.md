# 🩺 CardioOncoPredict: Multimodal Edge AI System for Subclinical Anthracycline Cardiotoxicity Detection

### 🔗 Project Ecosystem & Access Links
* **Live Clinical Simulation Dashboard:** https://github.io
* **Production Cloud Backend (PyTorch Core & Server API):** https://railway.app
* **Open-Source Research Workbook (Google Colab):** https://google.com

---

## 🔬 1. Scientific Rationale & Pathophysiological Mechanism

### The Clinical Problem
Doxorubicin and other anthracycline chemotherapies trigger cumulative, dose-dependent cardiomyocyte destruction via a specialized, non-apoptotic programmed cell death pathway known as **ferroptosis** [INDEX]. Conventional diagnostic protocols, such as Echocardiography, are highly reactive: they measure declines in Left Ventricular Ejection Fraction (LVEF) only *after* irreversible transmural cellular necrosis has taken place [INDEX].

### The Predictive AI Solution
CardioOncoPredict shifts the clinical oncology paradigm from reactive management to proactive subclinical intervention [INDEX]. By targeting the electrophysiological **ST-T segment (ventricular repolarization phase)** of 12-lead ECG streams [INDEX], the system isolates microvolt-level structural alterations—**T-wave alternans (TWA)**—that are entirely invisible to the human eye [INDEX], capturing membrane degradation before macroscopic myocardial mechanical failure occurs [INDEX].

[Doxorubicin Infiltration] ➔ [Mitochondrial Iron Accumulation (Fe²⁺)] ➔ [Fenton Reaction]│▼[T-Wave Alternans on ECG] ➔ [I_Kr Potassium Channel Failure] ➔ [Lipid Peroxidation (LOOH)]
---

## 📐 2. Advanced Mathematical Engineering & Data Pipeline

The production core architecture ingests, purifies, analyzes, and visualizes raw biomedical telemetry via an isolated 5-stage deterministic computational pipeline [INDEX]:

[1. EDF/WFDB Stream] ➔ [2. CWT Morlet Decomp] ➔ [3. Bilinear Tensor Fusion] ➔ [4. Tikhonov Loss] ➔ [5. 3D Projector]
### Step 1: Discrete Binary Stream Sifting & Calibration
Clinical voltage metrics are ingested as 16-bit signed integer (`int16`) Little-Endian streams in compliance with the **PhysioNet PTB-XL** data specification [INDEX]. Amplitudes are demultiplexed and scaled into physical millivolts ($V$) via calibrated gain factors:
$$\text{Voltage}(t) = \frac{X_{\text{raw}}(t) - \text{Baseline}}{\text{Gain}}$$
At a standard sampling frequency ($f_s$) of $250\text{ Hz}$ [INDEX], the discrete step resolution maps to a tight interval of $\Delta t = 1/f_s = 0.004\text{ seconds}$ [INDEX].

### Step 2: Time-Frequency Multi-Scale Decomposition (Complex Morlet CWT)
To capture volatile, non-stationary microvolt fluctuations within the repolarization zone [INDEX], the 1D time-series data array is transformed into a 2D spectral energy scalogram using a Continuous Wavelet Transform (CWT) [INDEX]. The mathematical engine applies a complex Morlet wavelet basis in $L^2(\mathbb{R})$ space to enforce minimal Heisenberg uncertainty box limits ($\Delta t \cdot \Delta \omega \geq \frac{1}{2}$):
$$\psi(t) = \pi^{-1/4} e^{i\omega_0 t} e^{-\frac{t^2}{2}}$$
The continuous convolution resolves across vector hardware registers via optimized Riemann summation:
$$W(a, b) = \frac{1}{\sqrt{a}} \sum_{n=0}^{N-1} x[n] \cdot \psi^*\left(\frac{n\Delta t - b}{a}\right)$$
