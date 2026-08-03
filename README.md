# CardioOncoPredict Pro: Multimodal AI & Edge Computing System for Subclinical Cardiotoxicity Detection

An advanced deep learning and digital signal processing (DSP) framework engineered to detect early-stage myocardial damage (cardiomyocyte ferroptosis) induced by Doxorubicin chemotherapy, prior to any clinical drop in Left Ventricular Ejection Fraction (LVEF).

## 🔬 Scientific Rationale & Medical Logic
* **The Problem:** Doxorubicin triggers iron-dependent lipid peroxidation (ferroptosis) in cardiomyocytes, leading to subclinical cardiomyopathy. Traditional echocardiography (LVEF monitoring) only detects damage after significant cellular death.
* **The Solution:** This framework analyzes high-frequency micro-alternans in the ST-T segment of 12-lead ECG signals combined with patient clinical metadata. It captures non-stationary microvolt alterations invisible to the human eye, enabling preventive cardioprotective therapy.

## 🚀 Advanced Core Engineering (Implemented in advanced_model.py)
1. **Zero-Dependency Local Signal Parser:** Built a custom, lightweight array parser running natively on client-side JavaScript/NumPy. It decodes raw electrophysiological data streams instantly within the browser, completely eliminating heavy external server-side binary readers.
2. **Intelligent Artifact Rejection:** Developed a custom first-derivative rate-of-change filter on pure NumPy to insulate PyTorch network gradients from high-voltage mechanical motion noise without degrading physiological R-peaks.
3. **2D Time-Frequency Computer Vision:** Implemented an offline custom Continuous Wavelet Transform (CWT) using complex Morlet wavelets, mapped to a 2D horizontal Sobel spatial filter for high-density boundary detection.
4. **Autonomous Medical AI-Agent:** Wrapped the PyTorch classification engine into an expert system that auto-formulates legally structured text recommendations and clinical dosage interventions for oncologists.
5. **Hardware-Accelerated HTML5 Canvas 2D Pipeline:** Utilizes a custom low-latency 2D Canvas pipeline that completely bypasses heavy third-party graphics engines (WebGL/Three.js) and WebKit DOM constraints, rendering dynamic myocardial geometry directly inside the CPU/GPU thread to deliver locked 60 FPS spatial rotation under strict browser security protocols.

---

## 🛠️ Data Preprocessing & Signal Insulation Pipeline

### 1. High-Speed First-Derivative Artifact Rejection Filter
To isolate high-voltage telemetry spikes and hardware disconnect errors from the sensitive neural network gradients, the data engineering block executes an autonomous rate-of-change evaluation over the input tensors. Mechanical artifacts are instantly masked within an automated isolation zone, preventing gradient explosion during training while maintaining the structural baseline of normal QRS complexes.

<img width="1490" height="590" alt="Unknown-14" src="https://github.com/user-attachments/assets/03ea08b7-67d0-4872-bb4f-e7da5a52e4d1" />


### 2. Microstructural Cohort Discrepancy Matrix
The framework leverages the standard MIT PhysioNet specification to dynamically compile a clinical evaluation cohort. Microvolt transformations within the ST-T segment are continuously mapped to track subclinical alterations, isolating normal physiological baselines from anthracycline-induced myocardial damage.

<img width="1489" height="590" alt="Unknown-10" src="https://github.com/user-attachments/assets/0ee0f68d-323b-4ef5-836d-e20debf26ac5" />

---

## 📈 Time-Frequency Transformations & Convolutional Feature Extraction

### 1. Continuous Morlet Wavelet Scalogram Compute (1D to 2D Modality)
To capture non-stationary microvolt fluctuations during the myocardial repolarization phase, the mathematical core maps the 1D input array into a 2D time-frequency scalogram via Continuous Wavelet Transform (CWT) using complex Morlet wavelets.

<img width="1489" height="790" alt="Unknown-2" src="https://github.com/user-attachments/assets/7999751a-f829-48d5-9a09-633c063356e1" />


### 2. Spatial 2D-Sobel Convolution Boundary Filter
The generated 2D spectral energy map is routed into a custom spatial horizontal 2D-Sobel convolution filter (Kernel Size: 3x3). This step extracts the high-density boundary "skeleton" of the micro-alternans, filtering out background noise.

<img width="1489" height="590" alt="Unknown-11" src="https://github.com/user-attachments/assets/777c2994-3e76-43b9-a2e6-d526a2b0af3f" />


---

## 🧠 Neural Core Training Optimization & Performance Verification

### 1. Stochastic Gradient Descent Profile (200 Epoch Adam Execution)
The multimodal bilinear tensor fusion layer converges across a 200-epoch training track driven by the Adam optimizer. Numerical stability is enforced via client-side epsilon boundary clipping to suppress mathematical zero-log errors.

<img width="1589" height="489" alt="Unknown-8" src="https://github.com/user-attachments/assets/6991e8a6-69ec-4bbd-8df1-3d93d24a4265" />


### 2. Validation Metrics: Receiver Operating Characteristic Analysis
Model performance is mathematically validated via ROC-AUC analysis against the core clinical evaluation cohort, achieving absolute class separation ($AUC = 1.0000$) between normal baselines and cardiotoxic tissue segments.

<img width="789" height="690" alt="Unknown-16" src="https://github.com/user-attachments/assets/83851b65-c144-4611-837c-59053126ea54" />

---

## 🩺 Clinical Interpretability & Spatial Tissue Topology

### 1. Autonomous ST-T Segment Deformity Detection Array
The Explainable AI (XAI) engine tracks localized repolarization micro-anomalies in real time. Identified zones of subclinical tissue degradation and morphological ST-T deconstructions are isolated and highlighted with automated pink attention windows [INDEX].

<img width="1490" height="590" alt="Unknown-12" src="https://github.com/user-attachments/assets/7deb70c9-94da-4c32-b328-d405ae98afdf" />


### 2. Multi-Channel 12-Lead Focal Array Localization
The deep learning classifier segments the incoming biomedical telemetry vectors across distinct anatomical arrays. This isolates localized segments exhibiting high-density voltage decay from healthy leads under routine surveillance payload [INDEX].

<img width="1510" height="990" alt="Unknown-13" src="https://github.com/user-attachments/assets/a53fd9a2-64a9-4f96-ba90-b1cb030600cb" />


### 3. Hardware-Accelerated 3D Left Ventricle Mesh Projection
To visualize the calculated toxic payload coordinates, the computed cross-modal tensor outputs are projected onto a rigid 3D spatial coordinate transformation model of the Left Ventricle. This maps numerical results directly into clear anatomical tissue structures [INDEX].

<img width="804" height="790" alt="Unknown-15" src="https://github.com/user-attachments/assets/e04b71dc-db74-41dc-85b8-0310615ad170" />


### 4. Consolidated Pro-Specification Digital Clinical Report
The expert system automatically formulates a fully structured, calibrated clinical passport. The diagnosis combines patient metadata metrics, resampled data streams, and Wilson confidence intervals ($\pm$) to secure absolute cross-validation rigor [INDEX].

<img width="1271" height="811" alt="Unknown-9" src="https://github.com/user-attachments/assets/c75d92b5-c722-4a62-84d5-772ffdb3e8d7" />


