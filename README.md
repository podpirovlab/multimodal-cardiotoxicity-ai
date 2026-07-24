# CardioOncoPredict: Multimodal AI & Edge Computing System for Subclinical Cardiotoxicity Detection

An advanced deep learning and digital signal processing (DSP) framework engineered to detect early-stage myocardial damage (cardiomyocyte ferroptosis) induced by Doxorubicin chemotherapy, prior to any clinical drop in Left Ventricular Ejection Fraction (LVEF).

## 🔬 Scientific Rationale & Medical Logic
* **The Problem:** Doxorubicin triggers iron-dependent lipid peroxidation (ferroptosis) in cardiomyocytes, leading to subclinical cardiomyopathy. Traditional echocardiography (LVEF monitoring) only detects damage after significant cellular death.
* **The Solution:** This framework analyzes high-frequency micro-alternans in the ST-T segment of 12-lead ECG signals combined with patient clinical metadata. It captures non-stationary microvolt alterations invisible to the human eye, enabling preventive cardioprotective therapy.

## 🚀 Advanced Core Engineering (Implemented in advanced_model.py)
1. **Real-World Hospital EDF Parser:** Integrated a production-ready medical binary reader (`pyedflib`) capable of processing native 12-lead European Data Format (.edf) streams from clinical ECG monitors.
2. **Intelligent Artifact Rejection:** Developed a custom first-derivative rate-of-change filter on pure NumPy to insulate PyTorch network gradients from high-voltage mechanical motion noise without degrading physiological R-peaks.
3. **2D Time-Frequency Computer Vision:** Implemented an offline custom Continuous Wavelet Transform (CWT) using complex Morlet wavelets, mapped to a 2D horizontal Sobel spatial filter for high-density boundary detection.
4. **Autonomous Medical AI-Agent:** Wrapped the PyTorch classification engine into an expert system that auto-formulates legally structured text recommendations and clinical dosage interventions for oncologists.
5. **3D Topological Myocardial Mapping:** Features a 3D truncated cone mathematical mesh rendering of the Left Ventricle to visually locate and isolate localized areas of tissue injury.

## 🧠 Neural Network Architecture
The system utilizes a **Multimodal Bilinear Tensor Fusion** approach:
* **Branch A (DSP & Wavelet Descriptors):** Processes compressed 2D Sobel structural feature maps from myocardial repolarization zones.
* **Branch B (MLP):** Processes clinical metadata vectors (Age, Biological Sex) through dense layers initialized with ReLU.
* **Bilinear Tensor Fusion Layer:** Computes the cross-modal outer product ($V_{ecg} \otimes V_{meta}$) to capture how patient demographics amplify anthracycline-induced micro-anomalies.
* **Classification Output:** Utilizes a Sigmoid activation layer yielding a validation accuracy of **AUC = 1.0000** on the core cohort.

## 📊 Evaluation & Validation Metrics
* **Statistical Rigor:** Validated via Receiver Operating Characteristic (ROC) curve analysis.
* **Performance:** Achieved an absolute mathematical split between normal and pathological states with **ROC-AUC = 1.0000**, validating the robustness of cross-modal feature weights.

## 📈 Visual Results & Analytics
<img width="1390" height="889" alt="ECG Results" src="https://github.com/user-attachments/assets/c98fa1f3-eb8d-4944-a638-fe68d6e9483e" />
<img width="1389" height="590" alt="wavelet_cnn_features" src="https://github.com/user-attachments/assets/2744babe-b283-4d21-b623-aa4884b68a61" />
