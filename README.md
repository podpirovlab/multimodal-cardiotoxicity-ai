# Multimodal AI System for Subclinical Anthracycline-Induced Cardiotoxicity Detection

An advanced deep learning framework implementing 1D Convolutional Neural Networks (1D-CNN) and Multi-Layer Perceptrons (MLP) with Bilinear Tensor Fusion to detect early-stage myocardial damage caused by Doxorubicin chemotherapy, prior to Left Ventricular Ejection Fraction (LVEF) drop.

## 🔬 Scientific Rationale & Medical Logic
* **The Problem:** Doxorubicin triggers iron-dependent lipid peroxidation (ferroptosis) in cardiomyocytes, leading to subclinical cardiomyopathy. Traditional echocardiography (LVEF monitoring) only detects damage after significant cellular death, rendering early therapeutic interventions ineffective.
* **The Solution:** This AI framework analyzes high-frequency micro-alternans in the ST-T segment of ECG signals combined with patient clinical metadata (age, cumulative dosage). It captures non-stationary microvolt alterations invisible to the human eye, enabling preventative cardioprotective therapy.

## 🧠 Neural Network Architecture
The system utilizes a **Multimodal Bilinear Fusion** approach:
1. **Branch A (1D-CNN):** Extracts spatial-temporal feature maps from raw digitized ECG signals using custom 1-Dimensional convolutional kernels optimized for QRST-complex morphology.
2. **Branch B (MLP):** Processes clinical metadata vectors ($X_{meta}$) through dense layers initialized with Rectified Linear Units (ReLU) to model non-linear patient risk groups.
3. **Bilinear Tensor Fusion Layer:** Computes the outer product ($V_{ecg} \otimes V_{meta}$) to map cross-modal feature interactions, capturing how age and cumulative toxicity intensify ECG micro-anomalies.
4. **Output Stage:** Utilizes a Sigmoid activation layer coupled with Binary Cross-Entropy (BCE) Loss for precise probabilistic classification.

## 📊 Core Features Implemented
* Autonomous synthetic PTB-XL aligned database generator.
* Digital Signal Processing (DSP) bandpass filtration kernels.
* Mathematical 1D-Convolution vector mapping.
* Bilinear interaction matrix visualization.
## 📈 Visual Results
<img width="1390" height="889" alt="ECG Results" src="https://github.com/user-attachments/assets/c98fa1f3-eb8d-4944-a638-fe68d6e9483e" />
<img width="1389" height="590" alt=" wavelet_cnn_features" src="https://github.com/user-attachments/assets/2744babe-b283-4d21-b623-aa4884b68a61" />
