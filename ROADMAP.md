# CardioOncoPredict roadmap

Goal: turn a teaching prototype into a tool that genuinely helps detect heart injury earlier in patients receiving chemotherapy. The path runs from honest mathematics, through real data, to a clinical study. Every phase ends with a result that can be checked, not a promise.

[Русская версия](ROADMAP.ru.md) · [README](README.md)

---

## Where we are (version 0.3, September 2026)

| Done | Not done |
|---|---|
| TWA algorithm (Spectral Method + MMA) with tests | Training the network on real ECGs |
| The same maths in the browser, parity with Python tested | Validating TWA on real long recordings |
| Tensor-fusion network and a complete PTB-XL training script | Data from patients receiving chemotherapy |
| Valid FHIR report, ONNX export, CI | Clinical partner, ethics approval |

---

## Phase 0: this week (UWC application)

1. **Push version 0.3 to GitHub** and check that CI is green and the website loads.
2. **Train on PTB-XL** on your own Mac:
   ```bash
   bash scripts/download_ptbxl.sh
   python train_ptbxl.py --data data/ptb-xl --epochs 30 --export-onnx
   python train_ptbxl.py --data data/ptb-xl --epochs 30 --no-meta --out runs/ptbxl_nometa
   cp runs/ptbxl/metrics.json assets/metrics.json
   ```
   The second run is an **ablation**: the same network without age and sex. Comparing AUCs answers whether bilinear fusion helps. An honest negative answer is also a result.
3. **Put the real numbers** into README section 5.4 and regenerate figure 08 with trained weights.
4. **Send README.md to the professor** with a short note: what was built, what was verified, and one concrete question (for example: "Is it realistic to obtain de-identified ECGs from patients before and after anthracyclines?").
5. **UWC achievements entry:** one sentence on the problem, one on the method, one on what has been verified, and the website link. Say "research prototype", not "a system that saves lives".

**Done when:** the website shows real AUCs with confidence intervals and all tests pass.

---

## Phase 1: real data (October–December 2026)

### 1.1 Validate TWA on real recordings
- **T-Wave Alternans Challenge Database** (PhysioNet, Computing in Cardiology Challenge 2008): 100 records with alternans of varying magnitude and a reference ranking. Compute V_alt with our method and compare ranks with the reference (Spearman correlation).
- Long Holter-type recordings on PhysioNet: test the R-peak detector on real artefacts, ectopic beats and changing heart rate.
- **Success metric:** Spearman correlation with the reference ≥ 0.7; R-peak detector sensitivity and precision ≥ 99% on annotated records.

### 1.2 Make the network stronger and more honest
- **External validation:** train on PTB-XL, test on a different open 12-lead dataset (for example from the PhysioNet/CinC Challenge 2020). The drop in AUC on unseen data is the key measure of robustness.
- **Calibration:** reliability diagram and temperature scaling, so that "probability 0.8" means 80%.
- **Uncertainty:** an ensemble of 5 models or MC dropout; say "uncertain" when the models disagree.
- **Explainability:** integrated gradients over time show which parts of the ECG drive a decision. For STTC we expect the ST-T window; if not, that is an error to understand.
- **In-browser inference:** onnxruntime-web on the website, so an ECG file never leaves the user's computer.

### 1.3 Features cardiologists already use
- QTc (Bazett, Fridericia), T-wave amplitude and area, summed QRS voltage. Reduced QRS voltage has been described in anthracycline cardiomyopathy.
- Add them as a third branch of the model and repeat the ablation.

---

## Phase 2: clinical partner (2027)

Without clinicians and real patients the project cannot answer its main question. This is the most important and the hardest phase.

1. **Find a cardio-oncology mentor** through the professor and the school: a national cardiology or oncology research centre, or a university hospital.
2. **Retrospective study.** Patients treated with anthracyclines who have an ECG before treatment, an ECG during or after treatment, and echocardiography.
   - Endpoint: cardiac dysfunction by the 2022 ESC criteria (LVEF fall, GLS, troponin rise).
   - Size: at least 200–300 patients. At about 9% incidence that gives 20–30 events; fewer would make confidence intervals too wide.
   - **Key modelling idea for this phase:** compare each patient's ECG *with their own baseline ECG* (a Siamese network, or "after minus before" features). Each patient is their own control, which removes differences between people.
3. **Ethics and data:** ethics-committee approval, de-identification before transfer, data kept inside the hospital (the federated-learning code in this repository is meant for this).
4. **Pre-registered analysis plan:** hypothesis, metrics and thresholds are written down before looking at the data.

**Done when:** a signed agreement with a clinic, ethics approval and an analysis plan exist.

---

## Phase 3: device (2027–2028)

- **Requirement from the detection map (figure 06):** input noise ≤ 20 µV RMS. That determines the analogue front end: dedicated 24-bit biopotential ADCs rather than the cheapest modules.
- Holter prototype: 24-hour recording, TWA computed in windows of equal heart rate (alternans depends on heart rate, README section 3.5).
- INT8 model on a phone via ONNX; compare float32 and INT8 accuracy on the test set.
- FHIR report tested against a public FHIR test server (for example HAPI FHIR).

---

## Phase 4: prospective study and publication (2028+)

- Follow patients during chemotherapy: an ECG before every cycle, compared with standard monitoring (echo + troponin). Question: does TWA or the model detect injury earlier?
- Preprint (medRxiv / arXiv), a paper at Computing in Cardiology, open code and reproducible results.
- Continue at university: medical engineering, signal processing, machine learning for medicine.

---

## Technical backlog

- [ ] Run `train_ptbxl.py` and publish the metrics (Phase 0)
- [ ] Ablation without metadata (`--no-meta`)
- [ ] Validate TWA on the T-Wave Alternans Challenge Database
- [ ] External validation of the network on a second dataset
- [ ] Calibration and uncertainty estimates
- [ ] Integrated gradients to explain decisions
- [ ] onnxruntime-web: the real model in the browser
- [ ] QTc, T-wave amplitude and QRS voltage features
- [ ] "Patient versus own baseline" model (Siamese network)
- [ ] FHIR validation on HAPI FHIR
- [ ] Measure INT8 accuracy

---

## Risks to keep in mind

| Risk | Response |
|---|---|
| TWA turns out not to be an early marker | That is also a scientific result. The raw-ECG model and QTc / QRS-voltage features are tested in parallel |
| Clinical data cannot be obtained | Approach several partners; start with open data and publications on the method |
| Overfitting on a small cohort | Pre-registered analysis plan, external validation, simple models as baselines |
| Temptation to overstate results | Every claim in the README is tied to a test or a figure; open items are marked ⏳ or ❌ |
