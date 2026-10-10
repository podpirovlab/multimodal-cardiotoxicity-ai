# CardioOncoPredict roadmap

Goal: turn a teaching prototype into a tool that genuinely helps detect heart injury earlier in patients receiving chemotherapy. The path runs from honest mathematics, through real data, to a clinical study. Every phase ends with a result that can be checked, not a promise.

[Русская версия](ROADMAP.ru.md) · [README](README.md)

---

## Where we are (version 1.4, October 2026)

| Done | Not done |
|---|---|
| TWA over the whole recording (Spectral Method + MMA), three outcomes, tests | A TWA result that agrees with a reference on real ECGs |
| The same maths in the browser, parity with Python tested in CI | Validation of the network on a second, independent dataset |
| Checked on the PhysioNet 2008 challenge with a held-out protocol: τ = 0.43 overall, 0.08 on held-out real records (README §7.1) | Data from patients receiving chemotherapy |
| Network trained on PTB-XL: test macro-AUC 0.921, the published level; age/sex fusion gives no measurable gain (README §5.4) | Clinical partner, ethics approval |
| Valid research FHIR export, installable package, CI, DOI, offline web tool with a real sample recording | |

---

## Next steps (autumn 2026)

1. ~~Train on PTB-XL and run the ablation without age and sex~~ — done (README §5.4): 0.921 against 0.920, difference +0.0007 [−0.003, +0.004]. Each model was trained once.
2. ~~Repeat both runs with five random seeds~~ — done (README §5.4): 0.9205 ± 0.0015 with age/sex, 0.9192 ± 0.0012 without; paired effect +0.0013 [−0.0005, +0.0032]. The five models' ensemble reaches 0.930 and is released.
3. **Find recordings with a known answer** for TWA (paced or exercise tests with clinical TWA results), because the 2008 reference is a consensus of other algorithms, not measured truth.
4. **Find a cardio-oncology mentor** with one concrete question, for example: "Is it realistic to obtain de-identified ECGs from patients before and after anthracyclines?"
5. ~~Release v1.0 with a DOI~~ — done: [10.5281/zenodo.23090356](https://doi.org/10.5281/zenodo.23090356) (1.0.0); 1.1.0 with the trained model: [10.5281/zenodo.23094298](https://doi.org/10.5281/zenodo.23094298); 1.1.1: [10.5281/zenodo.23144064](https://doi.org/10.5281/zenodo.23144064); 1.2.0, the new R-peak detector: [10.5281/zenodo.23238175](https://doi.org/10.5281/zenodo.23238175); 1.3.0, coarse sampling: [10.5281/zenodo.23264684](https://doi.org/10.5281/zenodo.23264684); 1.4.0, measured figures and the ensemble: [10.5281/zenodo.23288316](https://doi.org/10.5281/zenodo.23288316).

**Done when:** a TWA data source with a known answer is identified.

---

## Phase 1: real data (October–December 2026)

### 1.1 Validate TWA on real recordings
- **T-Wave Alternans Challenge Database** (PhysioNet, Computing in Cardiology Challenge 2008): 100 records with alternans of varying magnitude and a reference ranking. **Done** (README §7.1): Kendall τ = 0.43 over all records, just below the organisers' significance line of 0.436; 0.48 on the simulated records; 0.08 on held-out real records, which is no better than chance.
- **R-peak detector on annotated records. Done** (README §4.3): 99.73% sensitivity and 99.92% precision on the MIT-BIH Arrhythmia Database, 99.50% / 99.93% on the 128 Hz Supraventricular Arrhythmia Database. Next: long Holter-type recordings with real artefacts and changing heart rate, and the MIT-BIH Noise Stress Test.
- **Success metric:** Kendall τ with the challenge reference above 0.436 and clearly above chance on real records (not yet met); R-peak detector sensitivity and precision ≥ 99% on annotated records (met in 1.2).

### 1.2 Make the network stronger and more honest
- **External validation:** train on PTB-XL, test on a different open 12-lead dataset (for example from the PhysioNet/CinC Challenge 2020). The drop in AUC on unseen data is the key measure of robustness.
- **Calibration:** measured (README §5.4): ECE 0.04–0.06 as trained, 0.01–0.02 after Platt scaling on validation. Next: ship the recalibrated probabilities in `predict.py`.
- **Uncertainty:** the 5-model ensemble is released (macro-AUC 0.930); next, report "uncertain" when its members disagree (their SD is 0.20 or more for 5% of predictions).
- **Explainability:** integrated gradients on two test ECGs (README figure 08) put an STTC decision mostly on QRS edges, not on the ST-T window. Next: attributions averaged over every STTC test ECG, to tell a shortcut from a single example.
- **In-browser inference:** onnxruntime-web on the website, so an ECG file never leaves the user's computer.

### 1.3 Features cardiologists already use
- QTc (Bazett, Fridericia), T-wave amplitude and area, summed QRS voltage. Reduced QRS voltage has been described in anthracycline cardiomyopathy.
- Add them as a third branch of the model and repeat the ablation.

---

## Phase 2: clinical partner (2027)

Without clinicians and real patients the project cannot answer its main question. This is the most important and the hardest phase.

1. **Find a cardio-oncology mentor** through school and university contacts: a national cardiology or oncology research centre, or a university hospital.
2. **Retrospective study.** Patients treated with anthracyclines who have an ECG before treatment, an ECG during or after treatment, and echocardiography.
   - Endpoint: cardiac dysfunction by the 2022 ESC criteria (LVEF fall, GLS, troponin rise).
   - Size: at least 200–300 patients. At about 9% incidence [Cardinale 2015, README ref. 2] that gives 18–27 events; fewer would make confidence intervals too wide.
   - **Key modelling idea for this phase:** compare each patient's ECG *with their own baseline ECG* (a Siamese network, or "after minus before" features). Each patient is their own control, which removes differences between people.
3. **Ethics and data:** ethics-committee approval, de-identification before transfer, data kept inside the hospital. Federated learning would allow that; the FedAvg script in `legacy/` only shows the idea, and a real study would need an established framework.
4. **Pre-registered analysis plan:** hypothesis, metrics and thresholds are written down before looking at the data.

**Done when:** a signed agreement with a clinic, ethics approval and an analysis plan exist.

---

## Phase 3: device (2027–2028)

- **Requirement from the detection map (README §4.5):** input noise of roughly 20 µV RMS or less under a white-noise model; to be confirmed on real recordings. That determines the analogue front end: dedicated 24-bit biopotential ADCs rather than the cheapest modules.
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

- [x] Run `train_ptbxl.py` and publish the metrics (macro-AUC 0.921, `models/ptbxl-1.0/`)
- [x] Ablation without metadata (`--no-meta`): no measurable gain
- [x] Five seeds per configuration, and their ensemble (0.930)
- [x] Validate TWA on the T-Wave Alternans Challenge Database (τ = 0.43; real records not yet in agreement)
- [x] R-peak detector scored on cardiologist-annotated databases (MIT-BIH, SVDB)
- [ ] R-peak detector under controlled noise (MIT-BIH Noise Stress Test)
- [ ] TWA recordings with a known answer (paced or exercise tests)
- [ ] The clinical "sustained for a minute" rule instead of a single window
- [ ] External validation of the network on a second dataset
- [x] Calibration measured (Platt scaling: ECE 0.06 → 0.015)
- [ ] Recalibrated probabilities and an "uncertain" flag in `predict.py`
- [x] Integrated gradients on two test ECGs
- [ ] Attributions over all STTC test ECGs (shortcut check)
- [ ] onnxruntime-web: the real model in the browser
- [ ] QTc, T-wave amplitude and QRS voltage features
- [ ] "Patient versus own baseline" model (Siamese network)
- [ ] FHIR validation on HAPI FHIR
- [x] INT8 accuracy measured: −0.0002 [−0.0008, +0.0003], 3.4× smaller

---

## Risks to keep in mind

| Risk | Response |
|---|---|
| TWA turns out not to be an early marker | That is also a scientific result. The raw-ECG model and QTc / QRS-voltage features are tested in parallel |
| Clinical data cannot be obtained | Approach several partners; start with open data and publications on the method |
| Overfitting on a small cohort | Pre-registered analysis plan, external validation, simple models as baselines |
| Temptation to overstate results | Every claim in the README is tied to a test or a figure; open items are marked ⚠️ or ❌ |
