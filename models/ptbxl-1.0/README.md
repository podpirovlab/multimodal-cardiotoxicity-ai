# CardioOncoNet trained on PTB-XL (model card)

**What it is.** The network of README §5 (1D ResNet over 12 leads, age/sex branch, bilinear fusion;
573,482 weights) trained on PTB-XL v1.0.3 at 100 Hz to recognise the five diagnostic
super-classes NORM, MI, STTC, CD and HYP. PTB-XL has no chemotherapy information: this is a
check that the pipeline works on real clinical ECGs, and **not** a detector of cardiotoxicity.
Research use only; not a medical device.

**How it was trained.** `python train_ptbxl.py --data data/ptb-xl --epochs 30 --export-onnx`
on an Apple Silicon Mac (Metal, `mps`), seed 42, official patient-wise folds: 1–8 for training
(17,084 ECGs), 9 for validation (2,146), 10 for testing (2,158); records without a diagnostic
super-class label are left out, as in the published benchmarks. The best epoch on validation
macro-AUC was 16; training stopped early at 24.

**Results on the untouched test fold** (`metrics.json`, bootstrap 95% CI over 1,000 resamples;
thresholds chosen on validation by Youden's J):

| Class | AUC [95% CI] | Sensitivity | Specificity |
|---|---|---|---|
| NORM | 0.943 [0.934, 0.951] | 0.92 | 0.82 |
| MI | 0.917 [0.906, 0.929] | 0.81 | 0.86 |
| STTC | 0.934 [0.923, 0.946] | 0.87 | 0.83 |
| CD | 0.915 [0.899, 0.932] | 0.82 | 0.89 |
| HYP | 0.895 [0.874, 0.915] | 0.86 | 0.76 |
| **macro** | **0.921** | | |

**Five seeds and the ensemble.** The same configuration was trained with four more random seeds
(1–4; `scripts/train_seeds.sh`), with and without age and sex, and every model scored the same
test ECGs (`scripts/compare_seeds.py`, `ensemble/ensemble_metrics.json`):

| | Macro-AUC over 5 seeds | Range |
|---|---|---|
| With age/sex (seed 42 released) | 0.9205 ± 0.0015 (SD) | 0.9182–0.9218 |
| Without age/sex | 0.9192 ± 0.0012 | 0.9176–0.9202 |
| **Ensemble: mean probability of the 5 with-age/sex models** | **0.9303** | |

- *Age and sex:* paired by seed, +0.0013, 95% CI [−0.0005, +0.0032] (pooled paired bootstrap over test
  ECGs), better in 4 of 5 seeds. The effect, if real, is small; it is not distinguishable from zero.
- *Ensemble:* +0.0098 over the mean single model, 95% CI [+0.0090, +0.0105]; per class NORM 0.950,
  MI 0.931, STTC 0.941, CD 0.922, HYP 0.907. The members disagree by a median standard deviation of
  0.04 in probability (95th percentile 0.20), a measure of the model's uncertainty for each ECG.

**Calibration.** As trained, the probabilities are too high for positives, because the loss weights
rare classes up: the expected calibration error (ECE, 10 bins) is 0.043–0.064 per class. Platt scaling
fitted on the validation fold lowers it to 0.011–0.018 and the Brier score from 0.082–0.104 to
0.068–0.097, with AUC unchanged (`docs/results/ptbxl_evaluation.json`). The released weights are not
recalibrated; the fitted slopes and intercepts are in that file.

**INT8.** The ONNX copy, quantised with ONNX Runtime (static QDQ, per-channel 8-bit weights, 8-bit
activations, calibrated on 512 training ECGs) is 0.70 MB instead of 2.39 MB. Its macro-AUC is
0.9206 against 0.9208, difference −0.0002, 95% CI [−0.0008, +0.0003]; the median probability
changes by 0.003, the largest by 0.28 (`scripts/evaluate_ptbxl_model.py`).

**Files.**

| File | What | SHA-256 |
|---|---|---|
| `model.pt` | PyTorch checkpoint with normalisation statistics | `5d25e587a3bec4cc553a299043428a428220c8592dfb69962d2c589b452e47f5` |
| `model.onnx` | the same network for ONNX Runtime (inputs `ecg` 12×1000, `meta` 3) | `8aa31b0d08da3082e04609390255a2421981186d1087a1cb57b8d5fd72b2ac3a` |
| `norm_stats.npz` | per-lead mean and standard deviation used to standardise the input | |
| `metrics.json`, `metrics_nometa.json` | full metrics and training history of both runs | |
| `comparison_meta_vs_nometa.json` | paired bootstrap comparison | |
| `roc_test.png` | ROC curves on the test fold | |
| `ensemble/seed1.pt` | ensemble member, seed 1 | `2d693ebfca11ee494408f0265ba888b0da377cc34aef43d80986acb9abe4f0ea` |
| `ensemble/seed2.pt` | ensemble member, seed 2 | `a5937893c7ad4a2e0ddce82df7e90ee11c085813328a782b3680f53f2f7686ad` |
| `ensemble/seed3.pt` | ensemble member, seed 3 | `2a6494db5eaad5a61ce5dce3c46b09d95b3a11b8260f94f805dc3283d6e89196` |
| `ensemble/seed4.pt` | ensemble member, seed 4 | `f759ecd46fc08ad508483c130ff8a176f3ddaeebf5dd3612bb2b891f976766f5` |
| `ensemble/*_metrics.json` | metrics of each member and of the ensemble | |

`predict.py --checkpoint models/ptbxl-1.0/model.pt` runs the model on a 12-lead ECG, and
`predict.py --ensemble` runs all five and averages them; the recording is resampled to 100 Hz and
its first 10 seconds are used, as in training.

**Use and limits.**
- *Intended use:* research and teaching — a reproducible baseline for ECG classification and the
  starting point for fine-tuning on cardio-oncology data.
- *Not for:* diagnosis, screening or any decision about a patient, and not for cardiotoxicity,
  which it was never trained to see.
- *Data:* one German centre (PTB-XL), labels from at most two cardiologists, 10-second resting ECGs
  at 100 Hz. Accuracy on other populations, devices and sampling rates is unknown until an external
  dataset is tested.
- *What it looks at:* integrated gradients on two test ECGs (README figure 08) put most of the
  weight of an STTC prediction on the edges of the QRS complexes rather than on the ST-T segment
  itself, so the network may rely on features that accompany ST-T change rather than on the change.
- *Not measured yet:* accuracy on an external dataset, and attributions over many ECGs rather than
  two examples.

**Licence and attribution.** The weights are derived from PTB-XL, which is distributed by PhysioNet
under CC BY 4.0, so they are shared under CC BY 4.0 as well. Cite Wagner P, Strodthoff N,
Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset.
*Scientific Data* 2020;7:154, and Goldberger AL, et al. *Circulation* 2000;101:e215–e220.
