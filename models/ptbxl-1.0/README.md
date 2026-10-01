# CardioOncoNet trained on PTB-XL (model card)

**What it is.** The network of README §5 (1D ResNet over 12 leads, age/sex branch, bilinear fusion;
573,482 weights) trained once on PTB-XL v1.0.3 at 100 Hz to recognise the five diagnostic
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

**Ablation.** The same network with age and sex set to zero (`--no-meta`, `metrics_nometa.json`)
reached a test macro-AUC of 0.920. On the same 2,158 test ECGs the difference is +0.0007, paired
bootstrap (2,000 resamples) 95% CI [−0.003, +0.004], and every per-class CI includes zero (`comparison_meta_vs_nometa.json`, from
`scripts/compare_ptbxl_runs.py`): no measurable benefit from the metadata on this task. Each model
was trained once, so the spread between random seeds is not measured.

**Files.**

| File | What | SHA-256 |
|---|---|---|
| `model.pt` | PyTorch checkpoint with normalisation statistics | `5d25e587a3bec4cc553a299043428a428220c8592dfb69962d2c589b452e47f5` |
| `model.onnx` | the same network for ONNX Runtime (inputs `ecg` 12×1000, `meta` 3) | `8aa31b0d08da3082e04609390255a2421981186d1087a1cb57b8d5fd72b2ac3a` |
| `norm_stats.npz` | per-lead mean and standard deviation used to standardise the input | |
| `metrics.json`, `metrics_nometa.json` | full metrics and training history of both runs | |
| `comparison_meta_vs_nometa.json` | paired bootstrap comparison | |
| `roc_test.png` | ROC curves on the test fold | |

`predict.py --checkpoint models/ptbxl-1.0/model.pt` runs the model on a 12-lead ECG.

**Licence and attribution.** The weights are derived from PTB-XL, which is distributed by PhysioNet
under CC BY 4.0, so they are shared under CC BY 4.0 as well. Cite Wagner P, Strodthoff N,
Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset.
*Scientific Data* 2020;7:154, and Goldberger AL, et al. *Circulation* 2000;101:e215–e220.
