# Early teaching prototypes

These four scripts are the project's first steps, kept for history. They are not part of
the tested library and nothing in `cardioonco/`, the web tool or the tests depends on them.

| Script | What it shows | Replaced by |
|---|---|---|
| `main_model.py` | bilinear fusion of ECG and patient features, step by step in NumPy | `cardioonco/model.py` |
| `advanced_model.py` | first PyTorch model, EDF reader, a 3D mesh demo | `cardioonco/model.py`, `assets/js/edf.js` |
| `export_edge_onnx.py` | ONNX export and INT8 quantisation of a toy network | `train_ptbxl.py --export-onnx` |
| `federated_fhir_core.py` | federated averaging (FedAvg) and an early FHIR example | `cardioonco/fhir.py` |

All data inside them is synthetic, and their networks are untrained. Run them from the
repository root, for example `python legacy/main_model.py`.
