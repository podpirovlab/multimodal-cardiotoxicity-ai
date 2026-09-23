"""CardioOncoPredict core library.

Modules
-------
synth       Parametric synthetic ECG generator (sum of Gaussians) with controllable
            microvolt T-wave alternans, noise, baseline wander and mains hum.
preprocess  Zero-phase band-pass / notch filtering and Pan-Tompkins-style R-peak detection.
twa         T-wave alternans quantification: Spectral Method (K-score, V_alt) and
            Modified Moving Average (MMA).
model       Multimodal 1D-CNN + clinical-metadata network with bilinear (tensor) fusion.
fhir        HL7 FHIR R4 DiagnosticReport builder with valid code systems.

Research prototype only -- not a medical device.
"""

__version__ = "0.3.0"
