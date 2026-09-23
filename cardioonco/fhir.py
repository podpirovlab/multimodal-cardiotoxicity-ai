"""HL7 FHIR R4 DiagnosticReport builder.

Only real, published code systems are used for standard concepts:
  * category  -> HL7 v2 table 0074 "EC" (Electrocardiac: EKG, Holter)
  * code      -> LOINC 11524-6 "EKG study"
  * units     -> UCUM ("uV", "/min", "1")
Project-specific measurements (TWA K-score, model probabilities) have no standard
LOINC code, so they are coded in a clearly-labelled *local* CodeSystem instead of
inventing codes that look official.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

LOCAL_CS = "https://github.com/podpirovlab/multimodal-cardiotoxicity-ai/fhir/CodeSystem/cardioonco"
UCUM = "http://unitsofmeasure.org"


def _obs(local_id: str, code: str, display: str, value: float, unit: str, ucum: str) -> dict:
    return {
        "resourceType": "Observation",
        "id": local_id,
        "status": "final",
        "code": {"coding": [{"system": LOCAL_CS, "code": code, "display": display}], "text": display},
        "valueQuantity": {"value": round(float(value), 4), "unit": unit, "system": UCUM, "code": ucum},
    }


def diagnostic_report(patient_ref: str, twa: dict | None = None, probs: dict | None = None,
                      conclusion: str = "") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    contained, results = [], []

    def add(o):
        contained.append(o)
        results.append({"reference": f"#{o['id']}"})

    if twa:
        add(_obs("hr", "heart-rate", "Heart rate (from R-R intervals)", twa["heart_rate_bpm"], "/min", "/min"))
        add(_obs("valt", "twa-valt", "T-wave alternans voltage (Spectral Method)", twa["v_alt_uv"], "uV", "uV"))
        add(_obs("kscore", "twa-k", "T-wave alternans K-score", twa["k_score"], "1", "1"))
        add(_obs("mma", "twa-mma", "T-wave alternans (Modified Moving Average)", twa["mma_uv"], "uV", "uV"))
    for cls, p in (probs or {}).items():
        add(_obs(f"p-{cls.lower()}", f"prob-{cls.lower()}", f"Model probability: {cls}", p, "1", "1"))

    return {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "meta": {"tag": [{"system": LOCAL_CS, "code": "research-only",
                          "display": "Research prototype output - not for clinical use"}]},
        "contained": contained,
        "status": "preliminary",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                                  "code": "EC", "display": "Electrocardiac (e.g., EKG, EEC, Holter)"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "11524-6", "display": "EKG study"}],
                 "text": "CardioOncoPredict ECG analysis (research prototype)"},
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now,
        "issued": now,
        "result": results,
        "conclusion": conclusion or "Research prototype output. Not validated for clinical decision-making.",
    }
