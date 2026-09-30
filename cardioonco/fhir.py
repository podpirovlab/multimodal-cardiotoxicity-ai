"""HL7 FHIR R4 DiagnosticReport builder -- a research-format demo, not a clinical record.

Only real, published code systems are used for standard concepts:
  * category  -> HL7 v2 table 0074 "EC" (Electrocardiac: EKG, Holter)
  * code      -> LOINC 11524-6 "EKG study"; heart rate -> LOINC 8867-4
  * units     -> UCUM ("uV", "/min", "1")
Project-specific measurements (TWA K-score, model probabilities) have no standard
LOINC code, so they are coded in a clearly-labelled *local* CodeSystem instead of
inventing codes that look official.  Every Observation points to a contained Device
that carries the algorithm version, so a number can always be traced to the code
that produced it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from . import __version__

LOCAL_CS = "https://github.com/podpirovlab/multimodal-cardiotoxicity-ai/fhir/CodeSystem/cardioonco"
UCUM = "http://unitsofmeasure.org"
LOINC = "http://loinc.org"
RESEARCH_PREFIX = "RESEARCH USE ONLY - NOT FOR CLINICAL RECORDS. "


def _obs(local_id: str, code: str, display: str, value: float, unit: str, ucum: str,
         loinc: str | None = None) -> dict:
    coding = [{"system": LOCAL_CS, "code": code, "display": display}]
    if loinc:
        coding.insert(0, {"system": LOINC, "code": loinc, "display": display})
    return {
        "resourceType": "Observation",
        "id": local_id,
        "status": "preliminary",
        "code": {"coding": coding, "text": display},
        "device": {"reference": "#software"},
        "valueQuantity": {"value": round(float(value), 4), "unit": unit, "system": UCUM, "code": ucum},
    }


def _coded_obs(local_id: str, code: str, display: str, value_code: str, text: str) -> dict:
    return {
        "resourceType": "Observation",
        "id": local_id,
        "status": "preliminary",
        "code": {"coding": [{"system": LOCAL_CS, "code": code, "display": display}], "text": display},
        "device": {"reference": "#software"},
        "valueCodeableConcept": {"coding": [{"system": LOCAL_CS, "code": value_code, "display": value_code}],
                                 "text": text},
    }


def software_device() -> dict:
    return {
        "resourceType": "Device",
        "id": "software",
        "deviceName": [{"name": "CardioOncoPredict TWA analysis (research software)", "type": "model-name"}],
        "version": [{"value": __version__}],
    }


def diagnostic_report(patient_ref: str | None = None, twa: dict | None = None, probs: dict | None = None,
                      conclusion: str = "", recorded_at: str | None = None) -> dict:
    """Build the report.  `recorded_at` is the ECG recording time (ISO 8601) if known --
    never the export time; when unknown, effective[x] is simply left out."""
    now = datetime.now(timezone.utc).isoformat()
    contained, results = [software_device()], []

    def add(o):
        contained.append(o)
        results.append({"reference": f"#{o['id']}"})

    if twa:
        add(_obs("hr", "heart-rate", "Heart rate (from R-R intervals)", twa["heart_rate_bpm"], "/min", "/min",
                 loinc="8867-4"))
        add(_obs("valt", "twa-valt", "T-wave alternans voltage (Spectral Method)", twa["v_alt_uv"], "uV", "uV"))
        add(_obs("kscore", "twa-k", "T-wave alternans K-score", twa["k_score"], "1", "1"))
        add(_obs("mma", "twa-mma", "T-wave alternans (Modified Moving Average)", twa["mma_uv"], "uV", "uV"))
        if "outcome" in twa:
            add(_coded_obs("outcome", "twa-outcome", "T-wave alternans outcome (Spectral Method rules)",
                           twa["outcome"], twa.get("reason", "")))
    for cls, p in (probs or {}).items():
        add(_obs(f"p-{cls.lower()}", f"prob-{cls.lower()}", f"Model probability: {cls}", p, "1", "1"))

    report = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "meta": {"tag": [{"system": LOCAL_CS, "code": "research-only",
                          "display": "Research prototype output - not for clinical use"}]},
        "contained": contained,
        "status": "preliminary",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                                  "code": "EC", "display": "Electrocardiac (e.g., EKG, EEC, Holter)"}]}],
        "code": {"coding": [{"system": LOINC, "code": "11524-6", "display": "EKG study"}],
                 "text": "CardioOncoPredict ECG analysis (research prototype)"},
        "issued": now,
        "result": results,
        "conclusion": RESEARCH_PREFIX + (conclusion or "Not validated for clinical decision-making."),
    }
    if patient_ref:
        report["subject"] = {"reference": patient_ref}
    if recorded_at:
        report["effectiveDateTime"] = recorded_at
    return report
